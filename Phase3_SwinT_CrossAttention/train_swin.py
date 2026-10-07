#!/usr/bin/env python3
"""
train_swin.py  —  Phase 3: Swin-T + U-Net Cross-Attention Architecture

Features:
  1. Swin-T Encoder (pre-trained, 4-stage feature pyramid)
  2. CrossAttentionSkip in UNet (replaces standard concatenation)
  3. Mixed precision (bf16) for main model
  4. Float32 isolated for compute_physics_loss
  5. Expandable segments memory management
  6. Batch size 4, 100 epochs
"""

import os, sys, glob, json, time, argparse
import numpy as np
from pathlib import Path
import math

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

DATA_DIR    = ROOT / "data" / "processed" / "patches_6ch"
RESULTS_DIR = ROOT / "results" / "phase3_swin"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ── Dataset ───────────────────────────────────────────────────────
class PatchDataset(Dataset):
    def __init__(self, split="train"):
        d = DATA_DIR / split
        self.inp = sorted(glob.glob(str(d / "input"  / "*.npz")))
        self.tgt = sorted(glob.glob(str(d / "target" / "*.npz")))
    def __len__(self): return len(self.inp)
    def __getitem__(self, idx):
        with np.load(self.inp[idx]) as di: x = di['image'].astype(np.float32)
        with np.load(self.tgt[idx]) as dt: y = dt['image'].astype(np.float32)
        if y.ndim == 2: y = y[None]
        return torch.from_numpy(x), torch.from_numpy(y)

# ── Metrics ───────────────────────────────────────────────────────
def compute_metrics(preds, targets, inputs):
    try:
        from skimage.metrics import structural_similarity as ski_ssim
        has_ski = True
    except ImportError:
        has_ski = False
    p = preds.astype(np.float64).ravel()
    t = targets.astype(np.float64).ravel()
    mae  = float(np.mean(np.abs(p - t)))
    rmse = float(np.sqrt(np.mean((p - t)**2)))
    r2   = float(1 - np.sum((p-t)**2) / (np.sum((t - t.mean())**2) + 1e-10))
    ssim = 0.0
    if has_ski:
        for i in range(preds.shape[0]):
            ssim += ski_ssim(preds[i,0].astype(np.float64),
                             targets[i,0].astype(np.float64), data_range=1.0)
        ssim /= preds.shape[0]
    chm_mean = inputs[:, 7, :, :].mean(axis=(1, 2))
    bio_mean = preds[:, 0, :, :].mean(axis=(1, 2))
    idx = np.argsort(chm_mean)
    diffs = np.diff(bio_mean[idx])
    pvr = float(np.mean(diffs < 0))
    mnd = float(np.mean(np.clip(-diffs, 0, None)))
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "SSIM": float(ssim), "PVR": pvr, "MND": mnd}

def physics_lambda(epoch, warmup=30, total=100, max_lam=0.5):
    if epoch <= warmup: return 0.0
    return max_lam * min(1.0, (epoch - warmup) / max(1, total - warmup))

# ── Cross-Attention Module ─────────────────────────────────────────
class CrossAttentionSkip(nn.Module):
    def __init__(self, decoder_channels, encoder_channels, num_heads=8):
        super().__init__()
        self.k_proj = nn.Conv2d(encoder_channels, decoder_channels, 1)
        self.v_proj = nn.Conv2d(encoder_channels, decoder_channels, 1)
        self.attn = nn.MultiheadAttention(decoder_channels, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(decoder_channels)
    
    def forward(self, decoder_feat, encoder_feat):
        B, C, H, W = decoder_feat.shape
        if encoder_feat.shape[-2:] != (H, W):
            encoder_feat = F.interpolate(encoder_feat, size=(H, W), mode='bilinear', align_corners=False)
        k = self.k_proj(encoder_feat).flatten(2).permute(0,2,1)
        v = self.v_proj(encoder_feat).flatten(2).permute(0,2,1)
        q = decoder_feat.flatten(2).permute(0,2,1)
        out, _ = self.attn(q, k, v)
        out = self.norm(out + q)
        return out.permute(0,2,1).reshape(B, C, H, W)

# ── Modified UpBlock with Cross-Attention ──────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU()
        )
    def forward(self, x):
        return self.double_conv(x)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, t):
        x = self.conv(x)
        x = x + self.time_mlp(t)[..., None, None]
        return x, self.pool(x)

class UpBlockCA(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, time_emb_dim, enc_ch=None):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.ca = CrossAttentionSkip(decoder_channels=in_channels // 2, encoder_channels=enc_ch if enc_ch else skip_channels)
        self.conv = DoubleConv(in_channels // 2, out_channels)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))

    def forward(self, x, skip_enc, t):
        x = self.up(x)
        x = self.ca(x, skip_enc)
        x = self.conv(x)
        return x + self.time_mlp(t)[..., None, None]

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

class SwinUNetCA(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, time_dim=512):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        self.down1 = DownBlock(in_channels, 128, time_dim)
        self.down2 = DownBlock(128, 256, time_dim)
        self.down3 = DownBlock(256, 512, time_dim)
        self.down4 = DownBlock(512, 1024, time_dim)
        self.bot1 = DoubleConv(1024, 1024)
        self.bot2 = DoubleConv(1024, 1024)
        
        # Swin-T channels [96, 192, 384, 768]
        self.up1 = UpBlockCA(1024, 512, 512, time_dim, enc_ch=768)
        self.up2 = UpBlockCA(512, 256, 256, time_dim, enc_ch=384)
        self.up3 = UpBlockCA(256, 128, 128, time_dim, enc_ch=192)
        self.up4 = UpBlockCA(128, 64,  128, time_dim, enc_ch=96)
        self.final_conv = nn.Conv2d(128, out_channels, kernel_size=1)

    def forward(self, xt, t, swin_feats):
        t_emb = self.time_mlp(t)
        x1, p1 = self.down1(xt, t_emb)
        x2, p2 = self.down2(p1, t_emb)
        x3, p3 = self.down3(p2, t_emb)
        x4, p4 = self.down4(p3, t_emb)
        b = self.bot2(self.bot1(p4))
        u1 = self.up1(b,  swin_feats[3], t_emb)
        u2 = self.up2(u1, swin_feats[2], t_emb)
        u3 = self.up3(u2, swin_feats[1], t_emb)
        u4 = self.up4(u3, swin_feats[0], t_emb)
        return self.final_conv(u4)

# ── Swin PILDM Wrapper ────────────────────────────────────────────
class SwinPILDM(nn.Module):
    def __init__(self, vae, encoder, unet, diffusion_scheduler):
        super().__init__()
        self.vae = vae
        self.encoder = encoder
        self.unet = unet
        self.scheduler = diffusion_scheduler
        from src.models.implicit_repr import SIRENINR
        # Pass x_cond (8 channels) directly into INR to preserve physics gradient
        self.inr = SIRENINR(student_chs=[96, 192, 384, 768, 8, 4])
        for p in self.vae.parameters():
            p.requires_grad = False

    def forward(self, x_cond, y_target, t, noise=None):
        swin_feats, projected_feats = self.encoder(x_cond)
        with torch.no_grad():
            z_0, _ = self.vae.encode(y_target)
        z_t, noise_true = self.scheduler.q_sample(z_0, t, noise)
        noise_pred = self.unet(z_t, t, swin_feats)
        loss_diff = F.l1_loss(noise_pred, noise_true)
        # Direct x_cond injection preserves gradient tracking
        y_pred = self.inr(swin_feats + [x_cond], z_0)
        loss_recon = F.mse_loss(y_pred, y_target)
        return loss_diff, loss_recon, projected_feats, y_pred

    def decode(self, x_cond, num_steps=50):
        B = x_cond.shape[0]
        device = x_cond.device
        swin_feats, _ = self.encoder(x_cond)
        z_T = torch.randn(B, 4, 32, 32, device=device)
        z_0_pred = self.scheduler.ddim_sample(self.unet, z_T.shape, swin_feats, steps=num_steps)
        # Direct x_cond injection preserves gradient tracking
        y_pred = self.inr(swin_feats + [x_cond], z_0_pred)
        return y_pred
    
    def predict(self, x_cond, num_steps=50):
        with torch.no_grad():
            return self.decode(x_cond, num_steps)

from src.models.diffusion import DiffusionScheduler

def patch_physics_loss(x_batch, pred_batch):
    chm  = x_batch[:, 7].mean(dim=[1, 2])
    carb = pred_batch[:, 0].mean(dim=[1, 2])
    dchm  = chm.unsqueeze(1)  - chm.unsqueeze(0)
    dcarb = carb.unsqueeze(1) - carb.unsqueeze(0)
    return F.relu(-dcarb * dchm.sign()).mean()

def run_swin_training(epochs=100, bs=4, lr=1e-4, grad_accum=4):
    device = torch.device("cuda")
    out_dir = RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best_swin.pt"
    
    print(f"\n{'='*68}")
    print(f"  PHASE 3: Swin-T + U-Net Cross Attention")
    print(f"  Epochs: {epochs} | BS: {bs} | GradAccum: {grad_accum}")
    print(f"{'='*68}\n")

    import timm
    class SwinEncoderWithSkipWrap(nn.Module):
        def __init__(self, in_chans=8):
            super().__init__()
            self.swin = timm.create_model(
                'swin_tiny_patch4_window7_224',
                pretrained=True,
                in_chans=in_chans,
                img_size=256,
                num_classes=0,
                features_only=True,
                out_indices=(0, 1, 2, 3)
            )
            # Projection layers to match Teacher (VGG16) KD features
            self.proj_out = nn.ModuleList([
                nn.Conv2d(c_s, c_t, 1, bias=False)
                for c_s, c_t in zip([96, 192, 384, 768], [64, 128, 256, 512])
            ])
            
        def forward(self, x):
            feats = self.swin(x)
            student_feats = [f.permute(0, 3, 1, 2).contiguous() for f in feats]
            projected_feats = [self.proj_out[i](student_feats[i]) for i in range(4)]
            return student_feats, projected_feats

    from src.models.vae import CarbonVAE
    vae = CarbonVAE(in_channels=1, latent_channels=4, base_channels=64).to(device)
    vae_ckpt = ROOT / "results" / "checkpoints" / "vae_best.pt"
    if vae_ckpt.exists():
        vae.load_state_dict(torch.load(vae_ckpt, map_location=device))
        print("  Loaded pretrained VAE ✓")
        
    encoder = SwinEncoderWithSkipWrap(in_chans=8).to(device)
    unet = SwinUNetCA(in_channels=4, out_channels=4, time_dim=512).to(device)
    sched = DiffusionScheduler(T=1000, device=str(device).replace(":0",""))
    
    model = SwinPILDM(vae, encoder, unet, sched).to(device)
    
    # Initialize KD Teacher
    from src.models.kd_vgg import VGG16Teacher, HierarchicalKDLoss
    teacher = VGG16Teacher(in_channels=8).to(device)
    teacher.eval()
    kd_criterion = HierarchicalKDLoss().to(device)
    
    if torch.cuda.device_count() > 1:
        print(f"  Using DataParallel across {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
        teacher = nn.DataParallel(teacher)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"  Trainable params: {sum(p.numel() for p in trainable)/1e6:.2f}M\n")

    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    tr_loader = DataLoader(PatchDataset("train"), bs, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    va_loader = DataLoader(PatchDataset("val"), bs, shuffle=False, num_workers=4, pin_memory=True)

    best_val, t0 = float('inf'), time.time()
    
    start_ep = 1
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model_module = model.module if isinstance(model, nn.DataParallel) else model
        model_module.load_state_dict(state['model'], strict=True)
        start_ep = state.get('epoch', 0) + 1
        opt.load_state_dict(state['opt'])
        print(f"  Resuming from epoch {start_ep} ✓")

    for ep in range(start_ep, epochs + 1):
        model.train()
        torch.cuda.empty_cache()
        lam_phys = physics_lambda(ep, warmup=30, total=epochs, max_lam=0.5)
        ep_diff = ep_recon = ep_phys = 0.0
        opt.zero_grad()

        for step, (x, y) in enumerate(tr_loader):
            x, y = x.to(device), y.to(device)
            t_rand = torch.randint(0, 1000, (x.shape[0],), device=device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss_diff, loss_recon, projected_feats, y_pred = model(x, y, t_rand)
                # Boost loss_recon to directly minimize RMSE
                loss = loss_diff.mean() + 1.0 * loss_recon.mean()
            
            ep_diff += loss_diff.mean().item()
            ep_recon += loss_recon.mean().item()

            if lam_phys > 0:
                l_phys = patch_physics_loss(x.float(), y_pred.float())
                loss = loss + lam_phys * l_phys
                ep_phys += l_phys.item()
                del l_phys

            scaler.scale(loss / grad_accum).backward()

            if (step + 1) % grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                torch.cuda.empty_cache()

        N = max(1, len(tr_loader))
        cosine.step()

        if ep % 5 == 0 or ep == epochs:
            model.eval()
            vl = 0.0
            with torch.no_grad():
                for xv, yv in va_loader:
                    xv, yv = xv.to(device), yv.to(device)
                    tv = torch.randint(0, 1000, (xv.shape[0],), device=device)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        ld, lr_, _, _ = model(xv, yv, tv)
                    vl += (ld.mean() + 0.1 * lr_.mean()).item()
            vl /= max(1, len(va_loader))
            elapsed  = (time.time() - t0) / 60
            eta      = elapsed / ep * (epochs - ep) if ep > 0 else 0
            
            lr_now = opt.param_groups[0]['lr']
            print(f"  Ep {ep:3d}/{epochs} | Ldiff={ep_diff/N:.4f} Lrecon={ep_recon/N:.4f} Lphys={ep_phys/N:.4f} | "
                  f"Lam={lam_phys:.2f} LR={lr_now:.2e} | Val={vl:.4f} | {elapsed:.0f}m ETA={eta:.0f}m")
            
            if vl < best_val:
                best_val = vl
                model_module = model.module if isinstance(model, nn.DataParallel) else model
                torch.save({'epoch': ep, 'model': model_module.state_dict(), 'val': vl, 'opt': opt.state_dict()}, ckpt_path)
                print(f"  ✔  Best ckpt saved (ep={ep}, val={vl:.4f})")

    # Final eval
    print("\n  Loading best checkpoint for final evaluation ...")
    state = torch.load(ckpt_path, map_location=device)
    model_module = model.module if isinstance(model, nn.DataParallel) else model
    model_module.load_state_dict(state['model'], strict=True)
    model.eval()

    test_loader = DataLoader(PatchDataset("test"), bs, shuffle=False, num_workers=4)
    all_p, all_t, all_i = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            torch.cuda.empty_cache()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = model_module.predict(x.to(device), num_steps=20).clamp(0,1).float()
            all_p.append(pred.cpu().numpy())
            all_t.append(y.numpy())
            all_i.append(x.numpy())

    metrics = compute_metrics(np.concatenate(all_p), np.concatenate(all_t), np.concatenate(all_i))
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*68}")
    print(f"  PHASE 3 SWIN-T FINAL RESULTS")
    for k, v in metrics.items():
        print(f"    {k:6s}: {v:.6f}")
    print(f"{'='*68}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--bs", type=int, default=4)
    args = parser.parse_args()
    run_swin_training(epochs=args.epochs, bs=args.bs)
