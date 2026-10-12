#!/usr/bin/env python3
"""
train_swin_no_eth.py  —  Modality Ablation: Swin-T + Cross-Attention w/o ETH Canopy Height (7 channels)
Ablation dropping Channel 7 (ETH Canopy Height) to evaluate whether model retains high accuracy
without reference circularity.
"""

import os, sys, glob, json, time, argparse
import numpy as np
from pathlib import Path
import math

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

DATA_DIR    = ROOT / "data" / "processed" / "patches_6ch"
RESULTS_DIR = ROOT / "results" / "phase3_swin_no_eth"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ── Dataset (7 channels: Drop Channel 7 = ETH Canopy Height) ────────
class PatchDataset(Dataset):
    def __init__(self, split="train"):
        d = DATA_DIR / split
        self.inp = sorted(glob.glob(str(d / "input"  / "*.npz")))
        self.tgt = sorted(glob.glob(str(d / "target" / "*.npz")))
    def __len__(self): return len(self.inp)
    def __getitem__(self, idx):
        with np.load(self.inp[idx]) as di: x = di['image'][:7].astype(np.float32)
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
    dem_mean = inputs[:, 6, :, :].mean(axis=(1, 2))
    bio_mean = preds[:, 0, :, :].mean(axis=(1, 2))
    idx = np.argsort(dem_mean)
    diffs = np.diff(bio_mean[idx])
    pvr = float(np.mean(diffs < 0))
    mnd = float(np.mean(np.clip(-diffs, 0, None)))
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "SSIM": float(ssim), "PVR": pvr, "MND": mnd}

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

# ── UNet with Cross-Attention Skips ─────────────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU()
        )
    def forward(self, x):
        return self.conv(x)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))

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

# ── Swin PILDM Wrapper (7 Channels) ──────────────────────────────
class SwinPILDM(nn.Module):
    def __init__(self, vae, encoder, unet, diffusion_scheduler):
        super().__init__()
        self.vae = vae
        self.encoder = encoder
        self.unet = unet
        self.scheduler = diffusion_scheduler
        from src.models.implicit_repr import SIRENINR
        # Pass x_cond (7 channels) directly into INR
        self.inr = SIRENINR(student_chs=[96, 192, 384, 768, 7, 4])
        for p in self.vae.parameters():
            p.requires_grad = False

    def forward(self, x_cond, y_target, t, noise=None):
        swin_feats, projected_feats = self.encoder(x_cond)
        with torch.no_grad():
            z_0, _ = self.vae.encode(y_target)
        z_t, noise_true = self.scheduler.q_sample(z_0, t, noise)
        noise_pred = self.unet(z_t, t, swin_feats)
        loss_diff = F.mse_loss(noise_pred, noise_true)
        
        # 1. Mixed Curriculum (Covariate Shift Fix)
        if self.training and torch.rand(1).item() < 0.5:
            alpha_prod_t = self.scheduler.alpha_bar.to(z_t.device)[t].view(-1, 1, 1, 1)
            pred_z0 = (z_t - (1 - alpha_prod_t) ** 0.5 * noise_pred) / alpha_prod_t ** 0.5
            inr_z = torch.clamp(pred_z0, min=-5.0, max=5.0).detach()
        else:
            inr_z = z_0

        # Direct x_cond (7 ch) injection
        y_pred = self.inr(swin_feats + [x_cond], inr_z)
        
        # 2. Support-Aware Supervision (Footprint-Integrated Loss: 3x3 avg pool)
        y_pred_footprint = F.avg_pool2d(y_pred, kernel_size=3, stride=1, padding=1)
        loss_recon = F.mse_loss(y_pred_footprint, y_target)
        
        return loss_diff, loss_recon, projected_feats, y_pred

    def decode(self, x_cond, num_steps=50):
        B = x_cond.shape[0]
        device = x_cond.device
        swin_feats, _ = self.encoder(x_cond)
        z_T = torch.randn(B, 4, 32, 32, device=device)
        z_0_pred = self.scheduler.ddim_sample(self.unet, z_T.shape, swin_feats, steps=num_steps)
        y_pred = self.inr(swin_feats + [x_cond], z_0_pred)
        return y_pred
    
    def predict(self, x_cond, num_steps=50):
        with torch.no_grad():
            return self.decode(x_cond, num_steps)

from src.models.diffusion import DiffusionScheduler

def run_swin_training(epochs=100, bs=4, lr=1e-4, grad_accum=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = RESULTS_DIR / "best_swin_no_eth.pt"

    print(f"\n{'='*68}")
    print(f"  MODALITY ABLATION: Swin-T w/o ETH Canopy Height (7 channels)")
    print(f"  Epochs: {epochs} | BS: {bs} | GradAccum: {grad_accum}")
    print(f"{'='*68}\n")

    import timm
    class SwinEncoderWithSkipWrap(nn.Module):
        def __init__(self, in_chans=7):
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
        
    encoder = SwinEncoderWithSkipWrap(in_chans=7).to(device)
    unet = SwinUNetCA(in_channels=4, out_channels=4, time_dim=512).to(device)
    sched = DiffusionScheduler(T=1000, device=str(device).replace(":0",""))
    
    model = SwinPILDM(vae, encoder, unet, sched).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"  Trainable params: {sum(p.numel() for p in trainable)/1e6:.2f}M\n")

    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    tr_loader = DataLoader(PatchDataset("train"), bs, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    va_loader = DataLoader(PatchDataset("val"), bs, shuffle=False, num_workers=4, pin_memory=True)

    best_val, t0 = float('inf'), time.time()

    for ep in range(1, epochs + 1):
        model.train()
        torch.cuda.empty_cache()
        ep_diff = ep_recon = 0.0
        opt.zero_grad()

        for step, (x, y) in enumerate(tr_loader):
            x, y = x.to(device), y.to(device)
            t_rand = torch.randint(0, 1000, (x.shape[0],), device=device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss_diff, loss_recon, projected_feats, y_pred = model(x, y, t_rand)
                loss = loss_diff.mean() + 1.0 * loss_recon.mean()

            scaler.scale(loss / grad_accum).backward()

            if (step + 1) % grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                torch.cuda.empty_cache()

            ep_diff += loss_diff.mean().item()
            ep_recon += loss_recon.mean().item()

        cosine.step()
        N = max(1, len(tr_loader))

        if ep % 5 == 0 or ep == epochs:
            model.eval()
            vl = 0.0
            with torch.no_grad():
                for xv, yv in va_loader:
                    xv, yv = xv.to(device), yv.to(device)
                    tv = torch.randint(0, 1000, (xv.shape[0],), device=device)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        ld, lr_recon, _, _ = model(xv, yv, tv)
                    vl += (ld.mean() + lr_recon.mean()).item()
            vl /= max(1, len(va_loader))
            elapsed = (time.time() - t0) / 60
            eta = elapsed / ep * (epochs - ep) if ep > 0 else 0
            free_now = torch.cuda.mem_get_info(0)[0] / 1024**3
            lr_now = opt.param_groups[0]['lr']
            print(f"  Ep {ep:3d}/{epochs} | Ldiff={ep_diff/N:.4f} Lrecon={ep_recon/N:.4f} | LR={lr_now:.2e} | Val={vl:.4f} | {elapsed:.0f}m ETA={eta:.0f}m | Free={free_now:.1f}GB")
            
            if vl < best_val:
                best_val = vl
                model_module = model.module if isinstance(model, nn.DataParallel) else model
                torch.save({'epoch': ep, 'model': model_module.state_dict(), 'opt': opt.state_dict(), 'val': vl}, ckpt_path)
                print(f"  ✔ Best ckpt saved (ep={ep}, val={vl:.4f})")

    # ── Final Test Evaluation ──────────────────────────────────
    print("\nRunning final test evaluation ...")
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
    metrics["RMSE_MgCha"] = metrics["RMSE"] * 130
    metrics["MAE_MgCha"] = metrics["MAE"] * 130
    
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\n{'='*68}")
    print(f"  MODALITY ABLATION (w/o ETH Canopy Height) FINAL RESULTS:")
    for k, v in metrics.items():
        print(f"    {k:6s}: {v:.6f}")
    print(f"{'='*68}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--bs", type=int, default=4)
    args = parser.parse_args()
    run_swin_training(epochs=args.epochs, bs=args.bs)
