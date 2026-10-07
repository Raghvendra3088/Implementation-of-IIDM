#!/usr/bin/env python3
"""
run_finetuned.py  —  Re-run P1 and P1-KD with 4 fine-tuning improvements,
using EXACTLY the same model construction as run_all_baselines.py.

Improvements:
  1. Curriculum physics: lambda_phys ramps 0→0.5 over epochs 50-200
  2. Patch-level physics loss: patch-to-patch CHM monotonicity
  3. Mixed precision (bf16): allows bs=4 for P1, bs=2 for P1-KD  
  4. Extended training: 200 epochs with cosine decay to 1e-6
  5. Warm-start: loads from best baseline checkpoint

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 run_finetuned.py --exp P1_FT   --epochs 200
  CUDA_VISIBLE_DEVICES=4 python3 run_finetuned.py --exp P1KD_FT --epochs 200
"""

import os, sys, glob, json, time, argparse
import numpy as np
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

DATA_DIR    = ROOT / "data" / "processed" / "patches_6ch"
RESULTS_DIR = ROOT / "results" / "finetuned"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ── Same encoder as run_all_baselines.py ─────────────────────────
def _build_student_cond_encoder(in_ch=8):
    """Exact copy from run_all_baselines.py — do not change."""
    class StudentEncoder(nn.Module):
        def __init__(self, in_ch):
            super().__init__()
            self.b1 = nn.Sequential(nn.Conv2d(in_ch,64,3,padding=1),   nn.GroupNorm(8,64),  nn.SiLU())
            self.b2 = nn.Sequential(nn.Conv2d(64,128,3,stride=2,padding=1), nn.GroupNorm(8,128), nn.SiLU())
            self.b3 = nn.Sequential(nn.Conv2d(128,256,3,stride=2,padding=1),nn.GroupNorm(8,256), nn.SiLU())
            self.b4 = nn.Sequential(nn.Conv2d(256,512,3,stride=2,padding=1),nn.GroupNorm(8,512), nn.SiLU())
        def forward(self, x):
            f1=self.b1(x); f2=self.b2(f1); f3=self.b3(f2); f4=self.b4(f3)
            fl=[f1,f2,f3,f4]
            return fl, fl, None
    return StudentEncoder(in_ch)

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

# ── Curriculum physics ────────────────────────────────────────────
def physics_lambda(epoch, warmup=50, total=200, max_lam=0.5):
    if epoch <= warmup: return 0.0
    return max_lam * min(1.0, (epoch - warmup) / max(1, total - warmup))

# ── Patch-level physics loss ──────────────────────────────────────
def patch_physics_loss(x_batch, pred_batch):
    """Penalise patch-to-patch violations: higher CHM → lower carbon."""
    chm  = x_batch[:, 7].mean(dim=[1, 2])
    carb = pred_batch[:, 0].mean(dim=[1, 2])
    dchm  = chm.unsqueeze(1)  - chm.unsqueeze(0)
    dcarb = carb.unsqueeze(1) - carb.unsqueeze(0)
    return F.relu(-dcarb * dchm.sign()).mean()

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


def run_finetuned(exp_name, use_kd, epochs=200, bs=4, lr=5e-5,
                  grad_accum=4, lambda_kd=0.05, lambda_patch=0.3, ddim_steps=20):

    device = torch.device("cuda:0")
    out_dir   = RESULTS_DIR / exp_name
    ckpt_path = out_dir / "best.pt"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Map to baseline checkpoint for warm-start
    baseline_map = {
        "P1_FT":   ROOT / "results" / "baselines" / "P1_PILDM"     / "best.pt",
        "P1KD_FT": ROOT / "results" / "baselines" / "P1KD_PILDM_KD" / "best.pt",
    }
    baseline_ckpt = baseline_map.get(exp_name)
    baseline_metrics = {
        "P1_FT":   {"MAE":0.0756,"RMSE":0.1059,"R2":-0.365,"SSIM":0.350,"PVR":0.510,"MND":0.0043},
        "P1KD_FT": {"MAE":0.0849,"RMSE":0.1242,"R2":-0.878,"SSIM":0.184,"PVR":0.000,"MND":0.000},
    }

    free_gb  = torch.cuda.mem_get_info(0)[0] / 1024**3
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\n{'='*68}")
    print(f"  FINE-TUNE: {exp_name}  |  use_kd={use_kd}")
    print(f"  GPU: {torch.cuda.get_device_name(0)} | {free_gb:.1f}GB free / {total_gb:.0f}GB total")
    print(f"  Improvements: curriculum_phys + patch_phys + bf16 + {epochs}ep")
    print(f"{'='*68}\n")

    if (out_dir / "metrics.json").exists():
        print("  Already DONE — cached metrics:")
        m = json.load(open(out_dir / "metrics.json"))
        for k, v in m.items(): print(f"    {k}: {v:.6f}")
        return m

    from src.models.vae import CarbonVAE
    from src.models.kd_unet import KDUNet
    from src.models.diffusion import DiffusionScheduler
    from src.models.pi_ldm import PILDM, compute_physics_loss

    vae  = CarbonVAE(in_channels=1, latent_channels=4, base_channels=64).to(device)
    unet = KDUNet(in_channels=4, out_channels=4, context_dim=256).to(device)
    # Use string device — avoids linspace TypeError
    sched = DiffusionScheduler(T=1000, device=str(device).replace(":0",""))
    for p in vae.parameters(): p.requires_grad = False

    if use_kd:
        from src.models.kd_vgg import VGG19_Adapter
        encoder = VGG19_Adapter(in_channels=8).to(device)
        for p in encoder.parameters(): p.requires_grad = False
        for name, p in encoder.named_parameters():
            if name.startswith("proj_in") or name.startswith("proj_out"):
                p.requires_grad = True
        vgg_t = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        print(f"  VGG19 frozen | {vgg_t/1e3:.0f}K adapter params trainable")
    else:
        encoder = _build_student_cond_encoder(8).to(device)

    model = PILDM(vae, encoder, unet, sched, freeze_encoder=False).to(device)

    # Load VAE pretrained if available
    vae_ckpt = ROOT / "results" / "checkpoints" / "vae_best.pt"
    if vae_ckpt.exists():
        model.vae.load_state_dict(torch.load(vae_ckpt, map_location=device))
        print("  Loaded pretrained VAE ✓")

    # Warm-start: ONLY load VAE pretrained weights (avoid architecture mismatch)
    # Do NOT load full PILDM checkpoint — INR/encoder channel mismatch causes RuntimeError
    start_ep = 1
    if ckpt_path.exists():
        # Resume our own fine-tune checkpoint (architecture matches exactly)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state['model'], strict=True)
        start_ep = state.get('epoch', 0) + 1
        print(f"  Resuming fine-tune from epoch {start_ep} ✓")
    elif baseline_ckpt and baseline_ckpt.exists():
        print(f"  Warm-starting from Baseline {exp_name} checkpoint...")
        state = torch.load(baseline_ckpt, map_location=device)
        # Filter out INR weights to avoid architecture mismatch with INR v2
        state_dict = {k: v for k, v in state['model'].items() if not k.startswith('inr.')}
        model.load_state_dict(state_dict, strict=False)
        start_ep = 1
        print(f"  Loaded UNet and Encoder from baseline. INR will train from scratch. ✓")
    else:
        print(f"  No existing ckpt — training fresh from VAE warm-start")

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params  = sum(p.numel() for p in trainable)
    print(f"  Trainable params: {n_params/1e6:.2f}M\n")

    opt    = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    tr_loader = DataLoader(PatchDataset("train"), bs, shuffle=True,
                           num_workers=4, pin_memory=True, drop_last=True)
    va_loader = DataLoader(PatchDataset("val"),   bs, shuffle=False,
                           num_workers=4, pin_memory=True)

    best_val, t0 = float('inf'), time.time()

    for ep in range(start_ep, epochs + 1):
        model.train()
        torch.cuda.empty_cache()
        lam_phys = physics_lambda(ep, warmup=50, total=epochs, max_lam=0.5)
        ep_diff = ep_recon = ep_phys = ep_patch = 0.0
        opt.zero_grad()

        for step, (x, y) in enumerate(tr_loader):
            x, y = x.to(device), y.to(device)
            t_rand = torch.randint(0, 1000, (x.shape[0],), device=device)

            # ── Main diffusion loss (bf16) ────────────────────────
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss_diff, loss_recon, f_proj, y_pred = model(x, y, t_rand)
                loss = loss_diff + 1.0 * loss_recon
            ep_diff  += loss_diff.item()
            ep_recon += loss_recon.item()

            # ── KD feature regularisation (mild, after curriculum warmup) ─
            if use_kd and f_proj is not None and isinstance(f_proj, (list, tuple)):
                kd_loss = sum(fi.float().pow(2).mean() * 0.001 for fi in f_proj)
                loss = loss + lambda_kd * kd_loss

            # ── Pixel-level physics loss (only after curriculum warmup) ──
            if lam_phys > 0:
                try:
                    x_phys = x.detach().float().requires_grad_(True)
                    l_phys = compute_physics_loss(model, x_phys)
                    loss = loss + lam_phys * l_phys
                    ep_phys += l_phys.item()
                    del x_phys, l_phys
                except (torch.cuda.OutOfMemoryError, RuntimeError):
                    torch.cuda.empty_cache()

            # ── Patch-level physics loss ──────────────────────────
            if lam_phys > 0:
                l_patch = patch_physics_loss(x.float(), y_pred.float())
                loss = loss + lambda_patch * l_patch
                ep_patch += l_patch.item()
                del l_patch

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
                    vl += (ld + 0.1 * lr_).item()
            vl /= max(1, len(va_loader))
            elapsed  = (time.time() - t0) / 60
            eta      = elapsed / ep * (epochs - ep) if ep > 0 else 0
            free_now = torch.cuda.mem_get_info(0)[0] / 1024**3
            lr_now   = opt.param_groups[0]['lr']
            print(f"  Ep {ep:3d}/{epochs} | "
                  f"Ldiff={ep_diff/N:.4f} Lrecon={ep_recon/N:.4f} "
                  f"Lphys={ep_phys/N:.4f} Lpatch={ep_patch/N:.4f} | "
                  f"Lam={lam_phys:.2f} LR={lr_now:.2e} | "
                  f"Val={vl:.4f} | {elapsed:.0f}m ETA={eta:.0f}m | "
                  f"VRAM_free={free_now:.1f}GB")
            if vl < best_val:
                best_val = vl
                torch.save({'epoch': ep, 'model': model.state_dict(), 'val': vl}, ckpt_path)
                print(f"  ✔  Best ckpt saved (ep={ep}, val={vl:.4f})")

    # ── Final Evaluation ──────────────────────────────────────────
    print("\n  Loading best checkpoint for final evaluation ...")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state['model'], strict=True)
    model.eval()

    test_loader = DataLoader(PatchDataset("test"), bs, shuffle=False, num_workers=4)
    all_p, all_t, all_i = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            torch.cuda.empty_cache()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = model.predict(x.to(device), num_steps=ddim_steps).clamp(0,1).float()
            all_p.append(pred.cpu().numpy())
            all_t.append(y.numpy())
            all_i.append(x.numpy())

    metrics = compute_metrics(
        np.concatenate(all_p), np.concatenate(all_t), np.concatenate(all_i))

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*68}")
    print(f"  {exp_name} — FINAL RESULTS vs BASELINE")
    bm = baseline_metrics.get(exp_name, {})
    for k, v in metrics.items():
        b = bm.get(k, 0.0)
        arrow = "✅ better" if (k in ["MAE","RMSE","PVR","MND"] and v < b) or \
                              (k in ["R2","SSIM"] and v > b) else "⚠️  worse"
        print(f"    {k:6s}: {v:.6f}  (baseline={b:.4f})  {arrow}")
    print(f"{'='*68}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp",    required=True, choices=["P1_FT", "P1KD_FT"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--bs",     type=int, default=None)
    args = parser.parse_args()

    use_kd  = (args.exp == "P1KD_FT")
    # P1-KD uses bs=2 (VGG19 features + diffusion heavier), P1 can use bs=4
    default_bs = 2 if use_kd else 4
    bs = args.bs if args.bs else default_bs

    print(f"  Config: exp={args.exp} | epochs={args.epochs} | bs={bs} | use_kd={use_kd}")
    run_finetuned(args.exp, use_kd=use_kd, epochs=args.epochs, bs=bs)


if __name__ == "__main__":
    main()
