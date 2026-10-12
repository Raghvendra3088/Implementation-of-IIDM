import os, sys, glob, json, time, argparse
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

DATA_DIR    = ROOT / "data" / "processed" / "patches_6ch"
RESULTS_DIR = ROOT / "results" / "baselines" / "swin_deterministic"

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

def stochastic_point_physics_loss(x_batch, pred_batch, num_points=250):
    B, _, H, W = x_batch.shape
    batch_idx = torch.randint(0, B, (num_points,), device=x_batch.device)
    y_idx = torch.randint(0, H, (num_points,), device=x_batch.device)
    x_idx = torch.randint(0, W, (num_points,), device=x_batch.device)
    selected_y = pred_batch[batch_idx, 0, y_idx, x_idx]
    dY_dX = torch.autograd.grad(outputs=selected_y.sum(), inputs=x_batch, create_graph=True, retain_graph=True, only_inputs=True)[0]
    return F.relu(-dY_dX[batch_idx, 7, y_idx, x_idx]).mean()

def physics_lambda(ep, warmup=30, total=100, max_lam=0.5):
    if ep < warmup: return 0.0
    return max_lam * (ep - warmup) / (total - warmup)

class SwinDirect(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        from src.models.implicit_repr import SIRENINR
        self.inr = SIRENINR(student_chs=[96, 192, 384, 768, 8, 4])
        
    def forward(self, x_cond, y_target=None):
        swin_feats, _ = self.encoder(x_cond)
        dummy_z = torch.zeros(x_cond.shape[0], 4, x_cond.shape[2]//8, x_cond.shape[3]//8, device=x_cond.device)
        y_pred = self.inr(swin_feats + [x_cond], dummy_z)
        
        if y_target is not None:
            y_pred_footprint = F.avg_pool2d(y_pred, kernel_size=3, stride=1, padding=1)
            loss_recon = F.mse_loss(y_pred_footprint, y_target)
            return loss_recon, y_pred
        return y_pred

def run_deterministic(epochs=100, bs=4, lr=1e-4, grad_accum=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
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
        def forward(self, x):
            feats = self.swin(x)
            student_feats = [f.permute(0, 3, 1, 2).contiguous() for f in feats]
            return student_feats, student_feats

    encoder = SwinEncoderWithSkipWrap(in_chans=8).to(device)
    model = SwinDirect(encoder).to(device)
    model = nn.DataParallel(model)
    
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")
    
    tr_loader = DataLoader(PatchDataset("train"), bs, shuffle=True, num_workers=4, drop_last=True)
    va_loader = DataLoader(PatchDataset("val"), bs, shuffle=False, num_workers=4)
    
    best_val = float('inf')
    ckpt_path = RESULTS_DIR / "best_deterministic.pt"
    
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        lam_phys = physics_lambda(ep, warmup=30, total=epochs)
        ep_loss = 0.0
        opt.zero_grad()
        
        for step, (x, y) in enumerate(tr_loader):
            x, y = x.to(device), y.to(device)
            if lam_phys > 0: x.requires_grad_(True)
            
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss, y_pred = model(x, y)
                
            if lam_phys > 0:
                l_phys = stochastic_point_physics_loss(x.float(), y_pred.float())
                loss = loss + lam_phys * l_phys
                
            scaler.scale(loss / grad_accum).backward()
            if (step + 1) % grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
            ep_loss += loss.item()
                
        cosine.step()
        
        if ep % 5 == 0 or ep == epochs:
            model.eval()
            vl = 0.0
            with torch.no_grad():
                for xv, yv in va_loader:
                    xv, yv = xv.to(device), yv.to(device)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        v_loss, _ = model(xv, yv)
                    vl += v_loss.item()
            vl /= max(1, len(va_loader))
            elapsed = (time.time() - t0) / 60
            eta = elapsed / ep * (epochs - ep) if ep > 0 else 0
            print(f"Ep {ep:3d}/{epochs} | TrainLoss={ep_loss/len(tr_loader):.4f} | ValLoss={vl:.4f} | {elapsed:.0f}m ETA={eta:.0f}m")
            if vl < best_val:
                best_val = vl
                torch.save(model.state_dict(), ckpt_path)
                print(f"  ✔ Best model saved (ep={ep}, val={vl:.4f})")
    
    # ── Final Test Evaluation ──────────────────────────────────
    print("\nRunning final test evaluation...")
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    
    test_loader = DataLoader(PatchDataset("test"), bs, shuffle=False, num_workers=4)
    all_p, all_t, all_i = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = model(x.to(device)).clamp(0, 1).float()
            all_p.append(pred.cpu().numpy())
            all_t.append(y.numpy())
            all_i.append(x.numpy())
            
    p_all = np.concatenate(all_p)
    t_all = np.concatenate(all_t)
    i_all = np.concatenate(all_i)
    
    metrics = compute_metrics(p_all, t_all, i_all)
    metrics["RMSE_MgCha"] = metrics["RMSE"] * 130
    metrics["MAE_MgCha"] = metrics["MAE"] * 130
    
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"\n{'='*60}")
    print("  Swin-T Direct Regression Final Metrics:")
    for k, v in metrics.items():
        print(f"    {k:12s}: {v:.6f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--bs", type=int, default=4)
    args = parser.parse_args()
    run_deterministic(epochs=args.epochs, bs=args.bs)
