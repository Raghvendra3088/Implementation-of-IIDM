import os, glob
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed" / "patches_6ch"

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

def evaluate_vae_ceiling():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    from src.models.vae import CarbonVAE
    vae = CarbonVAE(in_channels=1, latent_channels=4, base_channels=64).to(device)
    vae_ckpt = ROOT / "results" / "checkpoints" / "vae_best.pt"
    if not vae_ckpt.exists():
        print(f"VAE checkpoint not found at {vae_ckpt}!")
        return
        
    vae.load_state_dict(torch.load(vae_ckpt, map_location=device))
    vae.eval()
    
    loader = DataLoader(PatchDataset("test"), batch_size=8, shuffle=False, num_workers=4)
    all_p, all_t, all_i = [], [], []
    
    print("Evaluating VAE Reconstruction Ceiling (Oracle z0)...")
    with torch.no_grad():
        for x, y in loader:
            y_dev = y.to(device)
            z, _ = vae.encode(y_dev)
            y_recon = vae.decode(z).clamp(0, 1)
            
            # Support-aware footprint integrated evaluation (3x3 avg pool)
            y_recon_footprint = F.avg_pool2d(y_recon, kernel_size=3, stride=1, padding=1)
            all_p.append(y_recon_footprint.cpu().numpy())
            all_t.append(y.numpy())
            all_i.append(x.numpy())
            
    p_all = np.concatenate(all_p)
    t_all = np.concatenate(all_t)
    i_all = np.concatenate(all_i)
    
    metrics = compute_metrics(p_all, t_all, i_all)
    metrics["RMSE_MgCha"] = metrics["RMSE"] * 130
    metrics["MAE_MgCha"] = metrics["MAE"] * 130
    
    print("\n=== VAE Reconstruction Ceiling (Oracle z0) ===")
    print(f"RMSE: {metrics['RMSE_MgCha']:.2f} Mg C/ha (norm: {metrics['RMSE']:.4f})")
    print(f"MAE:  {metrics['MAE_MgCha']:.2f} Mg C/ha (norm: {metrics['MAE']:.4f})")
    print(f"R2:   {metrics['R2']:.4f}")
    print(f"SSIM: {metrics['SSIM']:.4f}")
    print(f"PVR:  {metrics['PVR']:.4f}")
    print(f"MND:  {metrics['MND']:.6f}")
    
    out_file = ROOT / "results" / "vae_ceiling_metrics.json"
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to {out_file}")

if __name__ == "__main__":
    evaluate_vae_ceiling()
