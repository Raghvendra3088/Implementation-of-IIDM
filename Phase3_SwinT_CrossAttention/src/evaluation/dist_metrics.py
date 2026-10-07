import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from scipy.stats import wasserstein_distance, entropy

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.models.cnn_baseline import CNNBaseline
from src.models.vae import CarbonVAE
from src.models.kd_vgg import LightweightStudentEncoder
from src.models.kd_unet import KDUNet
from src.models.diffusion import DiffusionScheduler
from src.models.pi_ldm import PILDM

C_MIN = 5.051
C_MAX = 198.261

def denorm(arr, vmin=C_MIN, vmax=C_MAX):
    return arr * (vmax - vmin) + vmin

def kl_divergence(p_pred, p_gt, bins=50):
    hist_gt, bin_edges = np.histogram(p_gt, bins=bins, range=(C_MIN, C_MAX), density=True)
    hist_pred, _ = np.histogram(p_pred, bins=bin_edges, density=True)
    
    # Add epsilon to avoid log(0)
    hist_gt = hist_gt + 1e-8
    hist_pred = hist_pred + 1e-8
    
    # Normalize back to proper probabilities
    hist_gt /= np.sum(hist_gt)
    hist_pred /= np.sum(hist_pred)
    
    return entropy(hist_pred, hist_gt)

class TestDataset(Dataset):
    def __init__(self):
        self.data_dir = ROOT / "data" / "processed" / "patches_6ch" / "test"
        self.input_files = sorted(glob.glob(str(self.data_dir / "input" / "*.npz")))
        self.target_files = sorted(glob.glob(str(self.data_dir / "target" / "*.npz")))
            
    def __len__(self):
        return len(self.input_files)
        
    def __getitem__(self, idx):
        with np.load(self.input_files[idx]) as d_in, np.load(self.target_files[idx]) as d_tgt:
            inp = d_in['image']
            tgt = d_tgt['image']
        if len(tgt.shape) == 2:
            tgt = np.expand_dims(tgt, axis=0)
        return torch.from_numpy(inp).float(), torch.from_numpy(tgt).float()

def load_pi_ldm(ckpt_dir, device):
    vae = CarbonVAE(in_channels=1, latent_channels=4, base_channels=64)
    encoder = LightweightStudentEncoder(in_channels=8)
    unet = KDUNet(in_channels=4, out_channels=4, context_dim=256)
    scheduler = DiffusionScheduler(T=1000, device=device)
    
    vae_ckpt = ROOT / "results" / "checkpoints" / "vae_best.pt"
    vae.load_state_dict(torch.load(vae_ckpt, map_location="cpu"))
        
    model = PILDM(vae, encoder, unet, scheduler, freeze_encoder=True).to(device)
    unet_ckpt = Path(ckpt_dir) / "unet_best.pt"
    model.unet.load_state_dict(torch.load(unet_ckpt, map_location=device))
    model.eval()
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    ds = TestDataset()
    loader = DataLoader(ds, batch_size=16, shuffle=False)
    
    cnn = CNNBaseline(in_channels=8).to(device)
    cnn.load_state_dict(torch.load(ROOT / "results" / "checkpoints" / "unet_baseline" / "unet_best.pt", map_location=device))
    cnn.eval()
    
    iidm = load_pi_ldm(ROOT / "results" / "checkpoints" / "pildm_seed42_L0.001", device)
    pi_iidm = load_pi_ldm(ROOT / "results" / "checkpoints" / "pildm_seed42_L0.5", device)
    
    metrics = {
        "cnn": {"wd": [], "kl": []},
        "iidm": {"wd": [], "kl": []},
        "pi_iidm": {"wd": [], "kl": []}
    }
    
    print("Evaluating Distributional Metrics (Wasserstein & KL Divergence)...")
    for i, (inp, tgt) in enumerate(loader):
        inp = inp.to(device)
        B = inp.shape[0]
        
        gt_np = denorm(tgt.numpy())
        
        with torch.no_grad():
            cnn_pred = denorm(cnn(inp).cpu().numpy())
            iidm_pred = denorm(iidm.predict(inp, num_steps=20).cpu().numpy())
            pi_iidm_pred = denorm(pi_iidm.predict(inp, num_steps=20).cpu().numpy())
            
        preds = {"cnn": cnn_pred, "iidm": iidm_pred, "pi_iidm": pi_iidm_pred}
            
        for b in range(B):
            gt_flat = gt_np[b].flatten()
            for name, p in preds.items():
                p_flat = p[b].flatten()
                metrics[name]["wd"].append(wasserstein_distance(gt_flat, p_flat))
                metrics[name]["kl"].append(kl_divergence(p_flat, gt_flat))
                
        if i >= 10: # Sample enough patches for stability
            break
            
    print("\n--- DISTRIBUTIONAL METRICS ---")
    for name in metrics:
        wd_mean = np.mean(metrics[name]["wd"])
        kl_mean = np.mean(metrics[name]["kl"])
        print(f"[{name.upper()}] Wasserstein Distance: {wd_mean:.4f} | KL Divergence: {kl_mean:.4f}")

if __name__ == "__main__":
    main()
