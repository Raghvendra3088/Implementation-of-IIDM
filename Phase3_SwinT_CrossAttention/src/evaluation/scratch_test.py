import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.models.vae import CarbonVAE
from src.models.kd_vgg import LightweightStudentEncoder
from src.models.kd_unet import KDUNet
from src.models.diffusion import DiffusionScheduler
from src.models.pi_ldm import PILDM

C_MIN = 5.051
C_MAX = 198.261

def denorm(arr, vmin=C_MIN, vmax=C_MAX):
    return arr * (vmax - vmin) + vmin

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
    # Check if this checkpoint is noKD
    if "noKD" in str(ckpt_dir):
        # We need to somehow tell PILDM it's noKD. But wait, in train_pi_ldm.py:
        # if args.disable_kd: cond = torch.zeros_like(...) 
        # KDUNet architecture is the same, just the condition input is zeros.
        pass
    
    unet = KDUNet(in_channels=4, out_channels=4, context_dim=256)
    scheduler = DiffusionScheduler(T=1000, device=device)
    
    vae_ckpt = ROOT / "results" / "checkpoints" / "vae_best.pt"
    if vae_ckpt.exists():
        vae.load_state_dict(torch.load(vae_ckpt, map_location="cpu"))
        
    model = PILDM(vae, encoder, unet, scheduler, freeze_encoder=True).to(device)
    
    unet_ckpt = Path(ckpt_dir) / "unet_best.pt"
    if unet_ckpt.exists():
        model.unet.load_state_dict(torch.load(unet_ckpt, map_location=device))
    
    model.eval()
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    ds = TestDataset()
    loader = DataLoader(ds, batch_size=16, shuffle=False)
    
    # Model B (LDM w/o physics, w/o KD)
    model_b = load_pi_ldm(ROOT / "results" / "checkpoints" / "pildm_seed42_L0.0_noKD", device)
    
    rmse_b = []
    
    print("Evaluating Model B (no KD, no Physics)...")
    for i, (inp, tgt) in enumerate(loader):
        inp = inp.to(device)
        B = inp.shape[0]
        gt_np = denorm(tgt.numpy())
        
        with torch.no_grad():
            # For Model B, condition is zero
            f_enc, f_proj, _ = model_b.encoder(inp)
            lat_hw = (32, 32)
            cond = torch.zeros(B, 256, *lat_hw, device=device) # Zeros for no KD
            
            z_T = torch.randn(B, 4, *lat_hw, device=device)
            z_0_pred = model_b.scheduler.ddim_sample(model_b.unet, z_T.shape, cond, steps=20)
            pred_b = denorm(model_b.vae.decode(z_0_pred).cpu().numpy())
            
        for b in range(B):
            err_b = (pred_b[b] - gt_np[b])**2
            rmse_b.append(np.sqrt(np.mean(err_b)))
                
        if i >= 10:
            break
            
    print(f"Model B Patch-Mean RMSE: {np.mean(rmse_b):.4f}")

if __name__ == "__main__":
    main()
