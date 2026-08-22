import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sys
from pathlib import Path
import argparse

# Add src to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.models.vae import CarbonVAE
from src.models.kd_vgg import LightweightStudentEncoder
from src.models.kd_unet import KDUnet
from src.models.diffusion import SimpleDiffusionScheduler
from src.models.pi_ldm import PILDM, compute_physics_loss

class PILdmDataset(Dataset):
    def __init__(self, split="train"):
        self.inp_dir = ROOT / "data" / "processed" / "patches_6ch" / split / "input"
        self.tgt_dir = ROOT / "data" / "processed" / "patches_6ch" / split / "target"
        
        self.inp_files = sorted(glob.glob(str(self.inp_dir / "*.npy")))
        self.tgt_files = sorted(glob.glob(str(self.tgt_dir / "*.npy")))
        
    def __len__(self):
        return len(self.inp_files)
        
    def __getitem__(self, idx):
        # Input (6, 256, 256), Target (1, 256, 256)
        inp = np.load(self.inp_files[idx])
        tgt = np.load(self.tgt_files[idx])
        return torch.from_numpy(inp).float(), torch.from_numpy(tgt).float()

def train_pi_ldm(lambda_phys=0.05, epochs=50, batch_size=16, lr=5e-5, seed=42):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training PI-LDM (seed={seed}, lambda_phys={lambda_phys}) on {device}")
    
    train_dataset = PILdmDataset("train")
    val_dataset = PILdmDataset("val")
    
    if len(train_dataset) == 0:
        print("No training data found. Please run preprocessing.")
        return
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Initialize components
    vae = CarbonVAE(in_channels=1, latent_channels=4, base_channels=64)
    encoder = LightweightStudentEncoder(in_channels=6)
    unet = KDUnet(in_channels=4, out_channels=4, context_dim=256)
    scheduler = SimpleDiffusionScheduler(num_train_timesteps=1000)
    
    # Optionally load VAE and Encoder weights if pretrained
    vae_ckpt = ROOT / "results" / "checkpoints" / "vae_best.pt"
    if vae_ckpt.exists():
        print("Loading pretrained VAE...")
        vae.load_state_dict(torch.load(vae_ckpt, map_location="cpu"))
        
    model = PILDM(vae, encoder, unet, scheduler).to(device)
    
    # Only optimize UNet since VAE/Encoder might be frozen (PI-LDM wrapper freezes them by default)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    best_val_loss = float('inf')
    save_dir = ROOT / "results" / "checkpoints" / f"pildm_seed{seed}_L{lambda_phys}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        train_diff = 0
        train_phys = 0
        
        for inp, tgt in train_loader:
            inp = inp.to(device)
            tgt = tgt.to(device)
            
            # Require grad on inp for physics loss
            inp.requires_grad_(True)
            
            B = inp.shape[0]
            t = torch.randint(0, scheduler.num_train_timesteps, (B,), device=device).long()
            
            optimizer.zero_grad()
            
            # Diffusion Loss
            loss_diff = model(inp, tgt, t)
            
            # Physics Loss
            if lambda_phys > 0:
                l_mono = compute_physics_loss(model, inp)
            else:
                l_mono = torch.tensor(0.0, device=device)
                
            loss = loss_diff + lambda_phys * l_mono
            loss.backward()
            optimizer.step()
            
            train_diff += loss_diff.item()
            train_phys += l_mono.item()
            
        # Eval
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inp, tgt in val_loader:
                inp = inp.to(device)
                tgt = tgt.to(device)
                B = inp.shape[0]
                t = torch.randint(0, scheduler.num_train_timesteps, (B,), device=device).long()
                loss = model(inp, tgt, t)
                val_loss += loss.item()
                
        t_diff = train_diff / len(train_loader)
        t_phys = train_phys / len(train_loader)
        v_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs} | T_Diff: {t_diff:.4f} | T_Phys: {t_phys:.4f} | Val L1: {v_loss:.4f}")
        
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.unet.state_dict(), save_dir / "unet_best.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_phys", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    train_pi_ldm(lambda_phys=args.lambda_phys, seed=args.seed)
