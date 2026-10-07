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
from src.models.kd_vgg import VGG19_Adapter
from src.models.kd_unet import KDUNet
from src.models.diffusion import DiffusionScheduler
from src.models.pi_ldm import PILDM, compute_physics_loss

class PILdmDataset(Dataset):
    def __init__(self, split="train"):
        self.data_dir = ROOT / "data" / "processed" / "patches_6ch" / split
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

def train_pi_ldm(lambda_phys=0.5, epochs=250, batch_size=16, lr=1e-4, seed=42):
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
    encoder = VGG19_Adapter(in_channels=8)
    unet = KDUNet(in_channels=4, out_channels=4, context_dim=256)
    scheduler = DiffusionScheduler(T=1000, device=device)
    
    # Optionally load VAE weights if pretrained
    vae_ckpt = ROOT / "results" / "checkpoints" / "vae_best.pt"
    if vae_ckpt.exists():
        print("Loading pretrained VAE...")
        vae.load_state_dict(torch.load(vae_ckpt, map_location="cpu"))
        
    # Freeze the main VGG19 backbone inside the adapter, but train the projections
    for name, param in encoder.named_parameters():
        if "block" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
            
    model = PILDM(vae, encoder, unet, scheduler, freeze_encoder=False).to(device)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for DataParallel!")
        model = torch.nn.DataParallel(model)
    
    # Only optimize parameters that require gradients
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_val_loss = float('inf')
    save_dir = ROOT / "results" / "checkpoints" / f"pildm_seed{seed}_L{lambda_phys}_VGG19_INR_MultiGPU"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        train_diff = 0
        train_recon = 0
        train_phys = 0
        
        # Warm-up schedule for Physics Loss: 0 for first 100 epochs, linearly scale to lambda_phys
        lambda_curr = lambda_phys * min(1.0, max(0.0, (epoch - 100) / 150.0))
        
        for i, (inp, tgt) in enumerate(train_loader):
            inp = inp.to(device)
            tgt = tgt.to(device)
            
            # Require grad on inp for physics loss
            inp.requires_grad_(True)
            
            B = inp.shape[0]
            t = torch.randint(0, scheduler.T, (B,), device=device).long()
            
            optimizer.zero_grad()
            
            # Diffusion & Recon Loss
            # DataParallel automatically splits the batch and gathers the loss.
            # We take the mean() in case DataParallel returns a vector of losses (one per GPU).
            loss_diff, loss_recon, _ = model(inp, tgt, t)
            loss_diff = loss_diff.mean()
            loss_recon = loss_recon.mean()
            
            # Physics Loss
            if lambda_curr > 0:
                l_mono = compute_physics_loss(model, inp).mean()
            else:
                l_mono = torch.tensor(0.0, device=device)
                
            loss = loss_diff + loss_recon + lambda_curr * l_mono
            loss.backward()
            optimizer.step()
            
            train_diff += loss_diff.item()
            train_recon += loss_recon.item()
            train_phys += l_mono.item()
            
        lr_scheduler.step()
            
        # Eval
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inp, tgt in val_loader:
                inp = inp.to(device)
                tgt = tgt.to(device)
                B = inp.shape[0]
                t = torch.randint(0, scheduler.T, (B,), device=device).long()
                l_diff, l_recon, _ = model(inp, tgt, t)
                l_diff = l_diff.mean()
                l_recon = l_recon.mean()
                val_loss += (l_diff.item() + l_recon.item())
                
        t_diff = train_diff / len(train_loader)
        t_recon = train_recon / len(train_loader)
        t_phys = train_phys / len(train_loader)
        v_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs} | LR: {lr_scheduler.get_last_lr()[0]:.2e} | L_diff: {t_diff:.4f} | L_recon: {t_recon:.4f} | L_phys({lambda_curr:.3f}): {t_phys:.4f} | Val: {v_loss:.4f}")
        
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            
            # Unwrap model for saving
            model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
            
            torch.save(model_to_save.unet.state_dict(), save_dir / "unet_best.pt")
            torch.save(model_to_save.inr.state_dict(), save_dir / "inr_best.pt")
            torch.save(model_to_save.encoder.state_dict(), save_dir / "encoder_best.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_phys", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    train_pi_ldm(lambda_phys=args.lambda_phys, epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)
