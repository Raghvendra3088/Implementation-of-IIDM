import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sys
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.models.vae import CarbonVAE, vae_loss

class TargetPatchDataset(Dataset):
    def __init__(self, split="train"):
        self.data_dir = ROOT / "data" / "processed" / "patches_6ch" / split / "target"
        self.files = sorted(glob.glob(str(self.data_dir / "*.npy")))
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        # Target shape is (1, 256, 256)
        tgt = np.load(self.files[idx])
        return torch.from_numpy(tgt).float()

def train_vae(epochs=50, batch_size=32, lr=1e-4, seed=42):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training VAE on {device}")
    
    train_dataset = TargetPatchDataset("train")
    val_dataset = TargetPatchDataset("val")
    
    if len(train_dataset) == 0:
        print("No training data found. Please run preprocessing.")
        return
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    model = CarbonVAE(in_channels=1, latent_channels=4, base_channels=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    save_dir = ROOT / "results" / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for tgt in train_loader:
            tgt = tgt.to(device)
            
            optimizer.zero_grad()
            x_rec, mean, logvar = model(tgt)
            loss, recon_loss, kl_loss = vae_loss(x_rec, tgt, mean, logvar, beta=0.01)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for tgt in val_loader:
                tgt = tgt.to(device)
                x_rec, mean, logvar = model(tgt)
                loss, _, _ = vae_loss(x_rec, tgt, mean, logvar, beta=0.01)
                val_loss += loss.item()
                
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_dir / "vae_best.pt")

if __name__ == "__main__":
    train_vae()
