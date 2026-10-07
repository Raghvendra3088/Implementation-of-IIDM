import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sys
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.models.cnn_baseline import CNNBaseline

class BaselineDataset(Dataset):
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

def train_unet_baseline(epochs=50, batch_size=16, lr=1e-4, seed=42):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training U-Net Baseline (seed={seed}) on {device}")
    
    train_dataset = BaselineDataset("train")
    val_dataset = BaselineDataset("val")
    
    if len(train_dataset) == 0:
        print("No training data found. Please run preprocessing.")
        return
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    model = CNNBaseline(in_channels=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()
    
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "unet_baseline.log"
    
    ckpt_dir = ROOT / "results" / "checkpoints" / "unet_baseline"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float('inf')
    
    with open(log_file, "w") as f:
        f.write("Epoch | Train Loss | Val L1\n")
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        
        for x_cond, x_tgt in train_loader:
            x_cond, x_tgt = x_cond.to(device), x_tgt.to(device)
            
            optimizer.zero_grad()
            y_pred = model(x_cond)
            loss = criterion(y_pred, x_tgt)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_cond, x_tgt in val_loader:
                x_cond, x_tgt = x_cond.to(device), x_tgt.to(device)
                y_pred = model(x_cond)
                val_loss += criterion(y_pred, x_tgt).item()
        val_loss /= len(val_loader)
        
        log_str = f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val L1: {val_loss:.4f}"
        print(log_str)
        with open(log_file, "a") as f:
            f.write(log_str + "\n")
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_dir / "unet_best.pt")

if __name__ == "__main__":
    train_unet_baseline()
