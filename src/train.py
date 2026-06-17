import torch
import torch.optim as optim
import os
import sys
from pathlib import Path
import wandb
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.dataset import IIDMDataset as CarbonDataset
from models.iidm import IIDM
from torch.utils.data import Dataset, DataLoader

class LocalDevDataset(Dataset):
    def __len__(self): return 16
    def __getitem__(self, idx):
        return {
            'optical': torch.randn(6, 256, 256),
            'structural': torch.randn(2, 256, 256),
            'carbon': torch.randn(1, 256, 256)
        }

def train_actual_iidm():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Actual IIDM (INR + KD) Training pipeline initialized on: {device}")

    # Initialize WandB
    wandb.init(
        project="IIDM-Carbon-Estimation",
        name="actual-iidm-inr-run",
        config={"epochs": 100, "batch_size": 4, "learning_rate": 1e-4}
    )

    try:
        dataset = CarbonDataset(patch_dir=Path('data/patches'))
        if len(dataset) == 0: raise ValueError("Empty directory")
        print("✅ Loaded real Preprocessed Dataset")
    except Exception as e:
        print("⚠️ Real dataset missing. Auto-switching to LocalDevDataset for architecture testing...")
        dataset = LocalDevDataset()

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Initialize the Full Unified IIDM Model
    # in_channels=6 matches our Sentinel-2 stacked optical input
    model = IIDM(in_channels=6, T=1000).to(device)

    # Optimizer only trains the Student, UNet, and INR. (Teacher is frozen)
    optimizer = optim.AdamW([
        {'params': model.student.parameters()},
        {'params': model.unet.parameters()},
        {'params': model.inr.parameters()}
    ], lr=1e-4)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    print("\n🔥 Starting INR-based Training Loop...")
    for epoch in range(1, 101):
        model.train()
        total_epoch_loss = 0
        epoch_metrics = {'diff': 0, 'kd': 0, 'recon': 0}
        
        for batch in dataloader:
            optical = batch['optical'].to(device)
            carbon = batch['carbon'].to(device)
            # Note: Structural features can be concatenated to optical in future upgrades, 
            # for now base paper relies heavily on optical for INR.

            optimizer.zero_grad()

            # The forward pass now handles the entire pipeline and returns aggregated loss
            L_total, loss_dict = model(optical, carbon)
            
            L_total.backward()
            optimizer.step()
            
            total_epoch_loss += L_total.item()
            for k in epoch_metrics:
                epoch_metrics[k] += loss_dict[k]

        scheduler.step()
        num_batches = len(dataloader)
        avg_loss = total_epoch_loss / num_batches
        
        print(f"Epoch [{epoch}/100] | Total Loss: {avg_loss:.4f} | Recon: {epoch_metrics['recon']/num_batches:.4f} | Diff: {epoch_metrics['diff']/num_batches:.4f}")
        
        # Log all individual components to WandB
        wandb.log({
            "epoch": epoch, 
            "total_loss": avg_loss, 
            "diffusion_loss": epoch_metrics['diff'] / num_batches,
            "kd_loss": epoch_metrics['kd'] / num_batches,
            "inr_recon_loss": epoch_metrics['recon'] / num_batches,
            "learning_rate": scheduler.get_last_lr()[0]
        })
        
        if epoch % 10 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/actual_iidm_epoch_{epoch}.pth")
        
    print("\n✅ Actual IIDM Pipeline Completed!")
    wandb.finish()

if __name__ == "__main__":
    train_actual_iidm()
