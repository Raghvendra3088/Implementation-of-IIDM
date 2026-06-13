import torch
import torch.optim as optim
import os
import sys
import wandb
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.dataset import IIDMDataset as CarbonDataset
from models.kd_vgg import StudentVGG, TeacherVGG, KDVGGLoss
from models.kd_unet import KDUnet
from models.diffusion import IIDM_Diffusion
from torch.utils.data import Dataset, DataLoader

class LocalDevDataset(Dataset):
    def __len__(self): return 16
    def __getitem__(self, idx):
        return {
            'optical': torch.randn(6, 256, 256),
            'structural': torch.randn(2, 256, 256),
            'carbon': torch.randn(1, 256, 256)
        }

def train_iidm():
    # Initialize WandB for logging
    wandb.init(
        project="IIDM-Carbon-Estimation",
        name="baseline-kd-diffusion",
        config={
            "learning_rate": 1e-4,
            "epochs": 3,
            "batch_size": 4,
            "architecture": "KD-VGG + KD-UNet + Diffusion",
            "timesteps": 1000
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Training pipeline initialized on: {device}")

    try:
        dataset = CarbonDataset(patch_dir=Path('data/patches'))
        if len(dataset) == 0: raise ValueError("Empty directory")
        print("✅ Loaded real Preprocessed Dataset")
    except Exception as e:
        print("⚠️ Auto-switching to LocalDevDataset for architecture testing...")
        dataset = LocalDevDataset()

    dataloader = DataLoader(dataset, batch_size=wandb.config.batch_size, shuffle=True)

    teacher_vgg = TeacherVGG().to(device)
    student_vgg = StudentVGG().to(device)
    unet = KDUnet(in_channels=3, out_channels=1).to(device)
    diffusion = IIDM_Diffusion(denoise_model=unet, timesteps=wandb.config.timesteps).to(device)

    kd_criterion = KDVGGLoss().to(device)
    optimizer = optim.AdamW(list(student_vgg.parameters()) + list(unet.parameters()), lr=wandb.config.learning_rate)

    print("\n🔥 Starting Training Loop...")
    for epoch in range(1, wandb.config.epochs + 1):
        student_vgg.train()
        unet.train()
        total_epoch_loss = 0
        total_kd_loss = 0
        total_diff_loss = 0
        
        for batch in dataloader:
            optical, structural, carbon = batch['optical'].to(device), batch['structural'].to(device), batch['carbon'].to(device)
            timesteps = torch.randint(0, wandb.config.timesteps, (optical.shape[0],)).to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                t_feats = teacher_vgg(optical)
            s_feats = student_vgg(optical)
            
            loss_kd = kd_criterion(s_feats, t_feats)
            loss_diff = diffusion.p_losses(carbon, timesteps, s_feats[-1], structural)

            total_loss = loss_diff + (0.1 * loss_kd)
            total_loss.backward()
            optimizer.step()
            
            total_epoch_loss += total_loss.item()
            total_kd_loss += loss_kd.item()
            total_diff_loss += loss_diff.item()

        avg_loss = total_epoch_loss / len(dataloader)
        avg_kd = total_kd_loss / len(dataloader)
        avg_diff = total_diff_loss / len(dataloader)

        print(f"Epoch [{epoch}/{wandb.config.epochs}] | KD: {avg_kd:.4f} | Diff: {avg_diff:.4f} | Total: {avg_loss:.4f}")
        
        # Send metrics to WandB Dashboard
        wandb.log({
            "epoch": epoch,
            "total_loss": avg_loss,
            "kd_loss": avg_kd,
            "diffusion_loss": avg_diff
        })
        
    print("\n✅ Run successful! Check your WandB dashboard for the graphs.")
    wandb.finish()

if __name__ == "__main__":
    train_iidm()
