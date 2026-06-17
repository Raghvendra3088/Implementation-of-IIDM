import torch
import torch.optim as optim
import os
import sys
from pathlib import Path
import wandb
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
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Training pipeline initialized on: {device}")

    wandb.init(
        project="IIDM-Carbon-Estimation",
        name="base-model-run",
        config={"epochs": 100, "batch_size": 4, "learning_rate": 1e-4}
    )

    try:
        dataset = CarbonDataset(patch_dir=Path('data/patches'))
        if len(dataset) == 0: raise ValueError("Empty directory")
        print("✅ Loaded real Preprocessed Dataset")
    except Exception as e:
        print("⚠️ Real dataset missing. Auto-switching to LocalDevDataset...")
        dataset = LocalDevDataset()

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    teacher_vgg = TeacherVGG().to(device)
    student_vgg = StudentVGG().to(device)
    unet = KDUnet(in_channels=3, out_channels=1).to(device)
    diffusion = IIDM_Diffusion(denoise_model=unet, timesteps=1000).to(device)

    kd_criterion = KDVGGLoss().to(device)
    optimizer = optim.AdamW(list(student_vgg.parameters()) + list(unet.parameters()), lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    print("\n🔥 Starting Training Loop...")
    for epoch in range(1, 101):
        student_vgg.train()
        unet.train()
        total_epoch_loss = 0
        
        for batch in dataloader:
            optical = batch['optical'].to(device)
            structural = batch['structural'].to(device)
            carbon = batch['carbon'].to(device)
            
            timesteps = torch.randint(0, 1000, (optical.shape[0],)).to(device)

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

        scheduler.step()
        avg_loss = total_epoch_loss/len(dataloader)
        print(f"Epoch [{epoch}/100] | Average Loss: {avg_loss:.4f}")
        
        wandb.log({
            "epoch": epoch, 
            "total_loss": avg_loss, 
            "kd_loss": loss_kd.item(), 
            "diffusion_loss": loss_diff.item(),
            "learning_rate": scheduler.get_last_lr()[0]
        })
        
        if epoch % 10 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(unet.state_dict(), f"checkpoints/unet_epoch_{epoch}.pth")
        
    print("\n✅ Training Pipeline Completed!")
    wandb.finish()

if __name__ == "__main__":
    train_iidm()
