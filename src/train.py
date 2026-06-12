import torch
import torch.optim as optim
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.dataset import CarbonDataset
from models.kd_vgg import StudentVGG, TeacherVGG, KDVGGLoss
from models.kd_unet import KDUnet
from models.diffusion import IIDM_Diffusion

def train_iidm():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Training on: {device}")

    dataset = CarbonDataset(root_dir='data/patches')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    teacher_vgg = TeacherVGG().to(device)
    student_vgg = StudentVGG().to(device)
    unet = KDUnet(in_channels=3, out_channels=1).to(device)
    diffusion = IIDM_Diffusion(denoise_model=unet, timesteps=1000).to(device)

    kd_criterion = KDVGGLoss().to(device)
    optimizer = optim.AdamW(list(student_vgg.parameters()) + list(unet.parameters()), lr=1e-4)

    for epoch in range(1, 11):
        student_vgg.train()
        unet.train()
        
        for batch in dataloader:
            optical, structural, carbon = batch['optical'].to(device), batch['structural'].to(device), batch['carbon'].to(device)
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

        print(f"Epoch {epoch} | Loss: {total_loss.item():.4f}")

if __name__ == "__main__":
    train_iidm()
