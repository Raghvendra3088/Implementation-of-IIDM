import torch
import torch.optim as optim
import os

# Import our custom modules
from models.kd_vgg import StudentVGG, TeacherVGG, KDVGGLoss
from models.kd_unet import KDUnet
from models.diffusion import IIDM_Diffusion

def train_iidm():
    # Automatically switch to GPU if available (for Colab/Kaggle later)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Starting training pipeline on: {device}")

    # 1. Initialize Models
    print("Loading models into memory...")
    teacher_vgg = TeacherVGG().to(device)
    teacher_vgg.eval() # Teacher is frozen, no gradients needed
    
    student_vgg = StudentVGG().to(device)
    unet = KDUnet(in_channels=3, out_channels=1).to(device)
    diffusion = IIDM_Diffusion(denoise_model=unet, timesteps=1000).to(device)

    # 2. Define Losses & Optimizer
    kd_criterion = KDVGGLoss().to(device)
    lambda_kd = 0.1 # Weight parameter for Knowledge Distillation loss
    
    # We optimize both the student feature extractor and the denoising UNet together
    optimizer = optim.AdamW(list(student_vgg.parameters()) + list(unet.parameters()), lr=1e-4)

    # 3. Setup Checkpoint Directory
    os.makedirs("checkpoints", exist_ok=True)
    epochs = 10 # Set to a low number for initial testing

    print("\n🔥 Initiating Training Loop...")
    for epoch in range(1, epochs + 1):
        student_vgg.train()
        unet.train()
        
        # --- DUMMY DATALOADER PLACEHOLDER ---
        # In the real run, this will be: for batch in dataloader:
        batch_size = 2
        dummy_optical = torch.randn(batch_size, 6, 256, 256).to(device)
        dummy_structural = torch.randn(batch_size, 2, 256, 256).to(device) # DEM + Canopy
        dummy_carbon_target = torch.randn(batch_size, 1, 256, 256).to(device)
        dummy_timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
        # ------------------------------------

        optimizer.zero_grad()

        # Phase A: Knowledge Distillation Forward Pass
        with torch.no_grad():
            t_feats = teacher_vgg(dummy_optical)
        s_feats = student_vgg(dummy_optical)
        
        loss_kd = kd_criterion(s_feats, t_feats)

        # Phase B: Diffusion Forward Pass
        cond_features = s_feats[-1] # Take deepest feature map for conditioning
        loss_diff = diffusion.p_losses(dummy_carbon_target, dummy_timesteps, cond_features, dummy_structural)

        # Phase C: Total Loss & Backpropagation
        total_loss = loss_diff + (lambda_kd * loss_kd)
        total_loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch}/{epochs}] | KD Loss: {loss_kd.item():.4f} | Diff Loss: {loss_diff.item():.4f} | Total Loss: {total_loss.item():.4f}")

        # Phase D: Save Checkpoints
        if epoch % 5 == 0:
            checkpoint_path = f"checkpoints/iidm_epoch_{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'student_vgg_state': student_vgg.state_dict(),
                'unet_state': unet.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss': total_loss.item()
            }, checkpoint_path)
            print(f"✅ Checkpoint safely stored at {checkpoint_path}")

if __name__ == "__main__":
    train_iidm()
