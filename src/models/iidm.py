import torch
import torch.nn as nn
import torch.nn.functional as F
from .kd_vgg import TeacherVGG19, StudentVGG, KDLoss
from .kd_unet import KDUNet
from .inr import ImplicitNeuralRepr
from .diffusion import DiffusionScheduler

class IIDM(nn.Module):
    def __init__(self, in_channels=6, T=1000):
        super().__init__()
        self.teacher = TeacherVGG19(in_channels=in_channels)
        self.student = StudentVGG(in_channels=in_channels)
        
        # FIX: The UNet diffuses the latent features, which have 32 channels.
        self.unet = KDUNet(in_channels=32, out_channels=32) 
        
        self.inr = ImplicitNeuralRepr(latent_dim=32, L=10)
        self.scheduler = DiffusionScheduler(T=T)
        self.kd_loss = KDLoss()

    def forward(self, x, carbon_gt):
        # 1. Feature Extraction
        with torch.no_grad():
            t_feats = self.teacher(x)
        s_feats, s_projs = self.student(x)

        # 2. Knowledge Distillation Loss
        L_kd = self.kd_loss(s_projs, t_feats)

        # 3. Latent Diffusion Forward Process
        latent_x0 = s_feats[0] # Finest scale student features (B, 32, H/2, W/2)
        t = torch.randint(0, self.scheduler.T, (x.shape[0],), device=x.device)
        x_t, noise = self.scheduler.q_sample(latent_x0, t)

        # 4. KD-UNet Denoising
        # Train condition is the teacher feature.
        # Ensure spatial dimensions match by interpolating if necessary
        condition_train = F.interpolate(t_feats[0], size=x_t.shape[-2:], mode='bilinear')
        pred_noise = self.unet(x_t, t, condition_train)

        # 5. INR Reconstruction
        # We train INR using the clean student features to map them to the full carbon map resolution
        carbon_pred = self.inr(latent_x0, x.shape[2], x.shape[3])

        # 6. Losses
        L_diff = F.mse_loss(pred_noise, noise)
        L_recon = F.l1_loss(carbon_pred, carbon_gt) # MAE loss as per paper
        
        # Total Loss (Paper weights: λ1 = 0.1, λ2 = 1.0)
        L_total = L_diff + 0.1 * L_kd + 1.0 * L_recon

        return L_total, {'diff': L_diff.item(), 'kd': L_kd.item(), 'recon': L_recon.item()}

    @torch.no_grad()
    def inference(self, x, steps=50):
        """DDIM Inference for generating Carbon Maps."""
        s_feats, _ = self.student(x)
        latent_shape = s_feats[0].shape
        
        x_T = torch.randn(latent_shape, device=x.device)
        
        # Reverse diffusion
        condition_infer = s_feats[0]
        x_0 = self.scheduler.ddim_sample(self.unet, x_T, condition=condition_infer, steps=steps)
        
        # Decode continuous map using INR
        carbon_map = self.inr(x_0, x.shape[2], x.shape[3])
        return carbon_map
