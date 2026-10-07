import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.implicit_repr import SIRENINR

class TeacherCondenser(nn.Module):
    """
    Teacher features → each projected to 64ch → adaptive pooled to match latent shape → concat = 256ch.
    """
    TEACHER_CHS = [64, 128, 256, 512]
    OUT_CH_EACH = 64

    def __init__(self):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(t_ch, self.OUT_CH_EACH, 1, bias=False),
                nn.GroupNorm(8, self.OUT_CH_EACH),
                nn.SiLU(),
            )
            for t_ch in self.TEACHER_CHS
        ])

    def forward(self, teacher_feats, target_hw):
        projected = []
        for feat, proj in zip(teacher_feats, self.projs):
            p = proj(feat)
            if p.shape[-2:] != target_hw:
                p = F.adaptive_avg_pool2d(p, target_hw)
            projected.append(p)
        return torch.cat(projected, dim=1)   # (B, 256, H', W')

class PILDM(nn.Module):
    def __init__(self, vae, encoder, unet, diffusion_scheduler, freeze_encoder=True):
        super().__init__()
        self.vae = vae
        self.encoder = encoder
        self.unet = unet
        self.scheduler = diffusion_scheduler
        self.teacher_cond = TeacherCondenser()
        self.inr = SIRENINR(student_chs=[64, 128, 256, 512, 4])
        
        # Freeze VAE (always frozen during diffusion)
        for p in self.vae.parameters():
            p.requires_grad = False
            
        self.freeze_encoder = freeze_encoder
        # Freeze Encoder if KD is pre-trained or we are using Teacher
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, x_cond, y_target, t, noise=None):
        """
        x_cond:   (B, 6, H, W)   Input condition (Satellite + DEM + Canopy)
        y_target: (B, 1, H, W)   Target carbon map
        t:        (B,)           Timesteps
        """
        # Encode Condition
        if not self.freeze_encoder:
            f_enc, f_proj, _ = self.encoder(x_cond)
        else:
            with torch.no_grad():
                f_enc, f_proj, _ = self.encoder(x_cond)
            
        with torch.no_grad():
            # Encode Target into Latent Space
            z_0, _ = self.vae.encode(y_target)
            
        lat_hw = z_0.shape[-2:]
        cond = self.teacher_cond(f_proj, lat_hw)
            
        # Forward diffusion
        z_t, noise_true = self.scheduler.q_sample(z_0, t, noise)
        
        # Denoising
        noise_pred = self.unet(z_t, t, cond)
        loss_diff = F.l1_loss(noise_pred, noise_true)
        
        # INR Reconstruction Loss (using un-noised z_0 as the target representation)
        y_pred = self.inr(f_enc, z_0)
        loss_recon = F.mse_loss(y_pred, y_target)
        
        return loss_diff, loss_recon, f_proj, y_pred
        
    def predict(self, x_cond, num_steps=50):
        """DDIM sampling for inference"""
        B = x_cond.shape[0]
        device = x_cond.device
        
        with torch.no_grad():
            f_enc, f_proj, _ = self.encoder(x_cond)
            z_T = torch.randn(B, 4, 32, 32, device=device)
            lat_hw = z_T.shape[-2:]
            cond = self.teacher_cond(f_proj, lat_hw)
            
            z_0_pred = self.scheduler.ddim_sample(self.unet, z_T.shape, cond, steps=num_steps)
            y_pred = self.inr(f_enc, z_0_pred)
        return y_pred
    
    def decode(self, x_cond, num_steps=50):
        """For physics loss: Requires grad enabled through decoder"""
        B = x_cond.shape[0]
        device = x_cond.device
        
        # Get conditions
        f_enc, f_proj, _ = self.encoder(x_cond)
        z_T = torch.randn(B, 4, 32, 32, device=device)
        
        lat_hw = z_T.shape[-2:]
        cond = self.teacher_cond(f_proj, lat_hw)
        
        z_0_pred = self.scheduler.ddim_sample(self.unet, z_T.shape, cond, steps=num_steps)
        y_pred = self.inr(f_enc, z_0_pred)
        return y_pred

def compute_physics_loss(model, x_cond):
    """
    Computes L_mono = mean(ReLU(-∂Ŷ/∂H_canopy))
    x_cond must have requires_grad=True
    """
    if isinstance(model, torch.nn.DataParallel):
        model_module = model.module
    else:
        model_module = model
        
    # Channel 7 is Canopy Height (CHM)
    y_pred = model_module.decode(x_cond, num_steps=5) # Reduced steps to 5 for gradient tracking to avoid OOM
    
    # Compute gradients of sum(y_pred) w.r.t x_cond
    dY_dX = torch.autograd.grad(
        outputs=y_pred.sum(),
        inputs=x_cond,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Extract derivative w.r.t Canopy Height (index 7 out of 8)
    dY_dH = dY_dX[:, 7, :, :]
    
    # L_mono penalizes negative derivatives
    l_mono = F.relu(-dY_dH).mean()
    return l_mono
