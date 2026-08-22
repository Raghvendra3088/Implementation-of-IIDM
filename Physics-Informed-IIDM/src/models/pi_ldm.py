import torch
import torch.nn as nn
import torch.nn.functional as F

class PILDM(nn.Module):
    def __init__(self, vae, encoder, unet, diffusion_scheduler):
        super().__init__()
        self.vae = vae
        self.encoder = encoder
        self.unet = unet
        self.scheduler = diffusion_scheduler
        
        # Freeze VAE and Encoder (if KD is pre-trained)
        for p in self.vae.parameters():
            p.requires_grad = False
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, x_cond, y_target, t, noise=None):
        """
        x_cond:   (B, 6, H, W)   Input condition (Satellite + DEM + Canopy)
        y_target: (B, 1, H, W)   Target carbon map
        t:        (B,)           Timesteps
        """
        # Encode Condition
        with torch.no_grad():
            f_enc, f_proj, _ = self.encoder(x_cond)
            # f_proj has scales for the U-Net cross-attention or concat
            
            # Encode Target into Latent Space
            z_0, _ = self.vae.encode(y_target)
            
        # Forward diffusion
        z_t, noise_true = self.scheduler.q_sample(z_0, t, noise)
        
        # Denoising
        noise_pred = self.unet(z_t, t, f_proj)
        
        loss_diff = F.l1_loss(noise_pred, noise_true)
        return loss_diff
        
    def predict(self, x_cond, num_steps=50):
        """DDIM sampling for inference"""
        B = x_cond.shape[0]
        device = x_cond.device
        
        with torch.no_grad():
            f_enc, f_proj, _ = self.encoder(x_cond)
            z_T = torch.randn(B, 4, 32, 32, device=device)
            z_0_pred = self.scheduler.ddim_sample(self.unet, z_T, f_proj, steps=num_steps)
            y_pred = self.vae.decode(z_0_pred)
        return y_pred
    
    def decode(self, x_cond, num_steps=50):
        """For physics loss: Requires grad enabled through decoder"""
        B = x_cond.shape[0]
        device = x_cond.device
        
        # Get conditions
        f_enc, f_proj, _ = self.encoder(x_cond)
        z_T = torch.randn(B, 4, 32, 32, device=device)
        
        # We need to trace gradients from x_cond (specifically CHM channel) through UNet and VAE
        # Standard DDIM sampling isn't fully differentiable if we detach, but we must NOT detach here
        z_0_pred = self.scheduler.ddim_sample(self.unet, z_T, f_proj, steps=num_steps)
        y_pred = self.vae.decode(z_0_pred)
        return y_pred

def compute_physics_loss(model, x_cond):
    """
    Computes L_mono = mean(ReLU(-∂Ŷ/∂H_canopy))
    x_cond must have requires_grad=True
    """
    # Channel 5 is Canopy Height (CHM)
    y_pred = model.decode(x_cond, num_steps=10) # Less steps for gradient tracking to avoid OOM
    
    # Compute gradients of sum(y_pred) w.r.t x_cond
    dY_dX = torch.autograd.grad(
        outputs=y_pred.sum(),
        inputs=x_cond,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Extract derivative w.r.t Canopy Height (Channel 5)
    dY_dH = dY_dX[:, 5, :, :]
    
    # L_mono penalizes negative derivatives
    l_mono = F.relu(-dY_dH).mean()
    return l_mono
