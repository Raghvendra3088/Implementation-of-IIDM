import torch
import torch.nn as nn

class DiffusionScheduler(nn.Module):
    """Diffusion process happening in the Latent/Feature Space."""
    def __init__(self, T=1000):
        super().__init__()
        self.T = T
        # Beta schedule: linear 1e-4 to 0.02
        betas = torch.linspace(1e-4, 0.02, T)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('alphas_cumprod', alphas_cumprod)

    def q_sample(self, x_0, t, noise=None):
        """
        Forward process: Adds noise to latent features x_0
        x_0 here is Student VGG Features, NOT the Carbon Map.
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        batch_size = t.shape[0]
        sqrt_a = self.sqrt_alphas_cumprod.gather(-1, t).reshape(batch_size, 1, 1, 1)
        sqrt_one_minus_a = self.sqrt_one_minus_alphas_cumprod.gather(-1, t).reshape(batch_size, 1, 1, 1)
        
        x_t = sqrt_a * x_0 + sqrt_one_minus_a * noise
        return x_t, noise

    @torch.no_grad()
    def ddim_sample(self, model, x_T, condition, steps=50):
        """Faster inference using DDIM on the latent space."""
        device = x_T.device
        b = x_T.shape[0]
        x_t = x_T
        
        step_size = self.T // steps
        timesteps = torch.arange(self.T - 1, -1, -step_size, device=device)
        
        for i, t_val in enumerate(timesteps):
            t = torch.full((b,), t_val, device=device, dtype=torch.long)
            
            # Predict noise using KD-UNet conditioned on Teacher features
            pred_noise = model(x_t, t, condition)
            
            # Reconstruct original feature (x_0)
            alpha_bar = self.alphas_cumprod[t_val]
            sqrt_alpha_bar = torch.sqrt(alpha_bar)
            sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)
            
            pred_x0 = (x_t - sqrt_one_minus_alpha_bar * pred_noise) / sqrt_alpha_bar
            
            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                alpha_bar_prev = self.alphas_cumprod[t_prev]
                x_t = torch.sqrt(alpha_bar_prev) * pred_x0 + torch.sqrt(1 - alpha_bar_prev) * pred_noise
            else:
                x_t = pred_x0
                
        return x_t # This is the fully denoised latent feature map
