import torch
import torch.nn as nn

class LatentDiffusionScheduler(nn.Module):
    """Handles noise addition/removal strictly in Latent Feature Space."""
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.timesteps = timesteps
        
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

    def _extract(self, a, t, x_shape):
        """Extracts appropriate coefficients at timestep t."""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start, t, noise=None):
        """
        Forward Pass: Add noise to LATENT FEATURES (not pixels).
        x_start: (B, C, H, W) student features
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alpha_bar_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha_bar_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        x_noisy = sqrt_alpha_bar_t * x_start + sqrt_one_minus_alpha_bar_t * noise
        return x_noisy, noise
