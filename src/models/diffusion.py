import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

def linear_beta_schedule(timesteps):
    return torch.linspace(0.0001, 0.02, timesteps)

class ImplicitNeuralRepresentation(nn.Module):
    def __init__(self, feature_dim=64, out_dim=1):
        super().__init__()
        in_dim = 2 + feature_dim 
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128), nn.GELU(),
            nn.Linear(128, 128), nn.GELU(),
            nn.Linear(128, out_dim)
        )
    def forward(self, coords, features):
        return self.mlp(torch.cat([coords, features], dim=-1))

class IIDM_Diffusion(nn.Module):
    def __init__(self, denoise_model, timesteps=1000):
        super().__init__()
        self.model = denoise_model
        self.timesteps = timesteps
        
        betas = linear_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

    def get_noise_schedule_value(self, vals, t, x_shape):
        batch_size = t.shape[0]
        out = vals.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start, t, noise=None):
        if noise is None: noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self.get_noise_schedule_value(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_noise_schedule_value(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_start, t, cond_features, structural_features, noise=None):
        if noise is None: noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        unet_input = torch.cat([x_noisy, structural_features], dim=1)
        predicted_noise = self.model(unet_input, t, cond_features)
        return F.mse_loss(noise, predicted_noise)

    @torch.no_grad()
    def p_sample(self, x, t, t_index, cond_features, structural_features):
        """Reverse process: one denoising step"""
        betas_t = self.get_noise_schedule_value(self.alphas, t, x.shape) # using alphas as a proxy for betas setup
        sqrt_one_minus_alphas_cumprod_t = self.get_noise_schedule_value(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.get_noise_schedule_value(self.alphas, t, x.shape))
        
        unet_input = torch.cat([x, structural_features], dim=1)
        model_mean = sqrt_recip_alphas_t * (x - betas_t * self.model(unet_input, t, cond_features) / sqrt_one_minus_alphas_cumprod_t)
        
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = betas_t # Simplified variance
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, cond_features, structural_features, image_size=256, batch_size=1):
        """Inference loop: Start from pure noise, denoise T steps"""
        device = cond_features.device
        x = torch.randn(batch_size, 1, image_size, image_size, device=device)
        
        for i in tqdm(reversed(range(0, self.timesteps)), desc='Denoising Sampling', total=self.timesteps):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t, i, cond_features, structural_features)
        return x
