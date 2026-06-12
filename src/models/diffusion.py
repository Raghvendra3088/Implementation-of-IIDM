import torch
import torch.nn as nn
import torch.nn.functional as F

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

class ImplicitNeuralRepresentation(nn.Module):
    def __init__(self, feature_dim=64, out_dim=1):
        super().__init__()
        in_dim = 2 + feature_dim 
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, coords, features):
        x = torch.cat([coords, features], dim=-1)
        return self.mlp(x)

class IIDM_Diffusion(nn.Module):
    def __init__(self, denoise_model, timesteps=1000):
        super().__init__()
        self.model = denoise_model
        self.timesteps = timesteps
        
        betas = linear_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        
        self.inr = ImplicitNeuralRepresentation(feature_dim=64)

    def get_noise_schedule_value(self, vals, t, x_shape):
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = self.get_noise_schedule_value(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_noise_schedule_value(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_start, t, cond_features, structural_features, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
            
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        
        unet_input = torch.cat([x_noisy, structural_features], dim=1)
        
        predicted_noise = self.model(unet_input, t, cond_features)
        
        loss = F.mse_loss(noise, predicted_noise)
        return loss

if __name__ == "__main__":
    pass
