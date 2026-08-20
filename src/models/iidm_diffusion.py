import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_timestep_embedding(timesteps, embedding_dim):
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_emb_dim, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, t_emb):
        h = self.relu(self.conv1(x))
        h = h + self.t_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(h)
        return self.relu(h + self.shortcut(x))

class CrossAttentionMLP(nn.Module):
    def __init__(self, ch, cond_ch):
        super().__init__()
        self.cond_proj = nn.Conv2d(cond_ch, ch, 1)
        self.q = nn.Linear(ch, ch)
        self.k = nn.Linear(ch, ch)
        self.v = nn.Linear(ch, ch)
        self.out = nn.Linear(ch, ch)
        self.scale = ch ** -0.5
        
        self.mlp = nn.Sequential(
            nn.Conv2d(ch, ch, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 1)
        )

    def forward(self, x, cond):
        B, C, H, W = x.shape
        c = self.cond_proj(cond)
        
        x_flat = x.view(B, C, -1).transpose(1, 2)
        c_flat = c.view(B, C, -1).transpose(1, 2)
        
        q = self.q(x_flat)
        k = self.k(c_flat)
        v = self.v(c_flat)
        
        attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) * self.scale, dim=-1)
        attn_out = self.out(torch.bmm(attn, v))
        
        attn_out = attn_out.transpose(1, 2).view(B, C, H, W)
        fused = x + attn_out
        return fused + self.mlp(fused)

class ConditionalLatentUNet(nn.Module):
    """
    U-Net based latent predictor operating on z_t (256 ch, H/16 x W/16).
    For 64x64 patches, H/16 is 4x4. We use a lightweight architecture.
    """
    def __init__(self, latent_dim=256, cond_dim=512, t_emb_dim=128):
        super().__init__()
        self.t_mlp = nn.Sequential(
            nn.Linear(t_emb_dim, t_emb_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(t_emb_dim * 4, t_emb_dim)
        )
        self.t_emb_dim = t_emb_dim
        
        self.inc = nn.Conv2d(latent_dim, 128, 3, padding=1)
        
        # Encoder
        self.down1 = ResBlock(128, 256, t_emb_dim)
        self.attn1 = CrossAttentionMLP(256, cond_dim)
        self.pool = nn.MaxPool2d(2) # 4x4 -> 2x2
        
        # Mid
        self.mid = ResBlock(256, 256, t_emb_dim)
        self.attn_mid = CrossAttentionMLP(256, cond_dim)
        
        # Decoder
        self.up = nn.Upsample(scale_factor=2, mode='nearest') # 2x2 -> 4x4
        self.up1 = ResBlock(256 + 256, 128, t_emb_dim)
        self.attn_up1 = CrossAttentionMLP(128, cond_dim)
        
        self.outc = nn.Conv2d(128, latent_dim, 3, padding=1)

    def forward(self, z_t, t, cond):
        t_emb = get_timestep_embedding(t, self.t_emb_dim)
        t_emb = self.t_mlp(t_emb)
        
        x = self.inc(z_t)
        
        x1 = self.down1(x, t_emb)
        x1 = self.attn1(x1, cond)
        
        x2 = self.pool(x1)
        
        x2 = self.mid(x2, t_emb)
        # Adapt cond spatial size if needed
        cond_mid = F.adaptive_avg_pool2d(cond, x2.shape[-2:])
        x2 = self.attn_mid(x2, cond_mid)
        
        x_up = self.up(x2)
        x_up = torch.cat([x_up, x1], dim=1)
        
        x_up = self.up1(x_up, t_emb)
        x_up = self.attn_up1(x_up, cond)
        
        return self.outc(x_up)

def make_beta_schedule(T=1000):
    return torch.linspace(1e-4, 0.02, T)

class ConditionalLatentDiffusion(nn.Module):
    def __init__(self, T=1000):
        super().__init__()
        self.T = T
        self.denoiser = ConditionalLatentUNet()
        
        betas = make_beta_schedule(T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

    def forward(self, z_0, cond):
        """
        Forward process and noise prediction for training.
        z_0: initial latent vector from student encoder
        cond: semantic feature from teacher encoder (e.g., F_T^4)
        """
        B = z_0.shape[0]
        t = torch.randint(1, self.T + 1, (B,), device=z_0.device)
        noise = torch.randn_like(z_0)
        
        sqrt_alpha = self.sqrt_alphas_cumprod[t - 1].view(B, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t - 1].view(B, 1, 1, 1)
        
        z_t = sqrt_alpha * z_0 + sqrt_one_minus_alpha * noise
        
        noise_pred = self.denoiser(z_t, t, cond)
        
        loss = F.mse_loss(noise_pred, noise)
        return loss

    @torch.no_grad()
    def sample(self, z_0, cond, ddim_steps=20):
        """
        Generates z^* using DDIM sampling starting from z_T (pure noise)
        guided by conditional feature.
        """
        B, C, H, W = z_0.shape
        device = z_0.device
        
        z_t = torch.randn((B, C, H, W), device=device)
        
        step_size = self.T // ddim_steps
        timesteps = list(range(self.T, 0, -step_size))
        if timesteps[-1] != 1:
            timesteps.append(1)

        for i, t in enumerate(timesteps):
            t_next = timesteps[i + 1] if i + 1 < len(timesteps) else 0
            
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
            ab_t = self.alphas_cumprod[t - 1]
            ab_tm1 = self.alphas_cumprod[t_next - 1] if t_next > 0 else torch.ones(1, device=device)
            
            eps_pred = self.denoiser(z_t, t_tensor, cond)
            
            z0_pred = (z_t - (1 - ab_t).sqrt() * eps_pred) / (ab_t.sqrt() + 1e-8)
            
            if t_next > 0:
                z_t = ab_tm1.sqrt() * z0_pred + (1 - ab_tm1).sqrt() * eps_pred
            else:
                z_t = z0_pred
                
        return z_t
