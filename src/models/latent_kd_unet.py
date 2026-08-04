import torch
import torch.nn as nn
import math

def sinusoidal_emb(t, dim):
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device).float() / half
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([args.sin(), args.cos()], dim=-1)

class TimeEmb(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )
    def forward(self, t):
        return self.mlp(sinusoidal_emb(t, self.dim))

class CrossAttentionMLP(nn.Module):
    """
    Cross-attention between u (latent UNet feature) and f (CNN student feature).
    u: [B, C, H_u, W_u] (e.g. 8x8)
    f: [B, C_cond, H_f, W_f] (e.g. 64x64)
    """
    def __init__(self, ch, cond_ch):
        super().__init__()
        self.cond_proj = nn.Conv2d(cond_ch, ch, 1)
        self.q   = nn.Linear(ch, ch)
        self.k   = nn.Linear(ch, ch)
        self.v   = nn.Linear(ch, ch)
        self.out = nn.Linear(ch, ch)
        self.scale = ch ** -0.5
        
        self.mlp = nn.Sequential(
            nn.Conv2d(ch, ch, 1),
            nn.ReLU(),
            nn.Conv2d(ch, ch, 1)
        )

    def forward(self, u, f):
        B, C, H_u, W_u = u.shape
        _, _, H_f, W_f = f.shape
        
        f_r = self.cond_proj(f)  # (B, C, H_f, W_f)
        
        u_flat = u.view(B, C, -1).transpose(1, 2)  # (B, H_u*W_u, C)
        f_flat = f_r.view(B, C, -1).transpose(1, 2) # (B, H_f*W_f, C)
        
        q = self.q(u_flat)
        k = self.k(f_flat)
        v = self.v(f_flat)
        
        attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) * self.scale, dim=-1)
        attn_out = self.out(torch.bmm(attn, v))
        
        attn_out = attn_out.transpose(1, 2).view(B, C, H_u, W_u)
        fused = u + attn_out
        return fused + self.mlp(fused)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_emb_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_emb_dim, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x, t_emb):
        h = self.relu(self.conv1(x))
        h = h + self.t_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(h)
        return self.relu(h + self.shortcut(x))

class LatentKDUNet12(nn.Module):
    """
    A unified model that operates on the 8x8 latent space.
    It integrates the 12 features from KDStudent12 via CrossAttention.
    Since 8x8 is very small, we use a 3-level UNet: 8x8 -> 4x4 -> 8x8.
    """
    def __init__(self, in_channels=4, student_channels=None):
        super().__init__()
        if student_channels is None:
            # PRITHVI_STUDENT_CH_12
            student_channels = [8, 12, 14, 16, 17, 19, 20, 21, 22, 23, 24, 24]
            
        self.time_mlp = TimeEmb(128)
        
        self.inc = nn.Conv2d(in_channels, 64, 3, padding=1)
        
        # Encoder (8x8)
        self.down1_res = ResBlock(64, 128)
        self.down1_attn = CrossAttentionMLP(128, student_channels[0])
        
        self.down2_res = ResBlock(128, 128)
        self.down2_attn = CrossAttentionMLP(128, student_channels[1])
        
        # Downsample to 4x4
        self.pool = nn.MaxPool2d(2)
        
        # Middle (4x4) - We inject all remaining middle features here
        self.mid_blocks = nn.ModuleList()
        for i in range(2, 10):
            self.mid_blocks.append(nn.ModuleDict({
                'res': ResBlock(128, 128),
                'attn': CrossAttentionMLP(128, student_channels[i])
            }))
            
        # Upsample to 8x8
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        
        # Decoder (8x8)
        self.up1_res = ResBlock(128 + 128, 128) # +128 for skip connection from down2
        self.up1_attn = CrossAttentionMLP(128, student_channels[10])
        
        self.up2_res = ResBlock(128 + 128, 64)  # +128 for skip connection from down1
        self.up2_attn = CrossAttentionMLP(64, student_channels[11])
        
        self.outc = nn.Conv2d(64, in_channels, 3, padding=1)

    def forward(self, z, t, student_feats):
        t_emb = self.time_mlp(t)
        
        x = self.inc(z)
        
        # Encoder
        x1 = self.down1_res(x, t_emb)
        x1 = self.down1_attn(x1, student_feats[0])
        
        x2 = self.down2_res(x1, t_emb)
        x2 = self.down2_attn(x2, student_feats[1])
        
        # Middle
        h = self.pool(x2)
        for i in range(8):
            h = self.mid_blocks[i]['res'](h, t_emb)
            h = self.mid_blocks[i]['attn'](h, student_feats[i+2])
            
        # Decoder
        h = self.up(h)
        h = torch.cat([h, x2], dim=1) # Skip connection
        h = self.up1_res(h, t_emb)
        h = self.up1_attn(h, student_feats[10])
        
        h = torch.cat([h, x1], dim=1) # Skip connection
        h = self.up2_res(h, t_emb)
        h = self.up2_attn(h, student_feats[11])
        
        return self.outc(h)
