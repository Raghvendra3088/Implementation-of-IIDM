import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class CrossAttention(nn.Module):
    """
    Flexible Cross-Attention module that maps context channels 
    to match the query dimensions dynamically.
    """
    def __init__(self, dim, context_dim, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(context_dim, dim)
        self.to_v = nn.Linear(context_dim, dim)
        self.to_out = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, context):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)          # (B, HW, C)
        ctx_flat = context.flatten(2).transpose(1, 2)  # (B, H_ctx*W_ctx, C_ctx)

        q = self.to_q(self.norm1(x_flat))
        k = self.to_k(ctx_flat)
        v = self.to_v(ctx_flat)

        q = q.reshape(B, -1, self.heads, C // self.heads).transpose(1, 2)
        k = k.reshape(B, -1, self.heads, C // self.heads).transpose(1, 2)
        v = v.reshape(B, -1, self.heads, C // self.heads).transpose(1, 2)

        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        out = x_flat + self.to_out(out)

        out = out + self.mlp(self.norm2(out))
        return out.transpose(1, 2).reshape(B, C, H, W)

class KDUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, time_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim), nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Encoder States
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, 64, 3, padding=1), nn.GELU())
        self.down1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.GELU())
        self.down2 = nn.MaxPool2d(2)
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.GELU())
        self.down3 = nn.MaxPool2d(2)
        
        # Bottleneck Layer with Cross-Attention (Context channel is 32 from StudentVGG stage3)
        self.bottleneck_conv1 = nn.Conv2d(256, 256, 3, padding=1)
        self.attn = CrossAttention(dim=256, context_dim=32, heads=4)
        self.bottleneck_conv2 = nn.Conv2d(256, 256, 3, padding=1)
        
        # Decoder States (Math-aligned channels to prevent OOM and shape mismatch)
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(128 + 256, 128, 3, padding=1), nn.GELU()) 
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(64 + 128, 64, 3, padding=1), nn.GELU())

        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(nn.Conv2d(32 + 64, 32, 3, padding=1), nn.GELU())
        
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, xt, t, s_feats):
        time_embed = self.time_mlp(t)
        
        # Forward Encoder
        e1 = self.enc1(xt)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        
        # Bottleneck handling
        b = self.bottleneck_conv1(self.down3(e3))
        b = b + time_embed.view(-1, 256, 1, 1)
        
        # Cross Attention using Student VGG stage 3 features as context map
        b = self.attn(b, s_feats[2])
        b = torch.relu(self.bottleneck_conv2(b))
        
        # Forward Decoder with exact skip connection shapes
        d1 = self.up1(b)
        d1 = torch.cat([d1, e3], dim=1) # 128 + 256 = 384 channels
        d1 = self.dec1(d1)
        
        d2 = self.up2(d1)
        d2 = torch.cat([d2, e2], dim=1) # 64 + 128 = 192 channels
        d2 = self.dec2(d2)

        d3 = self.up3(d2)
        d3 = torch.cat([d3, e1], dim=1) # 32 + 64 = 96 channels
        d3 = self.dec3(d3)
        
        return self.final_conv(d3)
