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
    def __init__(self, query_dim, context_dim, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = (query_dim // heads) ** -0.5
        self.to_q = nn.Linear(query_dim, query_dim)
        self.to_k = nn.Linear(context_dim, query_dim)
        self.to_v = nn.Linear(context_dim, query_dim)
        self.to_out = nn.Linear(query_dim, query_dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(query_dim),
            nn.Linear(query_dim, query_dim * 4), 
            nn.GELU(), 
            nn.Linear(query_dim * 4, query_dim)
        )
        self.norm1 = nn.LayerNorm(query_dim)

    def forward(self, x, context):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        ctx_flat = context.flatten(2).transpose(1, 2)

        q = self.to_q(self.norm1(x_flat))
        k = self.to_k(ctx_flat)
        v = self.to_v(ctx_flat)

        q = q.reshape(B, -1, self.heads, C // self.heads).transpose(1, 2)
        k = k.reshape(B, -1, self.heads, C // self.heads).transpose(1, 2)
        v = v.reshape(B, -1, self.heads, C // self.heads).transpose(1, 2)

        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        out = x_flat + self.to_out(out)

        out = out + self.mlp(out)
        return out.transpose(1, 2).reshape(B, C, H, W)

class KDUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, time_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim), nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # DOWN BLOCKS
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, 64, 3, padding=1), nn.GELU()) # 256
        self.down1 = nn.MaxPool2d(2) # 128
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.GELU()) # 128
        self.down2 = nn.MaxPool2d(2) # 64
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.GELU()) # 64
        self.down3 = nn.MaxPool2d(2) # 32
        self.enc4 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.GELU()) # 32
        self.down4 = nn.MaxPool2d(2) # 16 spatial resolution
        
        # BOTTLENECK (Safe memory zone)
        self.bot1 = nn.Conv2d(256, 256, 3, padding=1)
        # Student VGG features deepest is stage4 which has 64 channels
        self.attn = CrossAttention(query_dim=256, context_dim=64)
        self.bot2 = nn.Conv2d(256, 256, 3, padding=1)

        # UP BLOCKS (with skip connections)
        self.up1 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2) 
        self.dec1 = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1), nn.GELU())
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) 
        self.dec2 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.GELU())
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) 
        self.dec3 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.GELU())
        self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2) 
        self.dec4 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.GELU())

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x, time, s_feats):
        t = self.time_mlp(time)
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))
        
        b = self.bot1(self.down4(e4))
        b = b + t.view(-1, 256, 1, 1)
        
        # Extract deepest student feature map for Cross Attention
        context = torch.nn.functional.interpolate(s_feats[-1], size=b.shape[-2:], mode='bilinear')
        b = self.attn(b, context)
        b = torch.relu(self.bot2(b))
        
        d1 = self.up1(b)
        d1 = torch.cat([d1, e4], dim=1)
        d1 = self.dec1(d1)
        
        d2 = self.up2(d1)
        d2 = torch.cat([d2, e3], dim=1)
        d2 = self.dec2(d2)
        
        d3 = self.up3(d2)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)
        
        d4 = self.up4(d3)
        d4 = torch.cat([d4, e1], dim=1)
        d4 = self.dec4(d4)
        
        return self.final_conv(d4)
