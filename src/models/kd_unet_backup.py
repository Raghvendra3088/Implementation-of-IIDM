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

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU()
        )

    def forward(self, x):
        return self.double_conv(x)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, t):
        x = self.conv(x)
        time_emb = self.time_mlp(t)[..., None, None]
        x = x + time_emb
        return x, self.pool(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, time_emb_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # Input to conv is the upsampled output (in_channels // 2) + the skip connection channels
        self.conv = DoubleConv((in_channels // 2) + skip_channels, out_channels)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

    def forward(self, x, skip, t):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        time_emb = self.time_mlp(t)[..., None, None]
        x = x + time_emb
        return x

class KDUNet(nn.Module):
    def __init__(self, in_channels=480, out_channels=480, time_dim=512):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        # Input is concatenated: noisy latent (480) + condition teacher feats (480) = 960
        self.down1 = DownBlock(in_channels * 2, 128, time_dim)
        self.down2 = DownBlock(128, 256, time_dim)
        self.down3 = DownBlock(256, 512, time_dim)
        self.down4 = DownBlock(512, 1024, time_dim)

        self.bot1 = DoubleConv(1024, 1024)
        self.bot2 = DoubleConv(1024, 1024)

        self.up1 = UpBlock(in_channels=1024, skip_channels=1024, out_channels=512, time_emb_dim=time_dim)
        self.up2 = UpBlock(in_channels=512,  skip_channels=512,  out_channels=256, time_emb_dim=time_dim)
        self.up3 = UpBlock(in_channels=256,  skip_channels=256,  out_channels=128, time_emb_dim=time_dim)
        self.up4 = UpBlock(in_channels=128,  skip_channels=128,  out_channels=128, time_emb_dim=time_dim)

        self.final_conv = nn.Conv2d(128, out_channels, kernel_size=1)

    def forward(self, xt, t, cond):
        t_emb = self.time_mlp(t)
        
        # Concat noisy student features and teacher condition
        x = torch.cat([xt, cond], dim=1)

        x1, p1 = self.down1(x, t_emb)
        x2, p2 = self.down2(p1, t_emb)
        x3, p3 = self.down3(p2, t_emb)
        x4, p4 = self.down4(p3, t_emb)

        b = self.bot1(p4)
        b = self.bot2(b)

        u1 = self.up1(b, x4, t_emb)
        u2 = self.up2(u1, x3, t_emb)
        u3 = self.up3(u2, x2, t_emb)
        u4 = self.up4(u3, x1, t_emb)

        return self.final_conv(u4)

if __name__ == "__main__":
    # Test block with dummy data matching latent dimensions (B, 480, H, W)
    batch_size = 2
    latent_channels = 480
    h, w = 64, 64  # Downscaled spatial size of latent features
    
    xt = torch.randn(batch_size, latent_channels, h, w)
    cond = torch.randn(batch_size, latent_channels, h, w)
    t = torch.randint(0, 1000, (batch_size,))

    unet = KDUNet(in_channels=latent_channels, out_channels=latent_channels)
    out = unet(xt, t, cond)
    
    print(f"Input xt shape   : {xt.shape}")
    print(f"Condition shape  : {cond.shape}")
    print(f"Timestep shape   : {t.shape}")
    print(f"Output shape     : {out.shape}  (expected matching xt shape) ✓")
    print(f"UNet params      : {sum(p.numel() for p in unet.parameters())/1e6:.2f}M")
    print("✓ KD-UNet ready!")
