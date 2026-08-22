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
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

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
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, t):
        x = self.conv(x)
        x = x + self.time_mlp(t)[..., None, None]
        return x, self.pool(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, time_emb_dim):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv((in_channels // 2) + skip_channels, out_channels)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))

    def forward(self, x, skip, t):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x + self.time_mlp(t)[..., None, None]

class KDUNet(nn.Module):
    """
    Paper-correct: student block4 = 256ch, teacher cond = 256ch
    concat(xt=256, cond=256) = 512ch input  ← NO zero padding
    """
    def __init__(self, in_channels=256, out_channels=256, time_dim=512):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        # concat(noisy_latent=256, cond=256) = 512
        self.down1 = DownBlock(512,  128,  time_dim)
        self.down2 = DownBlock(128,  256,  time_dim)
        self.down3 = DownBlock(256,  512,  time_dim)
        self.down4 = DownBlock(512,  1024, time_dim)

        self.bot1 = DoubleConv(1024, 1024)
        self.bot2 = DoubleConv(1024, 1024)

        self.up1 = UpBlock(1024, 1024, 512, time_dim)
        self.up2 = UpBlock(512,  512,  256, time_dim)
        self.up3 = UpBlock(256,  256,  128, time_dim)
        self.up4 = UpBlock(128,  128,  128, time_dim)

        self.final_conv = nn.Conv2d(128, out_channels, kernel_size=1)

    def forward(self, xt, t, cond):
        t_emb = self.time_mlp(t)
        x = torch.cat([xt, cond], dim=1)   # (B, 512, H', W') — clean!

        x1, p1 = self.down1(x,  t_emb)
        x2, p2 = self.down2(p1, t_emb)
        x3, p3 = self.down3(p2, t_emb)
        x4, p4 = self.down4(p3, t_emb)

        b = self.bot2(self.bot1(p4))

        u1 = self.up1(b,  x4, t_emb)
        u2 = self.up2(u1, x3, t_emb)
        u3 = self.up3(u2, x2, t_emb)
        u4 = self.up4(u3, x1, t_emb)

        return self.final_conv(u4)

if __name__ == "__main__":
    B, H, W = 2, 16, 16
    xt   = torch.randn(B, 256, H, W)
    cond = torch.randn(B, 256, H, W)
    t    = torch.randint(0, 1000, (B,))
    
    model = KDUNet()
    out   = model(xt, t, cond)
    
    print(f"Input  : {xt.shape}")
    print(f"Output : {out.shape}  ← should be (2, 256, 16, 16)")
    print(f"Params : {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print("✓ KDUNet 256ch — NO zero padding!")
