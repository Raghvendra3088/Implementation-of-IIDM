import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = F.silu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.silu(h + self.shortcut(x))

class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        return self.conv(x)

class Encoder(nn.Module):
    def __init__(self, in_channels=1, latent_channels=4):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, 64, 3, padding=1)
        
        self.down1 = nn.Sequential(ResBlock(64, 128), Downsample(128))   # 128x128
        self.down2 = nn.Sequential(ResBlock(128, 256), Downsample(256))  # 64x64
        self.down3 = nn.Sequential(ResBlock(256, 512), Downsample(512))  # 32x32
        
        self.mid = nn.Sequential(ResBlock(512, 512), ResBlock(512, 512))
        self.conv_out = nn.Conv2d(512, latent_channels * 2, 3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.mid(x)
        x = self.conv_out(x)
        return x

class Decoder(nn.Module):
    def __init__(self, latent_channels=4, out_channels=1):
        super().__init__()
        self.conv_in = nn.Conv2d(latent_channels, 512, 3, padding=1)
        
        self.mid = nn.Sequential(ResBlock(512, 512), ResBlock(512, 512))
        
        self.up1 = nn.Sequential(ResBlock(512, 256), Upsample(256))  # 64x64
        self.up2 = nn.Sequential(ResBlock(256, 128), Upsample(128))  # 128x128
        self.up3 = nn.Sequential(ResBlock(128, 64), Upsample(64))    # 256x256
        
        self.conv_out = nn.Conv2d(64, out_channels, 3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.mid(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.conv_out(x)
        return x

class CarbonVAE(nn.Module):
    def __init__(self, in_channels=1, latent_channels=4):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_channels)
        self.decoder = Decoder(latent_channels, in_channels)

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
