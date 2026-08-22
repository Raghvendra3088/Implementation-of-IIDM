import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        h = F.silu(self.norm1(self.conv1(x)))
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.shortcut(x)

class CarbonVAE(nn.Module):
    def __init__(self, in_channels=1, latent_channels=4, base_channels=64):
        super().__init__()
        # Encoder (downsamples 3 times: 256 -> 128 -> 64 -> 32)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            ResBlock(base_channels, base_channels),
            nn.Conv2d(base_channels, base_channels, 4, stride=2, padding=1), # 128
            ResBlock(base_channels, base_channels*2),
            nn.Conv2d(base_channels*2, base_channels*2, 4, stride=2, padding=1), # 64
            ResBlock(base_channels*2, base_channels*4),
            nn.Conv2d(base_channels*4, base_channels*4, 4, stride=2, padding=1), # 32
            nn.Conv2d(base_channels*4, latent_channels*2, 3, padding=1) # mean and logvar
        )

        # Decoder (upsamples 3 times: 32 -> 64 -> 128 -> 256)
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, base_channels*4, 3, padding=1),
            ResBlock(base_channels*4, base_channels*4),
            nn.ConvTranspose2d(base_channels*4, base_channels*2, 4, stride=2, padding=1), # 64
            ResBlock(base_channels*2, base_channels*2),
            nn.ConvTranspose2d(base_channels*2, base_channels, 4, stride=2, padding=1), # 128
            ResBlock(base_channels, base_channels),
            nn.ConvTranspose2d(base_channels, base_channels, 4, stride=2, padding=1), # 256
            nn.Conv2d(base_channels, in_channels, 3, padding=1)
        )

    def encode(self, x):
        h = self.encoder(x)
        mean, logvar = torch.chunk(h, 2, dim=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_rec = self.decode(z)
        return x_rec, mean, logvar

def vae_loss(x_rec, x, mean, logvar, beta=0.001):
    recon_loss = F.l1_loss(x_rec, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss
