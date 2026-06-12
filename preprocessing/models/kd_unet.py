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

class KDUnet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, time_dim=256):
        super().__init__()
        # in_channels = 3 (1 ch Noisy Carbon + 1 ch DEM + 1 ch Canopy)
        
        # Time Embedding Multi-Layer Perceptron
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Encoder (Downsampling)
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, 64, 3, padding=1), nn.GELU())
        self.down1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.GELU())
        self.down2 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bot1 = nn.Conv2d(128, 256, 3, padding=1)
        self.bot2 = nn.Conv2d(256, 256, 3, padding=1)
        
        # Decoder (Upsampling with Skip Connections)
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        # 256 channels in dec1 because 128 (from up1) + 128 (from skip connection enc2)
        self.dec1 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.GELU()) 
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        # 128 channels in dec2 because 64 (from up2) + 64 (from skip connection enc1)
        self.dec2 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.GELU())
        
        # Final projection to output predicted noise (1 channel)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x, time, vgg_features=None):
        # 1. Embed the timestep
        t = self.time_mlp(time)
        
        # 2. Encoder Pass
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        
        # 3. Bottleneck Pass
        b = self.bot1(self.down2(e2))
        
        # Inject time embedding into the bottleneck
        b = b + t.view(-1, 256, 1, 1) 
        b = torch.relu(self.bot2(b))
        
        # NOTE: vgg_features fusion will be implemented here in the methodology upgrade phase
        
        # 4. Decoder Pass (with concatenations)
        d1 = self.up1(b)
        d1 = torch.cat([d1, e2], dim=1) # Skip connection
        d1 = self.dec1(d1)
        
        d2 = self.up2(d1)
        d2 = torch.cat([d2, e1], dim=1) # Skip connection
        d2 = self.dec2(d2)
        
        return self.final_conv(d2)

if __name__ == "__main__":
    # Local verification loop
    batch_size = 2
    dummy_x = torch.randn(batch_size, 3, 256, 256) # [Noisy Carbon, DEM, Canopy]
    dummy_t = torch.randint(0, 1000, (batch_size,)) # Random timesteps
    
    model = KDUnet()
    predicted_noise = model(dummy_x, dummy_t)
    
    print("Local Check Successful:")
    print(f"Input shape: {dummy_x.shape}")
    print(f"Predicted noise shape: {predicted_noise.shape} (Should match Carbon Target)")
