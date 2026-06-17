import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Applies Sinusoidal Positional Encoding with L frequencies (Paper default L=10)."""
    def __init__(self, L=10):
        super().__init__()
        self.L = L

    def forward(self, coords):
        # coords: (B, N, 2) where values are in [-1, 1]
        device = coords.device
        freq_bands = (2 ** torch.arange(self.L, dtype=torch.float32, device=device)) * math.pi
        
        # output shape will be (B, N, 2 * 2 * L) -> (B, N, 40)
        encoded = []
        for freq in freq_bands:
            encoded.append(torch.sin(coords * freq))
            encoded.append(torch.cos(coords * freq))
        
        return torch.cat(encoded, dim=-1)

class ImplicitNeuralRepr(nn.Module):
    """Maps PE(coords) + Latent Features -> Carbon Value."""
    def __init__(self, latent_dim=32, L=10, hidden_dim=256):
        super().__init__()
        self.pe = PositionalEncoding(L=L)
        
        # PE output is 40 dims (2 coords * 2 (sin/cos) * 10). Total input = 40 + latent_dim
        in_features = (2 * 2 * L) + latent_dim
        
        # 5-Layer MLP as per paper
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def get_coords(self, H, W, device):
        # Normalized coordinates [-1, 1]
        ys = torch.linspace(-1, 1, H, device=device)
        xs = torch.linspace(-1, 1, W, device=device)
        grid = torch.stack(torch.meshgrid(ys, xs, indexing='ij'), dim=-1) # (H, W, 2)
        return grid.reshape(-1, 2) # (H*W, 2)

    def forward(self, latents, H, W):
        # latents: (B, C, H, W) - VGG features
        B, C, _, _ = latents.shape
        device = latents.device
        
        coords = self.get_coords(H, W, device).unsqueeze(0).expand(B, -1, -1) # (B, H*W, 2)
        encoded_coords = self.pe(coords) # (B, H*W, 40)
        
        # Upsample latents to match target map resolution and flatten
        latents_up = F.interpolate(latents, size=(H, W), mode='bilinear', align_corners=False)
        latents_flat = latents_up.flatten(2).transpose(1, 2) # (B, H*W, C)
        
        # Concat PE and features
        mlp_in = torch.cat([encoded_coords, latents_flat], dim=-1) # (B, H*W, 40+C)
        
        carbon_pred = self.mlp(mlp_in) # (B, H*W, 1)
        return carbon_pred.transpose(1, 2).reshape(B, 1, H, W)
