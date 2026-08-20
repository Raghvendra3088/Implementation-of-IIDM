import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs=10):
        super().__init__()
        self.num_freqs = num_freqs
        # Frequencies: 2^0 * pi, 2^1 * pi, ..., 2^{L-1} * pi
        self.freqs = 2 ** torch.arange(num_freqs) * torch.pi

    def forward(self, coords):
        """
        coords: (B, N, 2) where N is number of points, 2 for (x, y)
        returns: (B, N, 2 * 2 * num_freqs) -> (B, N, 4 * num_freqs)
        """
        B, N, _ = coords.shape
        freqs = self.freqs.to(coords.device)
        
        # coords: (B, N, 2, 1)
        # freqs: (1, 1, 1, L)
        args = coords.unsqueeze(-1) * freqs.view(1, 1, 1, -1)
        
        # (B, N, 2, L)
        sin_args = torch.sin(args)
        cos_args = torch.cos(args)
        
        # Concatenate sin and cos along the last dimension: (B, N, 2, 2*L)
        encoded = torch.cat([sin_args, cos_args], dim=-1)
        
        # Flatten the last two dimensions: (B, N, 4*L)
        return encoded.view(B, N, -1)


class ImplicitNeuralDecoder(nn.Module):
    def __init__(self, latent_dim=256, num_freqs=10, hidden_dim=256, num_layers=4):
        super().__init__()
        self.pos_encoder = PositionalEncoding(num_freqs=num_freqs)
        
        coord_dim = 4 * num_freqs
        in_dim = latent_dim + coord_dim
        
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, z_star, coords):
        """
        z_star: (B, C, H_z, W_z) Refined latent feature grid from diffusion
        coords: (B, N, 2) Normalized coordinates in range [-1, 1]
        
        Returns:
        preds: (B, N, 1) Predicted carbon density
        """
        B, C, H_z, W_z = z_star.shape
        _, N, _ = coords.shape
        
        # 1. Sample latent features at coordinates using grid_sample
        # grid_sample expects grid of shape (B, H_out, W_out, 2)
        grid = coords.view(B, 1, N, 2)
        
        # Sampled features: (B, C, 1, N)
        sampled_features = F.grid_sample(z_star, grid, mode='bilinear', padding_mode='border', align_corners=False)
        
        # Transpose to (B, N, C)
        sampled_features = sampled_features.view(B, C, N).transpose(1, 2)
        
        # 2. Positional Encoding
        encoded_coords = self.pos_encoder(coords) # (B, N, 4*L)
        
        # 3. Concatenate and pass through MLP
        mlp_input = torch.cat([sampled_features, encoded_coords], dim=-1) # (B, N, C + 4*L)
        
        preds = self.mlp(mlp_input) # (B, N, 1)
        return preds

def generate_grid_coords(H, W, device):
    """
    Generates a flattened grid of normalized coordinates [-1, 1] for an HxW image.
    """
    y = torch.linspace(-1, 1, H, device=device)
    x = torch.linspace(-1, 1, W, device=device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    coords = torch.stack([xx, yy], dim=-1) # (H, W, 2)
    return coords.view(1, H * W, 2) # (1, H*W, 2)

class StandardDecoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        # Upsample 4x4 -> 64x64
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )

    def forward(self, z_star, coords=None):
        # coords is ignored, kept for compatibility with train script
        # z_star is (B, 256, 4, 4)
        B = z_star.shape[0]
        out = self.net(z_star) # (B, 1, 64, 64)
        # return as (B, N, 1) to match INR output format
        return out.view(B, -1, 1)
