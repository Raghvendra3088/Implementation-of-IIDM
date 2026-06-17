import torch
import torch.nn as nn
import torch.nn.functional as F

class SIRENLayer(nn.Module):
    """SIREN: Sinusoidal representation networks."""
    def __init__(self, in_f, out_f, is_first=False, omega=30.0):
        super().__init__()
        self.linear = nn.Linear(in_f, out_f)
        self.omega = omega
        self.is_first = is_first
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.linear.in_features, 1 / self.linear.in_features)
            else:
                bound = (6 / self.linear.in_features) ** 0.5 / self.omega
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega * self.linear(x))

class ImplicitNeuralRepr(nn.Module):
    def __init__(self, latent_dim=8, hidden=128, layers=4):
        super().__init__()
        self.net = nn.Sequential(
            SIRENLayer(2 + latent_dim, hidden, is_first=True),
            *[SIRENLayer(hidden, hidden) for _ in range(layers-2)],
            nn.Linear(hidden, 1)
        )

    def get_coords(self, H, W, device):
        ys = torch.linspace(-1, 1, H, device=device)
        xs = torch.linspace(-1, 1, W, device=device)
        grid = torch.stack(torch.meshgrid(ys, xs, indexing='ij'), dim=-1)
        return grid.reshape(-1, 2)

    def forward(self, features, H, W):
        B = features.shape[0]
        
        # Fix: Dynamically interpolate features to match high-res target coordinates (H, W)
        if features.shape[-2:] != (H, W):
            features = F.interpolate(features, size=(H, W), mode='bilinear', align_corners=False)
            
        coords = self.get_coords(H, W, features.device)
        coords = coords.unsqueeze(0).expand(B, -1, -1)

        feats_flat = features.flatten(2).transpose(1, 2)
        
        inp = torch.cat([coords, feats_flat], dim=-1)
        out = self.net(inp)
        return out.transpose(1, 2).reshape(B, 1, H, W)
