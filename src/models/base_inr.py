"""
Base IIDM paper - Implicit Representation module (exact paper).
Coordinate-based MLP as per paper Figure 4(b).
Fourier positional encoding + MLP on UNet decoder features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def make_coord_grid(H, W, device):
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    return torch.stack([gx, gy], dim=-1).reshape(1, H * W, 2)


class PositionalEncoding(nn.Module):
    """Fourier positional encoding — paper's coordinate encoding."""
    def __init__(self, L=10):
        super().__init__()
        freqs = 2.0 ** torch.arange(L).float() * np.pi
        self.register_buffer('freqs', freqs)

    @property
    def out_dim(self): return 4 * self.freqs.shape[0]   # 40

    def forward(self, coords):
        x = coords[..., 0:1] * self.freqs
        y = coords[..., 1:2] * self.freqs
        return torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=-1)


class BaseINR(nn.Module):
    """
    Paper implicit representation:
    - Global average pool on UNet features as context
    - Fourier PE on coordinates
    - 4 MLP blocks with ReLU + residual
    - Output: carbon density in [-1,1]
    """
    HIDDEN = 256
    PE_L   = 10

    def __init__(self, feat_ch=44):
        super().__init__()
        self.pe = PositionalEncoding(self.PE_L)
        self.feat_proj = nn.Linear(feat_ch, 256)
        in_dim = self.pe.out_dim + 256   # 40 + 256 = 296
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, self.HIDDEN), nn.ReLU()
        )
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.HIDDEN, self.HIDDEN * 2),
                nn.ReLU(),
                nn.Linear(self.HIDDEN * 2, self.HIDDEN),
            )
            for _ in range(4)
        ])
        self.out_head = nn.Sequential(
            nn.Linear(self.HIDDEN, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Tanh()
        )
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, feat_map, H=256, W=256, coords=None):
        """
        feat_map: (B, C, H', W') from UNet decoder
        Returns : (B, 1, H, W) carbon density
        """
        B      = feat_map.shape[0]
        device = feat_map.device

        if coords is None:
            coords = make_coord_grid(H, W, device).expand(B, -1, -1)

        N = coords.shape[1]

        # Global average pool (paper method)
        ctx = feat_map.mean(dim=[2, 3])           # (B, C)
        ctx = self.feat_proj(ctx)                 # (B, 256)
        ctx = ctx.unsqueeze(1).expand(-1, N, -1)  # (B, N, 256)

        pe  = self.pe(coords)                     # (B, N, 40)
        x   = self.input_proj(torch.cat([pe, ctx], dim=-1))
        for blk in self.blocks:
            x = x + blk(x)

        out = self.out_head(x)                    # (B, N, 1)
        return out.reshape(B, H, W, 1).permute(0, 3, 1, 2)   # (B,1,H,W)
