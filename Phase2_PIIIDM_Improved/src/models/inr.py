"""
INR v2 — Spatial feature injection (fixes R² collapse)
Global pooling hata ke spatial bilinear sampling use kiya
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import numpy as np
from typing import List


def make_coord_grid(H, W, device):
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy], dim=-1).reshape(1, H * W, 2)


class SpatialFeatureSampler(nn.Module):
    """
    Bilinear sample from multi-scale feature maps at query coordinates.
    Preserves spatial structure — fixes global pooling collapse.
    student_chs = [32, 64, 128, 256] → project each to 64 → concat = 256
    """
    OUT_CH = 64

    def __init__(self, student_chs=[32, 64, 128, 256]):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, self.OUT_CH, 1, bias=False),
                nn.GroupNorm(8, self.OUT_CH),
                nn.GELU(),
            )
            for ch in student_chs
        ])
        self.feat_dim = self.OUT_CH * len(student_chs)  # 256

    def forward(self, feats, coords):
        """
        feats  : list of (B, C, H', W')
        coords : (B, N, 2)  in [-1,1]
        returns: (B, N, feat_dim)
        """
        B, N, _ = coords.shape
        sampled = []
        for feat, proj in zip(feats, self.projs):
            p = proj(feat)                                    # (B, 64, H', W')
            # grid_sample expects (B, 1, N, 2)
            grid = coords.unsqueeze(1)                        # (B, 1, N, 2)
            s = F.grid_sample(p, grid, align_corners=True,
                              mode='bilinear', padding_mode='border')
            # s: (B, 64, 1, N) → (B, N, 64)
            sampled.append(s.squeeze(2).permute(0, 2, 1))
        return torch.cat(sampled, dim=-1)                     # (B, N, 256)


class PositionalEncoding(nn.Module):
    def __init__(self, L=10):
        super().__init__()
        self.L = L
        freqs = 2.0 ** torch.arange(L).float() * np.pi
        self.register_buffer("freqs", freqs)

    @property
    def out_dim(self):
        return 4 * self.L  # 40

    def forward(self, coords):
        x = coords[..., 0:1]
        y = coords[..., 1:2]
        return torch.cat([
            (x * self.freqs).sin(), (x * self.freqs).cos(),
            (y * self.freqs).sin(), (y * self.freqs).cos(),
        ], dim=-1)


class ResMLPBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1  = nn.Linear(dim, dim * 2)
        self.fc2  = nn.Linear(dim * 2, dim)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        h = self.norm(x)
        h = self.fc2(self.drop(self.act(self.fc1(h))))
        return x + h


class SIRENINR(nn.Module):
    """
    Spatial INR v2:
    PE(40) + SpatialSample(256) → 296 → proj(256) → 4xResBlock → 1 → Tanh
    """
    PE_L   = 10
    HIDDEN = 256

    def __init__(self, student_chs=[32, 64, 128, 256]):
        super().__init__()
        self.pe      = PositionalEncoding(self.PE_L)
        self.sampler = SpatialFeatureSampler(student_chs)

        in_dim = self.pe.out_dim + self.sampler.feat_dim  # 40+256=296

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, self.HIDDEN),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([ResMLPBlock(self.HIDDEN) for _ in range(6)])
        self.out = nn.Sequential(
            nn.Linear(self.HIDDEN, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, student_feats, coords=None, H=256, W=256):
        B      = student_feats[0].shape[0]
        device = student_feats[0].device

        if coords is None:
            coords = make_coord_grid(H, W, device).expand(B, -1, -1)

        N   = coords.shape[1]
        pe  = self.pe(coords)                                  # (B, N, 40)
        sp  = self.sampler(student_feats, coords)              # (B, N, 256)

        x = self.input_proj(torch.cat([pe, sp], dim=-1))      # (B, N, 256)

        for block in self.blocks:
            if self.training and N > 32768:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        return self.out(x).reshape(B, H, W, 1).permute(0, 3, 1, 2)
