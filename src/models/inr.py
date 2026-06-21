"""
INR — Implicit Neural Representation (SIREN)
=============================================
Paper: IIDM Section 3.2 — Core Contribution

SIREN maps (coordinates + VGG features) → carbon stock value
Uses sinusoidal activations (sin) instead of ReLU — critical for
representing continuous spatial signals (Sitzmann et al. 2020)

Architecture:
    coords (x,y) ∈ [-1,1]
        ↓ Positional Encoding (L=10 frequencies) → dim 80
        ↓ concat with pooled VGG student features → dim 80+480=560
        ↓ SIREN Layer 1: Linear(560, 256), sin, w0=30
        ↓ SIREN Layer 2: Linear(256, 256), sin
        ↓ SIREN Layer 3: Linear(256, 256), sin
        ↓ SIREN Layer 4: Linear(256, 256), sin
        ↓ Output Layer : Linear(256, 1)
        ↓ carbon value per coordinate

Input  : coords (B, H*W, 2), student_feats list of 4
Output : (B, 1, H, W) carbon map in [-1, 1]
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List


# ══════════════════════════════════════════════════════════════════════════════
# POSITIONAL ENCODING
# ══════════════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Fourier positional encoding for 2D coordinates.
    Paper uses L=10 frequency levels.

    PE(x) = [sin(2^0 π x), cos(2^0 π x),
             sin(2^1 π x), cos(2^1 π x),
             ...
             sin(2^9 π x), cos(2^9 π x)]

    For (x, y) pair: output dim = 2 × 2L = 2 × 20 = 40 per coord
    Total for (x,y) = 80
    """

    def __init__(self, L: int = 10):
        super().__init__()
        self.L = L
        # Precompute frequency bands: [2^0, 2^1, ..., 2^(L-1)]
        freqs = 2.0 ** torch.arange(L).float() * np.pi
        self.register_buffer("freqs", freqs)    # (L,)

    @property
    def out_dim(self) -> int:
        # x: sin+cos = 2L, y: sin+cos = 2L → total 4L
        return 4 * self.L  # = 40 for L=10 per coord, 80 total for (x,y)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords : (B, N, 2)  x,y in [-1, 1]
        Returns:
            pe     : (B, N, 4L)
        """
        x = coords[..., 0:1]   # (B, N, 1)
        y = coords[..., 1:2]   # (B, N, 1)

        # (B, N, 1) × (L,) → (B, N, L)
        x_freq = x * self.freqs
        y_freq = y * self.freqs

        pe = torch.cat([
            x_freq.sin(), x_freq.cos(),
            y_freq.sin(), y_freq.cos(),
        ], dim=-1)              # (B, N, 4L)

        return pe


# ══════════════════════════════════════════════════════════════════════════════
# COORDINATE GRID GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def make_coord_grid(H: int, W: int,
                    device: torch.device) -> torch.Tensor:
    """
    Generate normalized 2D coordinate grid for a patch of size H×W.

    Returns:
        coords : (1, H*W, 2)  values in [-1, 1]
    """
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")   # (H, W)
    coords = torch.stack([grid_x, grid_y], dim=-1)            # (H, W, 2)
    coords = coords.reshape(1, H * W, 2)                      # (1, H*W, 2)
    return coords


# ══════════════════════════════════════════════════════════════════════════════
# SIREN LAYER
# ══════════════════════════════════════════════════════════════════════════════

class SIRENLayer(nn.Module):
    """
    Single SIREN layer: Linear → sin activation.

    Weight initialization (Sitzmann et al. 2020):
        First layer : U(-1/in_dim, 1/in_dim) × w0
        Hidden layer: U(-√(6/in_dim), √(6/in_dim))

    w0 = 30 for first layer (paper default) controls frequency.
    """

    def __init__(self, in_dim: int, out_dim: int,
                 is_first: bool = False, w0: float = 30.0):
        super().__init__()
        self.in_dim   = in_dim
        self.is_first = is_first
        self.w0       = w0
        self.linear   = nn.Linear(in_dim, out_dim)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.in_dim
            else:
                bound = np.sqrt(6.0 / self.in_dim)
            self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * self.linear(x))


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE POOLING — VGG features → coordinate-level conditioning
# ══════════════════════════════════════════════════════════════════════════════

class FeaturePooling(nn.Module):
    """
    Pool multi-scale VGG student features into a single vector
    per sample for conditioning the SIREN MLP.

    Strategy: global average pool each scale → concat → Linear projection

    Student feature channels: [32, 64, 128, 256] → total 480
    Projected to: feat_dim (default 480 kept as-is)
    """

    def __init__(self, student_chs: List[int] = [32, 64, 128, 256],
                 feat_dim: int = 480):
        super().__init__()
        in_dim = sum(student_chs)   # 480
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.SiLU(),
        )
        self.feat_dim = feat_dim

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            feats : list of 4 feature maps from StudentVGG
        Returns:
            pooled : (B, feat_dim)
        """
        pooled = [self.pool(f).flatten(1) for f in feats]  # 4 × (B, C)
        concat = torch.cat(pooled, dim=1)                  # (B, 480)
        return self.proj(concat)                            # (B, feat_dim)


# ══════════════════════════════════════════════════════════════════════════════
# SIREN INR — FULL MODEL
# ══════════════════════════════════════════════════════════════════════════════

class SIRENINR(nn.Module):
    """
    Implicit Neural Representation using SIREN.
    Paper IIDM Section 3.2.

    Maps (coordinate_PE + pooled_VGG_features) → carbon stock

    Full pipeline:
        coords (B,N,2)
            → PositionalEncoding       (B, N, 40)
            → concat feat_vec (B,N, 40+480=520)
            → SIRENLayer(520→256, is_first=True, w0=30)
            → SIRENLayer(256→256)
            → SIRENLayer(256→256)
            → SIRENLayer(256→256)
            → Linear(256→1)            (B, N, 1)
            → reshape                  (B, 1, H, W)
    """

    HIDDEN_DIM  = 256
    N_HIDDEN    = 4      # number of hidden SIREN layers (paper: 4)
    PE_L        = 10     # positional encoding frequencies
    FEAT_DIM    = 480    # pooled VGG feature dim
    W0          = 30.0   # first layer frequency scale

    def __init__(self,
                 student_chs: List[int] = [32, 64, 128, 256]):
        super().__init__()

        self.pe       = PositionalEncoding(L=self.PE_L)
        self.feat_pool = FeaturePooling(student_chs, self.FEAT_DIM)

        in_dim = self.pe.out_dim + self.FEAT_DIM  # 40 + 480 = 520

        # Build SIREN layers
        layers = []

        # First SIREN layer (w0=30, special init)
        layers.append(SIRENLayer(in_dim, self.HIDDEN_DIM,
                                 is_first=True, w0=self.W0))

        # Hidden SIREN layers
        for _ in range(self.N_HIDDEN - 1):
            layers.append(SIRENLayer(self.HIDDEN_DIM, self.HIDDEN_DIM,
                                     is_first=False, w0=self.W0))

        self.siren = nn.ModuleList(layers)

        # Output layer — no sin activation, direct linear
        self.out = nn.Linear(self.HIDDEN_DIM, 1)
        nn.init.uniform_(self.out.weight,
                         -np.sqrt(6.0 / self.HIDDEN_DIM),
                          np.sqrt(6.0 / self.HIDDEN_DIM))

    def forward(self,
                student_feats: List[torch.Tensor],
                coords: torch.Tensor = None,
                H: int = 256, W: int = 256) -> torch.Tensor:
        """
        Args:
            student_feats : list of 4 tensors from StudentVGG
            coords        : (B, H*W, 2) or None (auto-generated)
            H, W          : patch spatial size (used if coords is None)

        Returns:
            carbon_map : (B, 1, H, W)  in [-1, 1]
        """
        B = student_feats[0].shape[0]
        device = student_feats[0].device

        # ── 1. Generate coordinate grid if not provided ────────────────────
        if coords is None:
            coords = make_coord_grid(H, W, device)          # (1, H*W, 2)
            coords = coords.expand(B, -1, -1)               # (B, H*W, 2)

        N = coords.shape[1]

        # ── 2. Positional encoding ─────────────────────────────────────────
        pe = self.pe(coords)                                 # (B, N, 80)

        # ── 3. Pool VGG features → (B, feat_dim) ──────────────────────────
        feat_vec = self.feat_pool(student_feats)             # (B, 480)

        # Expand to all coordinates: (B, 480) → (B, N, 480)
        feat_vec = feat_vec.unsqueeze(1).expand(-1, N, -1)  # (B, N, 480)

        # ── 4. Concatenate PE + features ───────────────────────────────────
        x = torch.cat([pe, feat_vec], dim=-1)               # (B, N, 560)

        # ── 5. SIREN forward pass ──────────────────────────────────────────
        for layer in self.siren:
            x = layer(x)                                     # (B, N, 256)

        # ── 6. Output projection ───────────────────────────────────────────
        x = self.out(x)                                      # (B, N, 1)

        # ── 7. Reshape to spatial map ──────────────────────────────────────
        carbon_map = x.reshape(B, H, W, 1).permute(0, 3, 1, 2)  # (B,1,H,W)

        return carbon_map


# ══════════════════════════════════════════════════════════════════════════════
# SANITY CHECK
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
    from src.models.kd_vgg import StudentVGG

    print("=" * 50)
    print("  INR (SIREN) Sanity Check")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device : {device}")

    B, H, W = 2, 256, 256
    dummy_input = torch.randn(B, 6, H, W).to(device)

    # Get student features
    student = StudentVGG(in_channels=6).to(device)
    with torch.no_grad():
        s_feats = student(dummy_input)

    print("\n  Student feature shapes:")
    for i, f in enumerate(s_feats):
        print(f"    Block {i+1}: {tuple(f.shape)}")

    # INR forward
    inr = SIRENINR(student_chs=[32, 64, 128, 256]).to(device)

    with torch.no_grad():
        carbon = inr(s_feats, H=H, W=W)

    print(f"\n  INR output shape : {tuple(carbon.shape)}")
    print(f"  Output range     : [{carbon.min():.4f}, {carbon.max():.4f}]")
    print(f"  NaN in output    : {torch.isnan(carbon).any().item()}")
    print(f"  Inf in output    : {torch.isinf(carbon).any().item()}")

    # Parameter count
    inr_params = sum(p.numel() for p in inr.parameters()) / 1e6
    print(f"\n  INR params : {inr_params:.2f}M")

    # PE check
    pe = PositionalEncoding(L=10)
    coords = make_coord_grid(4, 4, device="cpu")
    pe_out = pe(coords)
    print(f"\n  PE output dim : {pe_out.shape[-1]}  (expected 40: 4×L for x,y sin+cos)")
    print(f"  PE range      : [{pe_out.min():.3f}, {pe_out.max():.3f}]  (expected [-1,1])")

    print(f"\n  ✓ INR (SIREN) ready!")
