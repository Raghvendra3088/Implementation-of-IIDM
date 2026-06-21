"""
KD-UNet — Knowledge Distilled UNet Denoiser
============================================
Paper: IIDM Section 3.3

Conditioned denoiser that operates in VGG latent feature space.

Input:
    x_t       : noisy student features  (B, 480, H', W')
    t         : timestep                (B,)  → sinusoidal embedding
    condition : teacher VGG features    (B, 512+256+128+64=960, ...) — multi-scale

Output:
    eps_theta : predicted noise         (B, 480, H', W')

Architecture:
    Encoder : 4 DoubleConv blocks + MaxPool
              Each block receives concat(x_t_scale, teacher_scale) + time_emb
    Bottleneck: DoubleConv
    Decoder : 4 UpConv blocks + skip connections
    Output  : 1×1 Conv → same shape as x_t

Activations : SiLU (paper specifies, not ReLU)
Norm        : GroupNorm(8) — stable for small batch sizes
Time emb    : Sinusoidal → 2-layer MLP → added to each encoder/decoder block
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List


# ══════════════════════════════════════════════════════════════════════════════
# TIMESTEP EMBEDDING
# ══════════════════════════════════════════════════════════════════════════════

class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal timestep embedding — same as DDPM / Attention U-Net paper.

    t (scalar) → sinusoidal features (dim) → MLP → time_emb (out_dim)
    """

    def __init__(self, dim: int = 256, out_dim: int = 512):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t : (B,) integer timesteps
        Returns:
            emb : (B, out_dim)
        """
        half   = self.dim // 2
        freqs  = torch.exp(
            -np.log(10000) *
            torch.arange(half, device=t.device, dtype=torch.float32) / (half - 1)
        )
        args   = t[:, None].float() * freqs[None]          # (B, half)
        emb    = torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)
        return self.mlp(emb)                                 # (B, out_dim)


# ══════════════════════════════════════════════════════════════════════════════
# BUILDING BLOCKS
# ══════════════════════════════════════════════════════════════════════════════

class DoubleConv(nn.Module):
    """
    Conv → GroupNorm → SiLU → Conv → GroupNorm → SiLU
    With optional time embedding injection (additive after first norm).
    GroupNorm(8) chosen for stability with small batch sizes.
    """

    def __init__(self, in_ch: int, out_ch: int,
                 time_dim: int = 512,
                 num_groups: int = 8):
        super().__init__()

        # Adjust groups if out_ch not divisible
        groups = min(num_groups, out_ch)
        while out_ch % groups != 0:
            groups -= 1

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.act   = nn.SiLU()

        # Time embedding projection → out_ch (additive injection)
        self.time_proj = nn.Linear(time_dim, out_ch)

    def forward(self, x: torch.Tensor,
                time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x        : (B, in_ch, H, W)
            time_emb : (B, time_dim)
        Returns:
            out      : (B, out_ch, H, W)
        """
        h = self.act(self.norm1(self.conv1(x)))

        # Inject time embedding
        t = self.time_proj(time_emb)[:, :, None, None]   # (B, out_ch, 1, 1)
        h = h + t

        h = self.act(self.norm2(self.conv2(h)))
        return h


class DownBlock(nn.Module):
    """DoubleConv + MaxPool downsampling."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int = 512):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch, time_dim)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x, time_emb):
        feat = self.conv(x, time_emb)      # skip connection output
        down = self.pool(feat)
        return feat, down


class UpBlock(nn.Module):
    """Bilinear upsample + concat skip + DoubleConv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 time_dim: int = 512):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear",
                                align_corners=False)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch, time_dim)

    def forward(self, x, skip, time_emb):
        x = self.up(x)

        # Handle size mismatch (odd spatial dims)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, skip.shape[-2:],
                              mode="bilinear", align_corners=False)

        x = torch.cat([x, skip], dim=1)
        return self.conv(x, time_emb)


# ══════════════════════════════════════════════════════════════════════════════
# TEACHER FEATURE ADAPTER
# ══════════════════════════════════════════════════════════════════════════════

class TeacherConditionAdapter(nn.Module):
    """
    Adapt multi-scale teacher VGG features for use as UNet condition.

    Teacher channels: [64, 128, 256, 512]
    Project each scale → base UNet channel size for concat with x_t.
    """

    TEACHER_CHS = [64, 128, 256, 512]

    def __init__(self, unet_chs: List[int]):
        """
        Args:
            unet_chs : channel sizes at each UNet encoder scale
        """
        super().__init__()
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(t_ch, u_ch, 1, bias=False),
                nn.GroupNorm(min(8, u_ch), u_ch),
                nn.SiLU(),
            )
            for t_ch, u_ch in zip(self.TEACHER_CHS, unet_chs)
        ])

    def forward(self, teacher_feats: List[torch.Tensor],
                target_sizes: List[tuple]) -> List[torch.Tensor]:
        """
        Adapt and resize each teacher feature to match UNet encoder spatial size.

        Args:
            teacher_feats : list of 4 teacher feature maps
            target_sizes  : list of (H, W) for each UNet encoder level

        Returns:
            adapted : list of 4 adapted teacher tensors
        """
        adapted = []
        for i, (feat, adapter) in enumerate(
                zip(teacher_feats, self.adapters)):
            a = adapter(feat)
            if a.shape[-2:] != target_sizes[i]:
                a = F.interpolate(a, target_sizes[i],
                                  mode="bilinear", align_corners=False)
            adapted.append(a)
        return adapted


# ══════════════════════════════════════════════════════════════════════════════
# KD-UNet
# ══════════════════════════════════════════════════════════════════════════════

class KDUNet(nn.Module):
    """
    Knowledge Distilled UNet Denoiser.
    Paper IIDM Section 3.3.

    Predicts noise eps_theta(x_t, t, teacher_feats) in latent space.

    x_t shape: (B, 480, H', W')  — student VGG features, concatenated across scales
    Note: H', W' = H/16, W/16  (after 4 VGG pool layers on 256×256 input → 16×16)

    UNet base channels: 64 (expands to 128, 256, 512 through encoder)
    """

    # Latent input channels = sum of student VGG block channels
    # Student: [32, 64, 128, 256] but we use the LAST block only as latent
    # Paper: use f4 (deepest) as the diffusion target latent
    IN_CH   = 256    # student block4 channels (deepest, most semantic)
    BASE_CH = 64     # UNet base channels

    def __init__(self, time_dim: int = 512):
        super().__init__()

        C = self.BASE_CH
        self.time_emb = SinusoidalTimeEmbedding(dim=256, out_dim=time_dim)

        # Teacher condition adapter (for block4 only — deepest semantic level)
        # Teacher block4 = 512ch → project to C
        self.teacher_adapter = nn.Sequential(
            nn.Conv2d(512, C, 1, bias=False),
            nn.GroupNorm(8, C),
            nn.SiLU(),
        )

        # ── Encoder ──────────────────────────────────────────────────────────
        # Input: concat(x_t, teacher_adapted) = IN_CH + C channels
        self.enc1 = DownBlock(self.IN_CH + C, C,    time_dim)   # → C,    H/2
        self.enc2 = DownBlock(C,              C*2,  time_dim)   # → C*2,  H/4
        self.enc3 = DownBlock(C*2,            C*4,  time_dim)   # → C*4,  H/8

        # ── Bottleneck ───────────────────────────────────────────────────────
        self.bot  = DoubleConv(C*4, C*8, time_dim)              # → C*8

        # ── Decoder ──────────────────────────────────────────────────────────
        self.dec3 = UpBlock(C*8, C*4, C*4, time_dim)
        self.dec2 = UpBlock(C*4, C*2, C*2, time_dim)
        self.dec1 = UpBlock(C*2, C,   C,   time_dim)

        # ── Output ───────────────────────────────────────────────────────────
        self.out  = nn.Conv2d(C, self.IN_CH, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self,
                x_t:           torch.Tensor,
                t:             torch.Tensor,
                teacher_feats: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x_t           : (B, 256, H', W') noisy student latent (block4)
            t             : (B,) integer timesteps
            teacher_feats : list of 4 teacher feature maps

        Returns:
            eps_theta     : (B, 256, H', W') predicted noise
        """
        # ── Time embedding ────────────────────────────────────────────────────
        t_emb = self.time_emb(t)                              # (B, 512)

        # ── Adapt teacher condition ───────────────────────────────────────────
        # Use deepest teacher feature (block4, 512ch) as global condition
        t_feat = self.teacher_adapter(teacher_feats[3])       # (B, C, H', W')
        if t_feat.shape[-2:] != x_t.shape[-2:]:
            t_feat = F.interpolate(t_feat, x_t.shape[-2:],
                                   mode="bilinear", align_corners=False)

        # Concat x_t + teacher condition
        x = torch.cat([x_t, t_feat], dim=1)                  # (B, IN_CH+C, H', W')

        # ── Encoder ──────────────────────────────────────────────────────────
        skip1, x = self.enc1(x, t_emb)
        skip2, x = self.enc2(x, t_emb)
        skip3, x = self.enc3(x, t_emb)

        # ── Bottleneck ────────────────────────────────────────────────────────
        x = self.bot(x, t_emb)

        # ── Decoder ──────────────────────────────────────────────────────────
        x = self.dec3(x, skip3, t_emb)
        x = self.dec2(x, skip2, t_emb)
        x = self.dec1(x, skip1, t_emb)

        # ── Output ────────────────────────────────────────────────────────────
        return self.out(x)                                     # (B, 256, H', W')


# ══════════════════════════════════════════════════════════════════════════════
# SANITY CHECK
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
    from src.models.kd_vgg import TeacherVGG, StudentVGG

    print("=" * 50)
    print("  KD-UNet Sanity Check")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device : {device}")

    B, H, W = 2, 256, 256
    dummy   = torch.randn(B, 6, H, W, device=device)

    # Get teacher + student features
    teacher = TeacherVGG(in_channels=6).to(device)
    student = StudentVGG(in_channels=6).to(device)

    with torch.no_grad():
        t_feats = teacher(dummy)
        s_feats = student(dummy)

    # Use student block4 as latent x0
    x0     = s_feats[3]                      # (B, 256, 16, 16)
    print(f"\n  x0 (student block4) : {tuple(x0.shape)}")
    print(f"  Teacher block4      : {tuple(t_feats[3].shape)}")

    # Diffusion: add noise at t=500
    from src.models.diffusion import DiffusionScheduler
    sched  = DiffusionScheduler(T=1000, device=device)
    t_step = torch.full((B,), 500, dtype=torch.long, device=device)
    x_t, noise = sched.q_sample(x0, t_step)
    print(f"\n  x_t (noisy latent)  : {tuple(x_t.shape)}")

    # KD-UNet forward
    unet = KDUNet(time_dim=512).to(device)
    with torch.no_grad():
        eps_pred = unet(x_t, t_step, t_feats)

    print(f"\n  Predicted noise shape : {tuple(eps_pred.shape)}")
    print(f"  Expected              : {tuple(noise.shape)}")
    print(f"  Shape match           : {eps_pred.shape == noise.shape}")
    print(f"  NaN in output         : {torch.isnan(eps_pred).any().item()}")
    print(f"  Inf in output         : {torch.isinf(eps_pred).any().item()}")

    # Loss
    loss = torch.nn.functional.mse_loss(eps_pred, noise)
    print(f"\n  Diffusion loss (random): {loss.item():.4f}  (expected ~1.0 for random init)")

    # Param count
    unet_params = sum(p.numel() for p in unet.parameters()
                      if p.requires_grad) / 1e6
    print(f"\n  KD-UNet params : {unet_params:.2f}M")

    # DDIM full inference test
    def unet_fn(x, t, cond):
        return unet(x, t, cond)

    x0_sampled = sched.ddim_sample(
        unet_fn,
        shape=(B, KDUNet.IN_CH, x0.shape[2], x0.shape[3]),
        condition=t_feats,
        steps=10
    )
    print(f"\n  DDIM 10-step sample shape : {tuple(x0_sampled.shape)}")
    print(f"  NaN in sample             : {torch.isnan(x0_sampled).any().item()}")

    print(f"\n  ✓ KD-UNet ready!")
