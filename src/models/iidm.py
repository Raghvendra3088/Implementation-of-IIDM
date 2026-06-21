"""
IIDM — Improved Implicit Diffusion Model (Full Pipeline)
=========================================================
Fixed to match existing kd_unet.py interface:
    kd_unet.forward(xt, t, cond)
    cond = tensor (B, 480, H', W')  — teacher feats pooled + concat
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.models.kd_vgg    import TeacherVGG, StudentVGG, KDLoss
from src.models.inr       import SIRENINR, make_coord_grid
from src.models.diffusion import DiffusionScheduler
from src.models.kd_unet   import KDUNet


# ══════════════════════════════════════════════════════════════════════════════
# LOSS
# ══════════════════════════════════════════════════════════════════════════════

class IIDMLoss(nn.Module):
    """
    L_total = L_diff + 0.1*L_kd + 1.0*L_recon
    L_diff  = MSE(eps_theta, eps)
    L_kd    = (1/4) sum MSE(s_i, t_i.detach())
    L_recon = MAE(carbon_pred, carbon_gt)
    """
    def __init__(self, lambda_kd=0.1, lambda_recon=1.0):
        super().__init__()
        self.lambda_kd    = lambda_kd
        self.lambda_recon = lambda_recon
        self.kd_loss_fn   = KDLoss()
        self.mse          = nn.MSELoss()
        self.mae          = nn.L1Loss()

    def forward(self, eps_theta, eps_target,
                student_feats, teacher_feats,
                carbon_pred, carbon_gt):
        L_diff  = self.mse(eps_theta, eps_target)
        L_kd    = self.kd_loss_fn(student_feats, teacher_feats)
        L_recon = self.mae(carbon_pred, carbon_gt)
        L_total = L_diff + self.lambda_kd * L_kd + self.lambda_recon * L_recon
        return L_total, {
            "L_total": L_total.item(),
            "L_diff" : L_diff.item(),
            "L_kd"   : L_kd.item(),
            "L_recon": L_recon.item(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# TEACHER FEATURE → CONDITION TENSOR
# ══════════════════════════════════════════════════════════════════════════════

class TeacherCondenser(nn.Module):
    """
    Condense 4 teacher feature maps into single tensor
    matching kd_unet.py cond shape: (B, 480, H', W')

    Teacher channels: [64, 128, 256, 512] → project each to 120 → concat = 480
    Resize all to latent spatial size H'xW'
    """
    TEACHER_CHS = [64, 128, 256, 512]
    OUT_CH_EACH = 120   # 120 × 4 = 480 = student latent channels

    def __init__(self):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(t_ch, self.OUT_CH_EACH, 1, bias=False),
                nn.GroupNorm(8, self.OUT_CH_EACH),
                nn.SiLU(),
            )
            for t_ch in self.TEACHER_CHS
        ])

    def forward(self, teacher_feats: List[torch.Tensor],
                target_hw: tuple) -> torch.Tensor:
        """
        Args:
            teacher_feats : list of 4 teacher tensors
            target_hw     : (H', W') — latent spatial size
        Returns:
            cond : (B, 480, H', W')
        """
        projected = []
        for feat, proj in zip(teacher_feats, self.projs):
            p = proj(feat)
            if p.shape[-2:] != target_hw:
                p = F.adaptive_avg_pool2d(p, target_hw)
            projected.append(p)
        return torch.cat(projected, dim=1)   # (B, 480, H', W')


# ══════════════════════════════════════════════════════════════════════════════
# FULL IIDM
# ══════════════════════════════════════════════════════════════════════════════

class IIDM(nn.Module):
    def __init__(self, in_channels=6, T=1000,
                 lambda_kd=0.1, lambda_recon=1.0,
                 device=torch.device("cpu")):
        super().__init__()
        self.T      = T
        self.device = device

        self.teacher_vgg   = TeacherVGG(in_channels=in_channels)
        self.student_vgg   = StudentVGG(in_channels=in_channels)
        self.teacher_cond  = TeacherCondenser()
        self.unet          = KDUNet()
        self.inr           = SIRENINR(student_chs=[32, 64, 128, 256])
        self.scheduler     = DiffusionScheduler(T=T, device=device)
        self.loss_fn       = IIDMLoss(lambda_kd, lambda_recon)

        # Teacher always frozen
        for p in self.teacher_vgg.parameters():
            p.requires_grad = False

    def trainable_parameters(self):
        return (list(self.student_vgg.parameters())      +
                list(self.teacher_cond.parameters())     +
                list(self.unet.parameters())             +
                list(self.inr.parameters())              +
                list(self.loss_fn.kd_loss_fn.parameters()))

    def forward(self, x, carbon_gt):
        B, _, H, W = x.shape

        # 1. Features
        with torch.no_grad():
            teacher_feats = self.teacher_vgg(x)
        student_feats = self.student_vgg(x)

        # 2. Latent = student block4 (B, 256, H/16, W/16)
        latent = student_feats[3]
        lat_hw = latent.shape[-2:]

        # 3. Diffusion forward
        t          = torch.randint(0, self.T, (B,), device=x.device, dtype=torch.long)
        x_t, eps   = self.scheduler.q_sample(latent, t)

        # 4. Teacher condition tensor (B, 480, H/16, W/16)
        # kd_unet expects 480ch cond — pad latent (256) with zeros to match
        # OR use teacher condenser → 480ch
        cond = self.teacher_cond(teacher_feats, lat_hw)   # (B, 480, H', W')

        # kd_unet concat(xt, cond) internally → needs xt also 480ch
        # Pad x_t from 256 → 480 with zeros
        pad   = torch.zeros(B, 480 - 256, *lat_hw, device=x.device)
        x_t_480 = torch.cat([x_t, pad], dim=1)            # (B, 480, H', W')

        eps_theta_480 = self.unet(x_t_480, t, cond)       # (B, 480, H', W')
        eps_theta     = eps_theta_480[:, :256]             # take first 256ch

        # 5. INR decode
        carbon_pred = self.inr(student_feats, H=H, W=W)

        # 6. Loss
        loss, components = self.loss_fn(
            eps_theta, eps,
            student_feats, teacher_feats,
            carbon_pred, carbon_gt,
        )
        return loss, components

    @torch.no_grad()
    def predict(self, x, steps=50):
        B, _, H, W = x.shape
        self.eval()

        teacher_feats = self.teacher_vgg(x)
        student_feats = self.student_vgg(x)
        latent        = student_feats[3]
        lat_hw        = latent.shape[-2:]
        cond          = self.teacher_cond(teacher_feats, lat_hw)

        # DDIM: start from 480ch noise, unet outputs 480ch
        def unet_fn(x_t, t_batch, condition):
            return self.unet(x_t, t_batch, condition)

        x_0_480 = self.scheduler.ddim_sample(
            unet_fn,
            shape=(B, 480, *lat_hw),
            condition=cond,
            steps=steps,
        )
        x_0 = x_0_480[:, :256]   # take meaningful 256ch

        refined        = list(student_feats)
        refined[3]     = x_0
        return self.inr(refined, H=H, W=W)

    @staticmethod
    def denormalize(carbon_norm, vmin, vmax):
        return (carbon_norm + 1.0) / 2.0 * (vmax - vmin) + vmin


# ══════════════════════════════════════════════════════════════════════════════
# SANITY CHECK
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 55)
    print("  IIDM Full Pipeline Sanity Check")
    print("=" * 55)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device : {device}")

    B, H, W   = 2, 256, 256
    x         = torch.randn(B, 6, H, W, device=device)
    carbon_gt = torch.randn(B, 1, H, W, device=device).clamp(-1, 1)

    model = IIDM(in_channels=6, T=1000, device=device).to(device)

    print("\n  [1/3] Training forward pass ...")
    model.train()
    loss, components = model(x, carbon_gt)
    print(f"  L_total : {components['L_total']:.4f}")
    print(f"  L_diff  : {components['L_diff']:.4f}")
    print(f"  L_kd    : {components['L_kd']:.4f}")
    print(f"  L_recon : {components['L_recon']:.4f}")
    print(f"  NaN     : {torch.isnan(loss).item()}")
    print(f"  Inf     : {torch.isinf(loss).item()}")

    print("\n  [2/3] Backward pass ...")
    loss.backward()
    print(f"  Gradients OK ✓")

    print("\n  [3/3] Inference (DDIM 10 steps) ...")
    carbon_pred = model.predict(x, steps=10)
    print(f"  Prediction shape : {tuple(carbon_pred.shape)}")
    print(f"  Prediction range : [{carbon_pred.min():.4f}, {carbon_pred.max():.4f}]")
    print(f"  NaN              : {torch.isnan(carbon_pred).any().item()}")

    carbon_real = IIDM.denormalize(carbon_pred, vmin=0.04, vmax=207.97)
    print(f"  Denorm (Mg C/ha) : [{carbon_real.min():.2f}, {carbon_real.max():.2f}]")

    print("\n  Parameter Summary:")
    t_params = sum(p.numel() for p in model.trainable_parameters()) / 1e6
    print(f"  Total trainable  : {t_params:.2f}M")
    print(f"\n  ✓ IIDM Full Pipeline ready!")
