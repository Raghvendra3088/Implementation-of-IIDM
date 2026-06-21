"""
IIDM — Improved Implicit Diffusion Model (Full Pipeline)
=========================================================
Paper: All sections combined

Combines:
    1. TeacherVGG  — frozen, provides distillation target + UNet condition
    2. StudentVGG  — trainable, produces latent x0 for diffusion
    3. DiffusionScheduler — forward/reverse in latent space
    4. KDUNet      — denoiser in latent space, conditioned on teacher
    5. SIRENINR    — decodes denoised latent → carbon map

Training forward:
    x (B,6,H,W) → TeacherVGG → t_feats [f1..f4]  (frozen)
               → StudentVGG  → s_feats [s1..s4]  (trainable)
    t ~ Uniform(0, T)
    x_t, eps = scheduler.q_sample(s_feats[3], t)   # noise latent block4
    eps_theta = KDUNet(x_t, t, t_feats)             # predict noise
    carbon    = SIRENINR(s_feats, coords)            # decode carbon map

    L_diff  = MSE(eps_theta, eps)
    L_kd    = (1/4) Σ MSE(s_i, t_i.detach())
    L_recon = MAE(carbon, carbon_gt)
    L_total = L_diff + 0.1*L_kd + 1.0*L_recon

Inference:
    s_feats = StudentVGG(x)
    x_T     = randn_like(s_feats[3])
    x_0     = DDIM_sample(KDUNet, x_T, condition=t_feats, steps=50)
    carbon  = SIRENINR(with x_0 as block4, coords)
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
# LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

class IIDMLoss(nn.Module):
    """
    Total IIDM loss — Paper Eq. (5)

    L_total = L_diff + lambda_kd * L_kd + lambda_recon * L_recon

    L_diff  = MSE(eps_theta, eps)          diffusion noise prediction
    L_kd    = (1/4) Σ MSE(s_i, t_i)       knowledge distillation
    L_recon = MAE(carbon_pred, carbon_gt)  reconstruction (L1 for robustness)

    Paper values: lambda_kd=0.1, lambda_recon=1.0
    """

    def __init__(self, lambda_kd: float = 0.1,
                       lambda_recon: float = 1.0):
        super().__init__()
        self.lambda_kd    = lambda_kd
        self.lambda_recon = lambda_recon
        self.kd_loss_fn   = KDLoss()
        self.mse          = nn.MSELoss()
        self.mae          = nn.L1Loss()

    def forward(self,
                eps_theta:    torch.Tensor,
                eps_target:   torch.Tensor,
                student_feats: List[torch.Tensor],
                teacher_feats: List[torch.Tensor],
                carbon_pred:  torch.Tensor,
                carbon_gt:    torch.Tensor
                ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Returns:
            total_loss : scalar
            components : dict with individual loss values for logging
        """
        L_diff  = self.mse(eps_theta, eps_target)
        L_kd    = self.kd_loss_fn(student_feats, teacher_feats)
        L_recon = self.mae(carbon_pred, carbon_gt)

        L_total = (L_diff
                   + self.lambda_kd    * L_kd
                   + self.lambda_recon * L_recon)

        components = {
            "L_total" : L_total.item(),
            "L_diff"  : L_diff.item(),
            "L_kd"    : L_kd.item(),
            "L_recon" : L_recon.item(),
        }
        return L_total, components


# ══════════════════════════════════════════════════════════════════════════════
# FULL IIDM MODEL
# ══════════════════════════════════════════════════════════════════════════════

class IIDM(nn.Module):
    """
    Full Improved Implicit Diffusion Model.
    Paper: all sections.

    Components:
        teacher_vgg : TeacherVGG (frozen)
        student_vgg : StudentVGG (trainable)
        unet        : KDUNet    (trainable)
        inr         : SIRENINR  (trainable)
        scheduler   : DiffusionScheduler (no params)
        loss_fn     : IIDMLoss
    """

    def __init__(self,
                 in_channels:   int   = 6,
                 T:             int   = 1000,
                 lambda_kd:     float = 0.1,
                 lambda_recon:  float = 1.0,
                 device: torch.device = torch.device("cpu")):
        super().__init__()

        self.T      = T
        self.device = device

        # ── Components ────────────────────────────────────────────────────────
        self.teacher_vgg = TeacherVGG(in_channels=in_channels)
        self.student_vgg = StudentVGG(in_channels=in_channels)
        self.unet        = KDUNet(time_dim=512)
        self.inr         = SIRENINR(student_chs=[32, 64, 128, 256])
        self.scheduler   = DiffusionScheduler(T=T, device=device)
        self.loss_fn     = IIDMLoss(lambda_kd, lambda_recon)

        # Teacher always frozen
        for p in self.teacher_vgg.parameters():
            p.requires_grad = False

    def trainable_parameters(self):
        """Return only trainable parameters (exclude teacher)."""
        return (list(self.student_vgg.parameters()) +
                list(self.unet.parameters())        +
                list(self.inr.parameters())         +
                list(self.loss_fn.kd_loss_fn.parameters()))

    # ── TRAINING FORWARD ──────────────────────────────────────────────────────

    def forward(self,
                x:          torch.Tensor,
                carbon_gt:  torch.Tensor
                ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Full training forward pass.

        Args:
            x         : (B, 6, H, W)  input satellite stack, range [-1,1]
            carbon_gt : (B, 1, H, W)  ground truth carbon map, range [-1,1]

        Returns:
            loss       : scalar total loss
            components : dict of individual loss values
        """
        B, _, H, W = x.shape

        # ── 1. Extract features ───────────────────────────────────────────────
        with torch.no_grad():
            teacher_feats = self.teacher_vgg(x)        # frozen

        student_feats = self.student_vgg(x)             # trainable

        # ── 2. Diffusion in latent space (student block4) ─────────────────────
        latent = student_feats[3]                       # (B, 256, H/16, W/16)

        t      = torch.randint(0, self.T, (B,),
                               device=x.device, dtype=torch.long)

        x_t, eps = self.scheduler.q_sample(latent, t)  # add noise

        # ── 3. Predict noise with KD-UNet ─────────────────────────────────────
        eps_theta = self.unet(x_t, t, teacher_feats)   # (B, 256, H/16, W/16)

        # ── 4. Decode carbon map with INR ─────────────────────────────────────
        carbon_pred = self.inr(student_feats, H=H, W=W)  # (B, 1, H, W)

        # ── 5. Compute total loss ─────────────────────────────────────────────
        loss, components = self.loss_fn(
            eps_theta    = eps_theta,
            eps_target   = eps,
            student_feats = student_feats,
            teacher_feats = teacher_feats,
            carbon_pred  = carbon_pred,
            carbon_gt    = carbon_gt,
        )

        return loss, components

    # ── INFERENCE ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self,
                x:      torch.Tensor,
                steps:  int = 50) -> torch.Tensor:
        """
        Full inference pipeline.

        Args:
            x     : (B, 6, H, W)  input satellite stack, range [-1,1]
            steps : DDIM steps (paper default: 50)

        Returns:
            carbon_map : (B, 1, H, W) predicted carbon, range [-1,1]
        """
        B, _, H, W = x.shape
        self.eval()

        # ── 1. Extract features ───────────────────────────────────────────────
        teacher_feats = self.teacher_vgg(x)
        student_feats = self.student_vgg(x)

        latent_shape  = student_feats[3].shape    # (B, 256, H/16, W/16)

        # ── 2. DDIM reverse diffusion in latent space ─────────────────────────
        def unet_fn(x_t, t_batch, condition):
            return self.unet(x_t, t_batch, condition)

        x_0 = self.scheduler.ddim_sample(
            unet_fn   = unet_fn,
            shape     = latent_shape,
            condition = teacher_feats,
            steps     = steps,
        )                                         # (B, 256, H/16, W/16) denoised

        # ── 3. Replace student block4 with denoised latent ────────────────────
        # Use denoised x_0 as the refined block4 for INR decoding
        refined_feats      = list(student_feats)
        refined_feats[3]   = x_0

        # ── 4. INR decode → carbon map ────────────────────────────────────────
        carbon_map = self.inr(refined_feats, H=H, W=W)  # (B, 1, H, W)

        return carbon_map

    # ── DENORMALIZE ───────────────────────────────────────────────────────────

    @staticmethod
    def denormalize(carbon_norm: torch.Tensor,
                    vmin: float, vmax: float) -> torch.Tensor:
        """
        Convert normalized [-1,1] carbon back to Mg C/ha.

        norm = (raw - vmin) / (vmax - vmin) * 2 - 1
        raw  = (norm + 1) / 2 * (vmax - vmin) + vmin
        """
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

    B, H, W = 2, 256, 256
    x          = torch.randn(B, 6, H, W, device=device)
    carbon_gt  = torch.randn(B, 1, H, W, device=device).clamp(-1, 1)

    # Init model
    model = IIDM(in_channels=6, T=1000, device=device).to(device)

    # ── Training forward ──────────────────────────────────────────────────────
    print("\n  [1/3] Training forward pass ...")
    model.train()
    loss, components = model(x, carbon_gt)

    print(f"  L_total : {components['L_total']:.4f}")
    print(f"  L_diff  : {components['L_diff']:.4f}")
    print(f"  L_kd    : {components['L_kd']:.4f}")
    print(f"  L_recon : {components['L_recon']:.4f}")
    print(f"  NaN     : {torch.isnan(loss).item()}")
    print(f"  Inf     : {torch.isinf(loss).item()}")

    # ── Backward ──────────────────────────────────────────────────────────────
    print("\n  [2/3] Backward pass ...")
    loss.backward()
    print(f"  Gradients OK ✓")

    # ── Inference ─────────────────────────────────────────────────────────────
    print("\n  [3/3] Inference (DDIM 10 steps) ...")
    with torch.no_grad():
        carbon_pred = model.predict(x, steps=10)

    print(f"  Prediction shape : {tuple(carbon_pred.shape)}")
    print(f"  Prediction range : [{carbon_pred.min():.4f}, {carbon_pred.max():.4f}]")
    print(f"  NaN              : {torch.isnan(carbon_pred).any().item()}")

    # Denormalize
    carbon_mgcha = IIDM.denormalize(carbon_pred, vmin=0.04, vmax=207.97)
    print(f"  Denorm range (Mg C/ha): [{carbon_mgcha.min():.2f}, {carbon_mgcha.max():.2f}]")

    # ── Parameter summary ─────────────────────────────────────────────────────
    print("\n  Parameter Summary:")
    def count(m): return sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6
    print(f"    Teacher VGG (frozen) : {sum(p.numel() for p in model.teacher_vgg.parameters())/1e6:.2f}M")
    print(f"    Student VGG          : {count(model.student_vgg):.2f}M")
    print(f"    KD-UNet              : {count(model.unet):.2f}M")
    print(f"    SIREN INR            : {count(model.inr):.2f}M")
    total_trainable = count(model.student_vgg) + count(model.unet) + count(model.inr)
    print(f"    Total trainable      : {total_trainable:.2f}M")

    print(f"\n  ✓ IIDM Full Pipeline ready!")
