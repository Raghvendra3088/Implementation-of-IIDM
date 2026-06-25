"""
IIDM — Paper-correct implementation
Student block4 = 256ch throughout. Zero padding removed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.models.kd_vgg    import TeacherVGG, StudentVGG, KDLoss
from src.models.inr       import SIRENINR, make_coord_grid
from src.models.diffusion import DiffusionScheduler
from src.models.kd_unet   import KDUNet


class IIDMLoss(nn.Module):
    def __init__(self, lambda_kd=0.1, lambda_recon=1.0):
        super().__init__()
        self.lambda_kd    = lambda_kd
        self.lambda_recon = lambda_recon
        self.kd_loss_fn   = KDLoss()
        self.mse          = nn.MSELoss()
        self.mae          = nn.L1Loss()

    def forward(self, eps_theta, eps_target,
                student_feats, teacher_feats,
                carbon_pred, carbon_gt, lambda_diff=1.0):
        L_diff  = self.mse(eps_theta, eps_target)
        L_kd    = self.kd_loss_fn(student_feats, teacher_feats)
        L_recon = self.mae(carbon_pred, carbon_gt)
        L_total = lambda_diff * L_diff + self.lambda_kd * L_kd + self.lambda_recon * L_recon
        return L_total, {
            "L_total": L_total.item(), "L_diff": L_diff.item(),
            "L_kd":    L_kd.item(),   "L_recon": L_recon.item(),
        }


class TeacherCondenser(nn.Module):
    """
    Teacher [64,128,256,512] → each projected to 64ch → concat = 256ch
    Matches student block4 = 256ch  ← paper-correct, no padding
    """
    TEACHER_CHS = [64, 128, 256, 512]
    OUT_CH_EACH = 64   # 64 × 4 = 256

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

    def forward(self, teacher_feats, target_hw):
        projected = []
        for feat, proj in zip(teacher_feats, self.projs):
            p = proj(feat)
            if p.shape[-2:] != target_hw:
                p = F.adaptive_avg_pool2d(p, target_hw)
            projected.append(p)
        return torch.cat(projected, dim=1)   # (B, 256, H', W')


class IIDM(nn.Module):
    def __init__(self, in_channels=6, T=1000,
                 lambda_kd=0.1, lambda_recon=1.0,
                 device=torch.device("cpu")):
        super().__init__()
        self.T      = T
        self.device = device

        self.teacher_vgg  = TeacherVGG(in_channels=in_channels)
        self.student_vgg  = StudentVGG(in_channels=in_channels)
        self.teacher_cond = TeacherCondenser()
        self.unet         = KDUNet(in_channels=256, out_channels=256)
        self.inr          = SIRENINR(student_chs=[32, 64, 128, 256])
        self.scheduler    = DiffusionScheduler(T=T, device=device)
        self.loss_fn      = IIDMLoss(lambda_kd, lambda_recon)

        for p in self.teacher_vgg.parameters():
            p.requires_grad = False

    def trainable_parameters(self):
        return (list(self.student_vgg.parameters())  +
                list(self.teacher_cond.parameters()) +
                list(self.unet.parameters())         +
                list(self.inr.parameters())          +
                list(self.loss_fn.kd_loss_fn.parameters()))

    def forward(self, x, carbon_gt, lambda_diff=1.0):
        B, _, H, W = x.shape

        with torch.no_grad():
            teacher_feats = self.teacher_vgg(x)
        student_feats = self.student_vgg(x)

        latent = student_feats[3]          # (B, 256, H/16, W/16)
        lat_hw = latent.shape[-2:]

        # Normalize latent to N(0,1) — CRITICAL FIX
        # Raw latent mean=1.03 caused L_diff to be stuck at 0.55
        lat_mean = latent.mean(dim=[1,2,3], keepdim=True)
        lat_std  = latent.std(dim=[1,2,3],  keepdim=True).clamp(min=1e-6)
        latent_norm = (latent - lat_mean) / lat_std

        t        = torch.randint(0, self.T, (B,), device=x.device, dtype=torch.long)
        x_t, eps = self.scheduler.q_sample(latent_norm, t)

        # Both 256ch — clean concat inside unet → 512ch
        cond      = self.teacher_cond(teacher_feats, lat_hw)  # (B, 256, H', W')
        eps_theta = self.unet(x_t, t, cond)                   # (B, 256, H', W')

        carbon_pred = self.inr(student_feats, H=H, W=W)

        loss, components = self.loss_fn(
            eps_theta, eps,
            student_feats, teacher_feats,
            carbon_pred, carbon_gt,
            lambda_diff=lambda_diff,
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

        # Same normalization as training
        lat_mean = latent.mean(dim=[1,2,3], keepdim=True)
        lat_std  = latent.std(dim=[1,2,3],  keepdim=True).clamp(min=1e-6)

        x_0_norm = self.scheduler.ddim_sample(
            self.unet,
            shape=(B, 256, *lat_hw),
            condition=cond,
            steps=steps,
        )

        # Denormalize back to original latent space
        x_0 = x_0_norm * lat_std + lat_mean

        refined    = list(student_feats)
        refined[3] = x_0.clamp(-1, 1)
        return self.inr(refined, H=H, W=W)

    @staticmethod
    def denormalize(carbon_norm, vmin, vmax):
        return (carbon_norm + 1.0) / 2.0 * (vmax - vmin) + vmin


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    B, H, W   = 2, 256, 256
    x         = torch.randn(B, 6, H, W, device=device)
    carbon_gt = torch.randn(B, 1, H, W, device=device).clamp(-1, 1)

    model = IIDM(in_channels=6, T=1000, device=device).to(device)
    model.train()

    loss, comp = model(x, carbon_gt)
    print(f"L_total={comp['L_total']:.4f}  L_diff={comp['L_diff']:.4f}  "
          f"L_kd={comp['L_kd']:.4f}  L_recon={comp['L_recon']:.4f}")
    loss.backward()
    print("Backward ✓")

    pred = model.predict(x, steps=10)
    print(f"Predict shape: {tuple(pred.shape)}")
    print(f"✓ IIDM 256ch — paper correct!")
