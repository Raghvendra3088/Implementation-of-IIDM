"""
Biophysical Monotonicity Loss for Physics-Informed Latent Diffusion (PI-IIDM).

Scientific Principle:
Forest Aboveground Biomass Density (AGBD) is biophysically constrained by canopy
structure: for a given stand, aboveground carbon must monotonically correlate with
canopy height (dY/dH >= 0). We penalize violations via:
    L_mono = mean(ReLU(-dY/dH))

This formulation avoids uncalibrated, rigid empirical allometric power-law constants
(e.g., a=0.112, b=2.59), which vary widely across biomes and species compositions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def stochastic_point_physics_loss(x_batch, pred_batch, num_points=250, chm_idx=7):
    """
    Computes strict point-wise physical loss L_mono = mean(ReLU(-dY/dH))
    Evaluates on a random spatial subset of valid pixels to prevent OOM crashes.
    """
    B, _, H, W = x_batch.shape
    if num_points > H * W:
        num_points = H * W

    batch_idx = torch.randint(0, B, (num_points,), device=x_batch.device)
    y_idx = torch.randint(0, H, (num_points,), device=x_batch.device)
    x_idx = torch.randint(0, W, (num_points,), device=x_batch.device)

    selected_y = pred_batch[batch_idx, 0, y_idx, x_idx]

    dY_dX = torch.autograd.grad(
        outputs=selected_y.sum(),
        inputs=x_batch,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    dY_dH = dY_dX[batch_idx, chm_idx, y_idx, x_idx]
    return F.relu(-dY_dH).mean()


class MonotonicityPhysicsLoss(nn.Module):
    """
    Monotonicity constraint module enforcing non-negative carbon-canopy height gradient.
    """
    def __init__(self, lambda_phys=0.1, chm_idx=7, num_points=250):
        super().__init__()
        self.lambda_phys = lambda_phys
        self.chm_idx = chm_idx
        self.num_points = num_points

    def forward(self, x_cond, y_pred):
        if self.lambda_phys <= 0:
            return torch.tensor(0.0, device=y_pred.device)
        l_mono = stochastic_point_physics_loss(
            x_cond, y_pred, num_points=self.num_points, chm_idx=self.chm_idx
        )
        return self.lambda_phys * l_mono


# Backward-compatible alias for existing imports
PhysicsInformedLoss = MonotonicityPhysicsLoss
