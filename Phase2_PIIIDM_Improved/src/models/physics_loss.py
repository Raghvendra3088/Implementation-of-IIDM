import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsInformedLoss(nn.Module):
    def __init__(self, lambda_phys=0.1, c_scale=1e-3):
        super().__init__()
        self.lambda_phys = lambda_phys
        self.c_scale = c_scale
        self.a, self.b = 0.112, 2.59

    def allometric_residual(self, carbon_pred, canopy_h, C_MIN=0.04, C_MAX=207.97):
        carbon_real = (carbon_pred + 1)/2 * (C_MAX - C_MIN) + C_MIN
        canopy_real = canopy_h * 24.4
        agb_expected = self.a * (canopy_real.clamp(min=0.1) ** self.b)
        return F.mse_loss(carbon_real, agb_expected.clamp(0, C_MAX))

    def smoothness_residual(self, carbon_pred):
        gx = (carbon_pred[:,:,:,1:] - carbon_pred[:,:,:,:-1]).abs()
        gy = (carbon_pred[:,:,1:,:] - carbon_pred[:,:,:-1,:]).abs()
        return gx.mean() + gy.mean()

    def forward(self, carbon_pred, canopy_h, sigma_t=1.0):
        R1 = self.allometric_residual(carbon_pred, canopy_h)
        R3 = self.smoothness_residual(carbon_pred)
        R_total = R1 + 0.1*R3
        # Clamp sigma_bar to avoid near-zero division (physics loss explosion)
        sigma_bar = float(sigma_t) / self.c_scale
        sigma_bar = max(0.1, min(10.0, sigma_bar))   # clamp as float
        phys_weight = min(1.0, self.lambda_phys / (2*sigma_bar + 1e-8))
        return phys_weight * R_total
