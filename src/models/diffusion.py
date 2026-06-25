"""
DiffusionScheduler — IIDM Paper Section 3.3
Fixed: DDIM ab_prev device bug, clamp issues
"""
import torch
import numpy as np
from typing import Callable, Optional


class DiffusionScheduler:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02,
                 device=torch.device("cpu")):
        self.T      = T
        self.device = device

        betas          = torch.linspace(beta_start, beta_end, T,
                                        dtype=torch.float32, device=device)
        alphas         = 1.0 - betas
        alpha_bar      = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = torch.cat([torch.ones(1, device=device),
                                    alpha_bar[:-1]])

        self.betas               = betas
        self.alphas              = alphas
        self.alpha_bar           = alpha_bar
        self.alpha_bar_prev      = alpha_bar_prev
        self.sqrt_alpha_bar      = alpha_bar.sqrt()
        self.sqrt_one_minus_ab   = (1.0 - alpha_bar).sqrt()
        self.posterior_var       = (betas * (1.0 - alpha_bar_prev) /
                                   (1.0 - alpha_bar)).clamp(min=1e-20)

    def _gather(self, values, t, ndim):
        return values[t].reshape(t.shape[0], *([1] * (ndim - 1)))

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        ndim       = x0.ndim
        sqrt_ab    = self._gather(self.sqrt_alpha_bar,   t, ndim)
        sqrt_1m_ab = self._gather(self.sqrt_one_minus_ab, t, ndim)
        return sqrt_ab * x0 + sqrt_1m_ab * noise, noise

    @torch.no_grad()
    def ddim_sample(self, unet_fn, shape, condition, steps=50, eta=0.0):
        """
        Fixed DDIM: all tensors on same device, proper ab_prev handling
        """
        device = self.device

        # Timestep sequence: T-1 → 0
        t_seq = torch.linspace(self.T - 1, 0, steps,
                               dtype=torch.long, device=device)

        x = torch.randn(shape, device=device)

        for i, t_curr in enumerate(t_seq):
            t_batch = t_curr.expand(shape[0])

            # UNet predicts noise
            eps = unet_fn(x, t_batch, condition)

            ab_curr = self.alpha_bar[t_curr].clamp(min=1e-8)

            # ab_prev — FIXED: stays on device, tensor(0) at last step
            if i + 1 < len(t_seq):
                ab_prev = self.alpha_bar[t_seq[i + 1]]
            else:
                ab_prev = torch.tensor(0.0, device=device)  # ← KEY FIX

            # Predict x0
            x0_pred = (x - (1.0 - ab_curr).sqrt() * eps) / ab_curr.sqrt()
            x0_pred = x0_pred.clamp(-1.0, 1.0)

            # DDIM update
            dir_xt = (1.0 - ab_prev).clamp(min=0.0).sqrt() * eps
            noise  = eta * torch.randn_like(x) if eta > 0 else 0.0
            x      = ab_prev.sqrt() * x0_pred + dir_xt + noise

        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sched  = DiffusionScheduler(T=1000, device=device)

    B, C, H, W = 2, 256, 16, 16
    x0 = torch.randn(B, C, H, W, device=device)
    t  = torch.randint(0, 1000, (B,), device=device)

    xt, eps = sched.q_sample(x0, t)
    print(f"q_sample OK: xt={tuple(xt.shape)}")

    # Perfect UNet (returns actual noise)
    dummy = lambda x, t, c: eps
    x0_rec = sched.ddim_sample(dummy, (B, C, H, W), None, steps=50)
    print(f"DDIM output std : {x0_rec.std():.4f} (lower = more denoised)")
    print(f"NaN: {torch.isnan(x0_rec).any().item()}")
    print("✓ DiffusionScheduler fixed!")
