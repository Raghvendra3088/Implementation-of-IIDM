"""
Diffusion Scheduler — IIDM Paper Section 3.3
=============================================
Key distinction from standard diffusion:
    Diffusion operates in VGG LATENT SPACE, not pixel space.
    x0 = student VGG features (flattened), NOT carbon pixels.

Forward process q(x_t | x_0):
    x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps
    eps ~ N(0, I)

Reverse process p_theta(x_{t-1} | x_t):
    KD-UNet predicts eps_theta(x_t, t, teacher_feats)
    x0_pred = (x_t - sqrt(1-ab_t)*eps_theta) / sqrt(ab_t)

Inference: DDIM (50 steps) — deterministic, faster than DDPM

Schedule:
    beta: linear, 1e-4 → 0.02, T=1000  (paper default)
    alpha_t = 1 - beta_t
    alpha_bar_t = prod(alpha_0 ... alpha_t)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Callable, Optional


class DiffusionScheduler:
    """
    DDPM/DDIM scheduler operating in VGG latent feature space.
    Paper IIDM Section 3.3.

    All precomputed tensors registered as buffers on a given device.
    """

    def __init__(self,
                 T: int   = 1000,
                 beta_start: float = 1e-4,
                 beta_end:   float = 0.02,
                 device: torch.device = torch.device("cpu")):

        self.T      = T
        self.device = device

        # ── Beta schedule: linear ─────────────────────────────────────────────
        betas = torch.linspace(beta_start, beta_end, T,
                               dtype=torch.float32, device=device)

        # ── Derived quantities ────────────────────────────────────────────────
        alphas          = 1.0 - betas
        alpha_bar       = torch.cumprod(alphas, dim=0)
        alpha_bar_prev  = torch.cat([torch.ones(1, device=device),
                                     alpha_bar[:-1]])           # shifted by 1

        # Store all on device
        self.betas          = betas
        self.alphas         = alphas
        self.alpha_bar      = alpha_bar
        self.alpha_bar_prev = alpha_bar_prev

        # Precompute sqrt terms used in forward/reverse
        self.sqrt_alpha_bar         = alpha_bar.sqrt()
        self.sqrt_one_minus_ab      = (1.0 - alpha_bar).sqrt()
        self.sqrt_recip_alpha_bar   = (1.0 / alpha_bar).sqrt()
        self.sqrt_recip_m1_alpha_bar= (1.0 / alpha_bar - 1.0).sqrt()

        # DDPM posterior variance
        self.posterior_var = (betas *
                              (1.0 - alpha_bar_prev) /
                              (1.0 - alpha_bar)).clamp(min=1e-20)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _gather(self, values: torch.Tensor,
                t: torch.Tensor, ndim: int) -> torch.Tensor:
        """
        Index `values` at positions `t`, then reshape for broadcasting.
        ndim = number of dims of the target tensor (e.g. 4 for B,C,H,W).
        """
        out = values[t]
        return out.reshape(t.shape[0], *([1] * (ndim - 1)))

    # ── FORWARD PROCESS ───────────────────────────────────────────────────────

    def q_sample(self,
                 x0:    torch.Tensor,
                 t:     torch.Tensor,
                 noise: Optional[torch.Tensor] = None
                 ) -> tuple:
        """
        Forward diffusion: add noise to x0 at timestep t.
        Operates on VGG latent features.

        Args:
            x0    : (B, C, H, W) clean student VGG features
            t     : (B,) integer timesteps in [0, T-1]
            noise : optional pre-sampled noise (for reproducibility)

        Returns:
            x_t   : (B, C, H, W) noisy latent at timestep t
            noise : (B, C, H, W) the noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x0)

        ndim = x0.ndim
        sqrt_ab    = self._gather(self.sqrt_alpha_bar,    t, ndim)
        sqrt_1m_ab = self._gather(self.sqrt_one_minus_ab, t, ndim)

        x_t = sqrt_ab * x0 + sqrt_1m_ab * noise
        return x_t, noise

    # ── REVERSE STEP (DDPM) ───────────────────────────────────────────────────

    def p_sample_ddpm(self,
                      eps_theta: torch.Tensor,
                      x_t:       torch.Tensor,
                      t:         torch.Tensor) -> torch.Tensor:
        """
        One DDPM reverse step: x_t → x_{t-1}

        Args:
            eps_theta : predicted noise from KD-UNet (B, C, H, W)
            x_t       : noisy latent at step t        (B, C, H, W)
            t         : current timestep              (B,)

        Returns:
            x_{t-1}   : (B, C, H, W)
        """
        ndim = x_t.ndim
        betas     = self._gather(self.betas,          t, ndim)
        sqrt_1mab = self._gather(self.sqrt_one_minus_ab, t, ndim)
        sqrt_a    = self._gather(self.alphas.sqrt(),  t, ndim)
        post_var  = self._gather(self.posterior_var,  t, ndim)

        # Predict x0 from x_t and eps_theta
        x0_pred = (x_t - sqrt_1mab * eps_theta) / \
                  self._gather(self.sqrt_alpha_bar, t, ndim).clamp(min=1e-8)
        x0_pred = x0_pred.clamp(-1.0, 1.0)

        # Compute posterior mean
        coef1 = (self._gather(self.alpha_bar_prev, t, ndim).sqrt() * betas /
                 (1.0 - self._gather(self.alpha_bar, t, ndim)))
        coef2 = (sqrt_a * (1.0 - self._gather(self.alpha_bar_prev, t, ndim)) /
                 (1.0 - self._gather(self.alpha_bar, t, ndim)))

        mean = coef1 * x0_pred + coef2 * x_t

        # Add noise only if t > 0
        noise = torch.randn_like(x_t)
        mask  = (t > 0).float().reshape(-1, *([1] * (ndim - 1)))
        return mean + mask * post_var.sqrt() * noise

    # ── DDIM INFERENCE (fast) ─────────────────────────────────────────────────

    @torch.no_grad()
    def ddim_sample(self,
                    unet_fn:   Callable,
                    shape:     tuple,
                    condition: object,
                    steps:     int  = 50,
                    eta:       float = 0.0) -> torch.Tensor:
        """
        DDIM deterministic sampling — paper uses 50 steps at inference.

        Args:
            unet_fn   : callable(x_t, t, condition) → eps_theta
            shape     : output shape (B, C, H', W')
            condition : teacher VGG features (passed to unet_fn)
            steps     : number of DDIM steps (default 50)
            eta       : stochasticity (0 = fully deterministic DDIM)

        Returns:
            x_0       : (B, C, H', W') denoised latent
        """
        # Evenly spaced timesteps from T-1 → 0
        t_seq = torch.linspace(self.T - 1, 0, steps,
                               dtype=torch.long, device=self.device)

        x = torch.randn(shape, device=self.device)

        for i, t_curr in enumerate(t_seq):
            t_batch = t_curr.expand(shape[0])

            # Predict noise
            eps = unet_fn(x, t_batch, condition)

            ab_curr = self.alpha_bar[t_curr].clamp(min=1e-8)
            ab_prev = (self.alpha_bar[t_seq[i + 1]]
                       if i + 1 < len(t_seq)
                       else torch.tensor(1.0, device=self.device))

            # Predict x0 from current noisy x
            x0_pred = ((x - (1 - ab_curr).sqrt() * eps) /
                       ab_curr.sqrt())
            x0_pred = x0_pred.clamp(-1.0, 1.0)

            # DDIM update — deterministic (eta=0)
            # x_{t-1} = sqrt(ab_prev)*x0_pred + sqrt(1-ab_prev)*eps
            dir_xt = (1.0 - ab_prev).clamp(min=0).sqrt() * eps
            noise  = eta * torch.randn_like(x)

            x = ab_prev.sqrt() * x0_pred + dir_xt + noise

        return x

    # ── LOSS WEIGHT (optional — SNR weighting) ────────────────────────────────

    def snr_weight(self, t: torch.Tensor) -> torch.Tensor:
        """
        Signal-to-Noise Ratio based loss weighting.
        SNR(t) = alpha_bar_t / (1 - alpha_bar_t)
        Used to weight L_diff at different timesteps.
        """
        ab = self.alpha_bar[t]
        return ab / (1.0 - ab).clamp(min=1e-8)


# ══════════════════════════════════════════════════════════════════════════════
# SANITY CHECK
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 50)
    print("  Diffusion Scheduler Sanity Check")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device : {device}")

    sched = DiffusionScheduler(T=1000, device=device)

    # ── 1. Schedule values ────────────────────────────────────────────────────
    print(f"\n  Beta schedule:")
    print(f"    beta[0]   : {sched.betas[0].item():.6f}  (expected ~1e-4)")
    print(f"    beta[999] : {sched.betas[-1].item():.6f}  (expected ~0.02)")
    print(f"  Alpha bar:")
    print(f"    ab[0]     : {sched.alpha_bar[0].item():.6f}  (close to 1)")
    print(f"    ab[999]   : {sched.alpha_bar[-1].item():.6f}  (close to 0)")

    # ── 2. Forward process ────────────────────────────────────────────────────
    B, C, H, W = 2, 32, 128, 128
    x0 = torch.randn(B, C, H, W, device=device)

    # t=0: almost no noise
    t0 = torch.zeros(B, dtype=torch.long, device=device)
    xt0, noise0 = sched.q_sample(x0, t0)
    print(f"\n  Forward q_sample (t=0):")
    print(f"    x0 std   : {x0.std():.4f}")
    print(f"    xt std   : {xt0.std():.4f}  (should be ~x0 std)")

    # t=999: almost pure noise
    t999 = torch.full((B,), 999, dtype=torch.long, device=device)
    xt999, _ = sched.q_sample(x0, t999)
    print(f"\n  Forward q_sample (t=999):")
    print(f"    xt std   : {xt999.std():.4f}  (should be ~1.0, pure noise)")

    # ── 3. DDPM reverse step ──────────────────────────────────────────────────
    t_mid = torch.full((B,), 500, dtype=torch.long, device=device)
    xt_mid, noise_mid = sched.q_sample(x0, t_mid)
    # Dummy noise prediction (perfect: use actual noise)
    x_prev = sched.p_sample_ddpm(noise_mid, xt_mid, t_mid)
    print(f"\n  DDPM reverse step (t=500):")
    print(f"    x_prev shape : {tuple(x_prev.shape)}")
    print(f"    NaN          : {torch.isnan(x_prev).any().item()}")

    # ── 4. DDIM inference ─────────────────────────────────────────────────────
    # Dummy UNet that predicts zero noise
    dummy_unet = lambda x, t, cond: torch.zeros_like(x)
    x0_sampled = sched.ddim_sample(dummy_unet,
                                    shape=(B, C, H, W),
                                    condition=None,
                                    steps=10)   # 10 for quick check
    print(f"\n  DDIM sampling (10 steps, zero-noise UNet):")
    print(f"    Output shape : {tuple(x0_sampled.shape)}")
    print(f"    Output std   : {x0_sampled.std():.4f}  (should be ~0, all noise removed)")
    print(f"    NaN          : {torch.isnan(x0_sampled).any().item()}")

    # ── 5. SNR weight ─────────────────────────────────────────────────────────
    t_test = torch.tensor([0, 250, 500, 750, 999], device=device)
    snr    = sched.snr_weight(t_test)
    print(f"\n  SNR weights at t=[0,250,500,750,999]:")
    print(f"    {[f'{v:.2f}' for v in snr.tolist()]}")
    print(f"    (should decrease: high SNR early, low SNR late)")

    print(f"\n  ✓ Diffusion Scheduler ready!")
