"""
Physics-Informed IIDM — Feature Extractor
SwinV2-Base + MoE (4 experts, top-2) for carbon estimation.
SwinV2 has cross-attention via shifted-window mechanism.
Input: (B, 6, 256, 256) — S2 4-band + DEM + Canopy
Output: (B, 256, 8, 8) — rich multi-scale features
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# cuDNN 9.2 incompatible with CUDA 12.2 on this server
torch.backends.cudnn.enabled = False

# ── Mixture of Experts ────────────────────────────────────────────────────────
class Expert(nn.Module):
    """Single FFN expert."""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )
    def forward(self, x): return self.net(x)

class MoELayer(nn.Module):
    """
    Sparse MoE: 4 experts, top-2 routing.
    Input: (B, N, C) — sequence of tokens
    Output: (B, N, C)
    """
    def __init__(self, dim, n_experts=4, top_k=2, hidden_ratio=2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k     = top_k
        self.experts   = nn.ModuleList([
            Expert(dim, dim * hidden_ratio) for _ in range(n_experts)
        ])
        self.gate = nn.Linear(dim, n_experts, bias=False)

    def forward(self, x):
        # x: (B, N, C)
        B, N, C = x.shape
        x_flat = x.reshape(B*N, C)

        # Routing
        scores  = self.gate(x_flat)                          # (B*N, n_experts)
        topk_w, topk_idx = torch.topk(scores, self.top_k, dim=-1)
        topk_w  = F.softmax(topk_w, dim=-1)                 # normalize weights

        # Expert computation
        out = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            # Find tokens routed to expert i
            mask = (topk_idx == i).any(dim=-1)              # (B*N,)
            if mask.sum() == 0: continue
            expert_out = expert(x_flat[mask])
            # Get weight for this expert
            w_idx = (topk_idx[mask] == i).float()
            w     = (topk_w[mask] * w_idx).sum(dim=-1, keepdim=True)
            out[mask] += w * expert_out

        return out.reshape(B, N, C)

# ── SwinV2 + MoE Encoder ─────────────────────────────────────────────────────
class SwinMoEEncoder(nn.Module):
    """
    SwinV2-Base encoder with MoE on deepest features.
    Uses shifted-window attention (cross-window interaction via SW-MSA).
    Input:  (B, 6, 256, 256)
    Output: list[(B, 256, 8, 8)]  — compatible with BaseKDUNet
    """
    def __init__(self, in_chans=6, out_ch=256, pretrained=True, n_experts=4):
        super().__init__()

        # SwinV2-Base: img_size=256, window_size=8 (fits 256px patches)
        self.swin = timm.create_model(
            'swinv2_base_window8_256',
            pretrained=pretrained,
            in_chans=in_chans,
            img_size=256,
            num_classes=0,
            features_only=True,
            out_indices=(3,)          # deepest: (B, H/32, W/32, 1024)
        )

        # MoE on top-level features
        self.moe = MoELayer(dim=1024, n_experts=n_experts, top_k=2)

        # Project 1024 → out_ch (256) for UNet compatibility
        self.proj = nn.Sequential(
            nn.Conv2d(1024, out_ch, 1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.GELU()
        )

    def forward(self, x):
        # Swin features
        feats = self.swin(x)                              # [(B, H, W, 1024)]
        f     = feats[0]                                  # (B, 8, 8, 1024)

        # MoE — flatten spatial to sequence
        B, H, W, C = f.shape
        f_seq = f.reshape(B, H*W, C)                     # (B, 64, 1024)
        f_moe = self.moe(f_seq)                           # (B, 64, 1024)
        f_moe = f_moe.reshape(B, H, W, C)

        # channels-last → channels-first
        f_out = f_moe.permute(0, 3, 1, 2).contiguous()  # (B, 1024, 8, 8)
        return [self.proj(f_out)]                         # [(B, 256, 8, 8)]


class SwinMoEEncoderWithSkip(nn.Module):
    """
    Multi-scale version for future SAR cross-attention fusion.
    Returns 4 feature maps: [(B,128,64,64),(B,256,32,32),(B,512,16,16),(B,256,8,8)]
    """
    def __init__(self, in_chans=6, pretrained=True, n_experts=4):
        super().__init__()
        self.swin = timm.create_model(
            'swinv2_base_window8_256',
            pretrained=pretrained,
            in_chans=in_chans,
            img_size=256,
            num_classes=0,
            features_only=True,
            out_indices=(0, 1, 2, 3)
        )
        self.moe      = MoELayer(dim=1024, n_experts=n_experts, top_k=2)
        self.proj_last = nn.Sequential(
            nn.Conv2d(1024, 256, 1, bias=False),
            nn.GroupNorm(8, 256), nn.GELU()
        )

    def forward(self, x):
        feats = self.swin(x)
        out   = []
        for i, f in enumerate(feats):
            fc = f.permute(0, 3, 1, 2).contiguous()
            if i == len(feats)-1:
                B, C, H, W = fc.shape
                f_seq = fc.flatten(2).transpose(1,2)
                f_moe = self.moe(f_seq)
                fc    = f_moe.transpose(1,2).reshape(B,C,H,W)
                fc    = self.proj_last(fc)
            out.append(fc)
        return out
