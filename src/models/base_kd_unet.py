"""
Base IIDM paper - KD-UNet (exact paper implementation).
Paper Figure 4(b):
  - f(0) concat with y_t -> UNet encoder (Actually: input is concat(x, y_t))
  - f(i) = Conv(f(i-1)) downsampled at each scale (Eq.2)
  - Cross-attention between encoder u(i) and conditional f(i)
  - 2-layer MLP implicit upsampler in decoder (Eq.3)
Paper Table A2 UNet distilled channels: (44,44,88,88,176,176,352,352,704,704)
We use 5-level UNet: [44, 88, 176, 352, 704]
Output: same H,W as input (full resolution noise prediction)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

UNET_CH = [44, 88, 176, 352, 704]


def sinusoidal_emb(t, dim):
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device).float() / half
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([args.sin(), args.cos()], dim=-1)


class TimeEmb(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )
    def forward(self, t):
        return self.mlp(sinusoidal_emb(t, self.dim))


class CrossAttentionMLP(nn.Module):
    """
    Paper Figure 4(b): cross-attention between u(i) and f(i), followed by MLP.
    Proper spatial cross-attention between local UNet features (Q) and global VGG context (K,V).
    """
    def __init__(self, ch, cond_ch):
        super().__init__()
        self.cond_proj = nn.Conv2d(cond_ch, ch, 1)
        self.q   = nn.Linear(ch, ch)
        self.k   = nn.Linear(ch, ch)
        self.v   = nn.Linear(ch, ch)
        self.out = nn.Linear(ch, ch)
        self.scale = ch ** -0.5
        
        # Paper student parameter counts suggest no * 4 expansion in the MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(ch, ch, 1),
            nn.ReLU(),
            nn.Conv2d(ch, ch, 1)
        )

    def forward(self, u, f):
        B, C, H, W = u.shape
        
        # Spatial dimensions of u and f match perfectly!
        f_r = self.cond_proj(f)  # (B, C, H, W)
        
        # Flatten spatial dimensions for sequence attention
        u_flat = u.view(B, C, -1).transpose(1, 2)
        f_flat = f_r.view(B, C, -1).transpose(1, 2)
        
        q = self.q(u_flat)
        k = self.k(f_flat)
        v = self.v(f_flat)
        
        # Cross Attention: Q(B, H*W, C) @ K^T(B, C, H*W) -> (B, H*W, H*W)
        attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) * self.scale, dim=-1)
        
        # Output: (B, H*W, H*W) @ V(B, H*W, C) -> (B, H*W, C)
        attn_out = self.out(torch.bmm(attn, v))
        
        # Reshape back to spatial
        attn_out = attn_out.transpose(1, 2).view(B, C, H, W)
        
        fused = u + attn_out
        return fused + self.mlp(fused)


class ImplicitMLPUp(nn.Module):
    """Paper Eq.3: u_up(i) = D_i(h_hat(i+1)) — 2-layer MLP upsampler."""
    def __init__(self, in_ch, out_ch, scale=2):
        super().__init__()
        self.scale = scale
        # No expansion to match paper parameter count of 14.6M
        self.net = nn.Sequential(
            nn.Linear(in_ch, in_ch),
            nn.ReLU(),
            nn.Linear(in_ch, out_ch * scale * scale)
        )
    def forward(self, x):
        B, C, H, W = x.shape
        flat = x.permute(0, 2, 3, 1).reshape(B * H * W, C)
        out  = self.net(flat).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return F.pixel_shuffle(out, self.scale)


class SingleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(),
        )
    def forward(self, x): return self.net(x)


class BaseKDUNet(nn.Module):
    """
    Paper Figure 4(b) exact implementation.
    5-level encoder + bottleneck + 4-level decoder.
    Output is full H,W resolution.
    """
    def __init__(self, in_ch=5, cond_chs=[34, 79, 154, 123, 16]):
        super().__init__()
        
        ch = UNET_CH

        self.t_emb = TimeEmb(ch[0])
        self.t_proj = nn.ModuleList([
            nn.Linear(ch[0], c) for c in ch
        ])

        # Cross-attention at each of the 5 encoder levels using multi-scale VGG features
        self.cross_attn = nn.ModuleList([
            CrossAttentionMLP(c, cond_ch) for c, cond_ch in zip(ch, cond_chs)
        ])

        self.pool = nn.MaxPool2d(2)

        # Encoder: in_ch -> 44 -> 88 -> 176 -> 352 -> 704
        self.enc1 = SingleConv(in_ch, ch[0])
        self.enc2 = SingleConv(ch[0], ch[1])
        self.enc3 = SingleConv(ch[1], ch[2])
        self.enc4 = SingleConv(ch[2], ch[3])
        self.bot  = SingleConv(ch[3], ch[4])

        # Decoder — 4 levels
        # up4: bot(704)->352, cat e4(352) -> 704, dec4->352
        self.up4  = ImplicitMLPUp(ch[4], ch[3])
        self.dec4 = SingleConv(ch[3] + ch[3], ch[3])

        # up3: 352->176, cat e3(176) -> 352, dec3->176
        self.up3  = ImplicitMLPUp(ch[3], ch[2])
        self.dec3 = SingleConv(ch[2] + ch[2], ch[2])

        # up2: 176->88, cat e2(88) -> 176, dec2->88
        self.up2  = ImplicitMLPUp(ch[2], ch[1])
        self.dec2 = SingleConv(ch[1] + ch[1], ch[1])

        # up1: 88->44, cat e1(44) -> 88, dec1->44 [restores full H,W]
        self.up1  = ImplicitMLPUp(ch[1], ch[0])
        self.dec1 = SingleConv(ch[0] + ch[0], ch[0])

        self.out  = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x, t, f_multi):
        """
        x : (B, in_ch, H, W) where in_ch = 5 (4 optical bands + 1 noisy carbon)
        t : (B,) timestep
        f_multi: list of 5 (B, C, H, W) multi-scale VGG features
        Returns: (B, 1, H, W) — full resolution noise prediction
        """
        te = self.t_emb(t)
        fi = f_multi

        # Encoder
        e1 = self.enc1(x)
        e1 = e1 + self.t_proj[0](te)[:, :, None, None]
        e1 = self.cross_attn[0](e1, fi[0])

        e2 = self.enc2(self.pool(e1))
        e2 = e2 + self.t_proj[1](te)[:, :, None, None]
        e2 = self.cross_attn[1](e2, fi[1])

        e3 = self.enc3(self.pool(e2))
        e3 = e3 + self.t_proj[2](te)[:, :, None, None]
        e3 = self.cross_attn[2](e3, fi[2])

        e4 = self.enc4(self.pool(e3))
        e4 = e4 + self.t_proj[3](te)[:, :, None, None]
        e4 = self.cross_attn[3](e4, fi[3])

        b = self.bot(self.pool(e4))
        b = b + self.t_proj[4](te)[:, :, None, None]
        b = self.cross_attn[4](b, fi[4])

        # Decoder — implicit MLP upsampler (paper Eq.3)
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        return self.out(d1)   # (B, 1, H, W)

    def forward_features(self, x, t, f_multi):
        te = self.t_emb(t)
        fi = f_multi

        e1 = self.enc1(x)
        e1 = e1 + self.t_proj[0](te)[:, :, None, None]
        e1 = self.cross_attn[0](e1, fi[0])

        e2 = self.enc2(self.pool(e1))
        e2 = e2 + self.t_proj[1](te)[:, :, None, None]
        e2 = self.cross_attn[1](e2, fi[1])

        e3 = self.enc3(self.pool(e2))
        e3 = e3 + self.t_proj[2](te)[:, :, None, None]
        e3 = self.cross_attn[2](e3, fi[2])

        e4 = self.enc4(self.pool(e3))
        e4 = e4 + self.t_proj[3](te)[:, :, None, None]
        e4 = self.cross_attn[3](e4, fi[3])

        b = self.bot(self.pool(e4))
        b = b + self.t_proj[4](te)[:, :, None, None]
        b = self.cross_attn[4](b, fi[4])

        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        return [e1, e2, e3, e4, b, d4, d3, d2, d1]



class TeacherUNet(nn.Module):
    """
    Teacher UNet: 31M parameters, trained from scratch using VGG Teacher features.
    Cross-attention uses multi-scale features from Teacher VGG (64, 128, 256, 512, 512).
    """
    def __init__(self, in_ch=5, cond_chs=[64, 128, 256, 512, 512]):
        super().__init__()
        ch = [64, 128, 256, 512, 1024]
        
        self.t_emb = TimeEmb(128)
        self.t_proj = nn.ModuleList([nn.Linear(128, c) for c in ch])
        
        self.pool = nn.MaxPool2d(2)

        self.enc1 = SingleConv(in_ch, ch[0])
        self.enc2 = SingleConv(ch[0], ch[1])
        self.enc3 = SingleConv(ch[1], ch[2])
        self.enc4 = SingleConv(ch[2], ch[3])

        self.cross_attn = nn.ModuleList([
            CrossAttentionMLP(c, cond_ch) for c, cond_ch in zip(ch, cond_chs)
        ])

        self.bot = SingleConv(ch[3], ch[4])

        self.up4  = ImplicitMLPUp(ch[4], ch[3])
        self.dec4 = SingleConv(ch[3] + ch[3], ch[3])

        self.up3  = ImplicitMLPUp(ch[3], ch[2])
        self.dec3 = SingleConv(ch[2] + ch[2], ch[2])

        self.up2  = ImplicitMLPUp(ch[2], ch[1])
        self.dec2 = SingleConv(ch[1] + ch[1], ch[1])

        self.up1  = ImplicitMLPUp(ch[1], ch[0])
        self.dec1 = SingleConv(ch[0] + ch[0], ch[0])

        self.out  = nn.Conv2d(ch[0], 1, 1)   # (B, 1, H, W)
        
    def forward(self, x, t, f_multi):
        feats = self.forward_features(x, t, f_multi)
        return self.out(feats[-1])
        
    def forward_features(self, x, t, f_multi):
        te = self.t_emb(t)
        fi = f_multi

        e1 = self.enc1(x)
        e1 = e1 + self.t_proj[0](te)[:, :, None, None]
        e1 = self.cross_attn[0](e1, fi[0])

        e2 = self.enc2(self.pool(e1))
        e2 = e2 + self.t_proj[1](te)[:, :, None, None]
        e2 = self.cross_attn[1](e2, fi[1])

        e3 = self.enc3(self.pool(e2))
        e3 = e3 + self.t_proj[2](te)[:, :, None, None]
        e3 = self.cross_attn[2](e3, fi[2])

        e4 = self.enc4(self.pool(e3))
        e4 = e4 + self.t_proj[3](te)[:, :, None, None]
        e4 = self.cross_attn[3](e4, fi[3])

        b = self.bot(self.pool(e4))
        b = b + self.t_proj[4](te)[:, :, None, None]
        b = self.cross_attn[4](b, fi[4])

        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        return [e1, e2, e3, e4, b, d4, d3, d2, d1]
