"""
Base IIDM paper - KD-UNet (exact paper implementation).
Paper Figure 4(b):
  - f(0) concat with y_t -> UNet encoder
  - f(i) = Conv(f(i-1)) downsampled at each scale (Eq.2)
  - Cross-attention between encoder u(i) and conditional f(i)
  - 2-layer MLP implicit upsampler in decoder (Eq.3)
Paper Table A2 UNet distilled channels: (44,44,88,88,176,176,352,352,704,704)
We use 4-level UNet: [44, 88, 176, 352]
Output: same H,W as input (full resolution noise prediction)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

UNET_CH = [44, 88, 176, 352]


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
            nn.Linear(dim, dim * 4), nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )
    def forward(self, t):
        return self.mlp(sinusoidal_emb(t, self.dim))


class CrossAttention(nn.Module):
    """
    Paper Figure 4(b): cross-attention between u(i) and f(i).
    Channel-wise via GAP — avoids pixel-level OOM.
    """
    def __init__(self, ch, cond_ch):
        super().__init__()
        self.cond_proj = nn.Conv2d(cond_ch, ch, 1)
        self.q   = nn.Linear(ch, ch)
        self.k   = nn.Linear(ch, ch)
        self.v   = nn.Linear(ch, ch)
        self.out = nn.Linear(ch, ch)
        self.scale = ch ** -0.5

    def forward(self, u, f):
        f_r   = F.interpolate(f, size=u.shape[-2:], mode='bilinear', align_corners=False)
        f_r   = self.cond_proj(f_r)
        u_gap = u.mean(dim=[2, 3])
        f_gap = f_r.mean(dim=[2, 3])
        q = self.q(u_gap)
        k = self.k(f_gap)
        v = self.v(f_gap)
        attn = torch.softmax((q * k) * self.scale, dim=-1)
        out  = self.out(attn * v).unsqueeze(-1).unsqueeze(-1)
        return u + out


class ImplicitMLPUp(nn.Module):
    """Paper Eq.3: u_up(i) = D_i(h_hat(i+1)) — 2-layer MLP upsampler."""
    def __init__(self, in_ch, out_ch, scale=2):
        super().__init__()
        self.scale = scale
        self.net = nn.Sequential(
            nn.Linear(in_ch, in_ch * 4),
            nn.ReLU(),
            nn.Linear(in_ch * 4, out_ch * scale * scale)
        )
    def forward(self, x):
        B, C, H, W = x.shape
        flat = x.permute(0, 2, 3, 1).reshape(B * H * W, C)
        out  = self.net(flat).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return F.pixel_shuffle(out, self.scale)


def _gn(ch):
    for g in [8, 4, 2, 1]:
        if ch % g == 0:
            return nn.GroupNorm(g, ch)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            _gn(out_ch), nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            _gn(out_ch), nn.SiLU(),
        )
    def forward(self, x): return self.net(x)


class BaseKDUNet(nn.Module):
    """
    Paper Figure 4(b) exact implementation.
    4-level encoder + bottleneck + 4-level decoder.
    All 4 skip connections used -> output is full H,W resolution.
    """
    def __init__(self, in_ch, t_dim=128, cond_ch=79):
        super().__init__()
        ch = UNET_CH   # [44, 88, 176, 352]

        self.t_emb  = TimeEmb(t_dim)
        self.t_proj = nn.ModuleList([nn.Linear(t_dim, c) for c in ch])

        # f(i) = Conv(f(i-1)) — paper Eq.2, stride=2 downsampling
        self.f_convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(cond_ch, cond_ch, 3, stride=2, padding=1), nn.ReLU())
            for _ in range(4)
        ])

        # Cross-attention at each encoder level
        self.cross_attn = nn.ModuleList([
            CrossAttention(ch[i], cond_ch) for i in range(4)
        ])

        self.pool = nn.MaxPool2d(2)

        # Encoder: in_ch -> 44 -> 88 -> 176 -> 352
        self.enc1 = DoubleConv(in_ch,  ch[0])
        self.enc2 = DoubleConv(ch[0],  ch[1])
        self.enc3 = DoubleConv(ch[1],  ch[2])
        self.enc4 = DoubleConv(ch[2],  ch[3])
        self.bot  = DoubleConv(ch[3],  ch[3])

        # Decoder — 4 levels, all skip connections used
        # up3: bot(352)->176, cat e4(352) -> 528, dec3->176
        self.up3  = ImplicitMLPUp(ch[3], ch[2])
        self.dec3 = DoubleConv(ch[2] + ch[3], ch[2])

        # up2: 176->88, cat e3(176) -> 264, dec2->88
        self.up2  = ImplicitMLPUp(ch[2], ch[1])
        self.dec2 = DoubleConv(ch[1] + ch[2], ch[1])

        # up1: 88->44, cat e2(88) -> 132, dec1->44
        self.up1  = ImplicitMLPUp(ch[1], ch[0])
        self.dec1 = DoubleConv(ch[0] + ch[1], ch[0])

        # up0: 44->44, cat e1(44) -> 88, dec0->44  [restores full H,W]
        self.up0  = ImplicitMLPUp(ch[0], ch[0])
        self.dec0 = DoubleConv(ch[0] + ch[0], ch[0])

        self.out  = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x, t, f0):
        """
        x : (B, in_ch, H, W)
        t : (B,) timestep
        f0: (B, cond_ch, H, W) student f(0) upsampled
        Returns: (B, 1, H, W) — full resolution noise prediction
        """
        te = self.t_emb(t)

        # f(i) = Conv(f(i-1)) — paper Eq.2
        fi = [f0]
        for conv in self.f_convs:
            fi.append(conv(fi[-1]))

        # Encoder
        e1 = self.enc1(x)
        e1 = e1 + self.t_proj[0](te)[:, :, None, None]
        e1 = self.cross_attn[0](e1, fi[1])

        e2 = self.enc2(self.pool(e1))
        e2 = e2 + self.t_proj[1](te)[:, :, None, None]
        e2 = self.cross_attn[1](e2, fi[2])

        e3 = self.enc3(self.pool(e2))
        e3 = e3 + self.t_proj[2](te)[:, :, None, None]
        e3 = self.cross_attn[2](e3, fi[3])

        e4 = self.enc4(self.pool(e3))
        e4 = e4 + self.t_proj[3](te)[:, :, None, None]
        e4 = self.cross_attn[3](e4, fi[4])

        b = self.bot(self.pool(e4))

        # Decoder — implicit MLP upsampler (paper Eq.3)
        d3 = self.dec3(torch.cat([self.up3(b),  e4], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e3], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e2], dim=1))
        d0 = self.dec0(torch.cat([self.up0(d1), e1], dim=1))
        return self.out(d0)   # (B, 1, H, W)
