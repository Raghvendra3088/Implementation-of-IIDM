import torch
import torch.nn as nn


def make_coord_grid(H, W, device):
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy], dim=-1).reshape(1, H * W, 2)


class SIRENBlock(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        return x + self.fc2(torch.sin(self.fc1(x)))


class PE(nn.Module):
    def __init__(self, L=10):
        super().__init__()
        self.register_buffer('freqs', 2.0 ** torch.arange(L).float())

    def forward(self, coords):
        args = coords.unsqueeze(-1) * self.freqs.view(1,1,1,-1)
        enc = torch.cat([args.sin(), args.cos()], dim=-1)
        return enc.reshape(coords.shape[0], coords.shape[1], -1)


class FeatPool(nn.Module):
    def __init__(self, total_ch=480):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(total_ch, total_ch))

    def forward(self, x):
        return torch.sin(self.proj(x))


class SIRENINRCkpt(nn.Module):
    def __init__(self, student_chs=[32,64,128,256], L=10):
        super().__init__()
        self.pe       = PE(L)
        self.feat_pool= FeatPool(sum(student_chs))  # 480
        self.input_proj = nn.Sequential(nn.Linear(480 + L*4, 256))
        self.blocks   = nn.ModuleList([SIRENBlock(256) for _ in range(4)])
        self.out      = nn.Sequential(nn.Linear(256,64), nn.ReLU(), nn.Linear(64,1), nn.Tanh())

    def forward(self, feats, H=256, W=256):
        B, device = feats[0].shape[0], feats[0].device
        feat_vec = self.feat_pool(torch.cat([f.mean(dim=[2,3]) for f in feats], dim=1))
        coords   = make_coord_grid(H, W, device).expand(B,-1,-1)
        coord_enc= self.pe(coords)
        feat_exp = feat_vec.unsqueeze(1).expand(-1, H*W, -1)
        x = torch.sin(self.input_proj(torch.cat([feat_exp, coord_enc], dim=-1)))
        for blk in self.blocks:
            x = blk(x)
        return self.out(x).reshape(B, 1, H, W)
