"""
Paper Appendix B, Eq. B.1-B.3: Global eigenbasis W_N,g for ALL 16 relu layers.
Note: Eq. B.3 shows N in {1,2,3,4} as an EXAMPLE (paper's own written equation
restricts to 4 for the shown derivation), but Table A2 and the surrounding text
confirm all N=1..16 layers get their own W_N,g. We train all 16.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalEigenbasis(nn.Module):
    def __init__(self, C_N, C_N_e):
        super().__init__()
        W = torch.empty(C_N_e, C_N)
        nn.init.orthogonal_(W)
        self.W = nn.Parameter(W)

    @torch.no_grad()
    def orthonormalize(self):
        Wt = self.W.t()
        Q, R = torch.linalg.qr(Wt)
        sign = torch.sign(torch.diagonal(R))
        Q = Q * sign.unsqueeze(0)
        self.W.copy_(Q.t())

    def forward(self, feat):
        B, C, H, W = feat.shape
        f = feat.view(B, C, H * W)
        f_mean = f.mean(dim=2, keepdim=True)
        f_bar  = f - f_mean

        proj  = torch.einsum('ec,bcs->bes', self.W, f_bar)
        recon = torch.einsum('ec,bes->bcs', self.W, proj)
        recon_loss = F.mse_loss(recon, f_bar)

        return proj.view(B, -1, H, W), recon_loss, f_mean


class MultiLayerEigenbasis16(nn.Module):
    """16 eigenbases, N=1..16, paper exact (Table A2)."""
    def __init__(self, teacher_ch, student_ch):
        super().__init__()
        assert len(teacher_ch) == len(student_ch) == 16
        self.bases = nn.ModuleList([
            GlobalEigenbasis(teacher_ch[i], student_ch[i]) for i in range(16)
        ])

    def orthonormalize(self):
        for b in self.bases: b.orthonormalize()

    def forward(self, teacher_feats):
        projs, means, losses = [], [], []
        for i, f in enumerate(teacher_feats):
            p, l, m = self.bases[i](f)
            projs.append(p); means.append(m); losses.append(l)
        return projs, means, sum(losses) / len(losses)
