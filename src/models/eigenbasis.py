"""
Paper Appendix B — Global eigenbasis W_N,g for PCA-based KD.
Eq. B.3: min_{W W^T=I} (1/|Bt|) sum_N sum_k ||W^T W F_bar_Nk - F_bar_Nk||^2
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalEigenbasis(nn.Module):
    """One eigenbasis W_N,g for one relu-block N. Maps C_N -> C_N_e (reduced)."""
    def __init__(self, C_N, C_N_e):
        super().__init__()
        W = torch.empty(C_N_e, C_N)
        nn.init.orthogonal_(W)
        self.W = nn.Parameter(W)

    @torch.no_grad()
    def orthonormalize(self):
        """Retract W onto Stiefel manifold (W W^T = I) via QR, per Eq. B.3 constraint."""
        Wt = self.W.t()                      # (C_N, C_N_e)
        Q, R = torch.linalg.qr(Wt)
        sign = torch.sign(torch.diagonal(R))
        Q = Q * sign.unsqueeze(0)
        self.W.copy_(Q.t())

    def forward(self, feat):
        """feat: (B, C_N, H, W) -> projected (B, C_N_e, H, W), recon_loss (scalar)"""
        B, C, H, W = feat.shape
        f = feat.view(B, C, H * W)
        f_mean = f.mean(dim=2, keepdim=True)
        f_bar  = f - f_mean                                   # F_bar_Nk (Eq. B.3)

        proj  = torch.einsum('ec,bcs->bes', self.W, f_bar)    # W F_bar   (B, C_e, HW)
        recon = torch.einsum('ec,bes->bcs', self.W, proj)     # W^T W F_bar (B, C, HW)
        recon_loss = F.mse_loss(recon, f_bar)

        return proj.view(B, -1, H, W), recon_loss


class MultiLayerEigenbasis(nn.Module):
    """4 eigenbases for relu blocks N=1..4, paper exact."""
    def __init__(self, teacher_ch, student_ch):
        super().__init__()
        assert len(teacher_ch) == len(student_ch) == 4
        self.bases = nn.ModuleList([
            GlobalEigenbasis(teacher_ch[i], student_ch[i]) for i in range(4)
        ])

    def orthonormalize(self):
        for b in self.bases: b.orthonormalize()

    def forward(self, teacher_feats):
        """teacher_feats: list of 4 tensors (B, C_N, H, W)"""
        projs, losses = [], []
        for i, f in enumerate(teacher_feats):
            p, l = self.bases[i](f)
            projs.append(p)
            losses.append(l)
        return projs, sum(losses) / len(losses)
