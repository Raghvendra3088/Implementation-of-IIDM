import torch
import torch.nn as nn
from src.models.eigenbasis16 import GlobalEigenbasis

TEACHER_CH = [64, 128, 256, 512, 1024, 512, 256, 128, 64]
STUDENT_CH = [44, 88, 176, 352, 704, 352, 176, 88, 44]

class MultiLayerEigenbasisUNet(nn.Module):
    def __init__(self, teacher_ch=TEACHER_CH, student_ch=STUDENT_CH):
        super().__init__()
        self.bases = nn.ModuleList([
            GlobalEigenbasis(tc, sc) for tc, sc in zip(teacher_ch, student_ch)
        ])
        
    def forward(self, features):
        """
        features: list of 9 tensors (e1, e2, e3, e4, bot, d4, d3, d2, d1)
        """
        projections = []
        means = []
        total_recon_loss = 0.0
        
        for i, feat in enumerate(features):
            proj, recon_loss, mean = self.bases[i](feat)
            projections.append(proj)
            means.append(mean)
            total_recon_loss += recon_loss
            
        return projections, means, total_recon_loss

    def orthonormalize(self):
        for base in self.bases:
            base.orthonormalize()
