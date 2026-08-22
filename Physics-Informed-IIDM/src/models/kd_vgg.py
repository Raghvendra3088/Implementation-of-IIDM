import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights

class VGG16Teacher(nn.Module):
    def __init__(self, in_channels=6):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        features = list(vgg.features)

        # Adapt first conv: 3ch -> 6ch (average weight tiling)
        old_conv = features[0]
        new_conv = nn.Conv2d(in_channels, 64, 3, padding=1)
        with torch.no_grad():
            avg_w = old_conv.weight.mean(dim=1, keepdim=True)
            new_conv.weight.copy_(avg_w.repeat(1, in_channels, 1, 1))
            new_conv.bias.copy_(old_conv.bias)
        features[0] = new_conv

        # 4 hierarchical stages matching VGG16 pooling layers
        self.block1 = nn.Sequential(*features[0:5])   # -> 64 ch, H/2
        self.block2 = nn.Sequential(*features[5:10])  # -> 128 ch, H/4
        self.block3 = nn.Sequential(*features[10:17]) # -> 256 ch, H/8
        self.block4 = nn.Sequential(*features[17:24]) # -> 512 ch, H/16

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        f1 = self.block1(x)
        f2 = self.block2(f1)
        f3 = self.block3(f2)
        f4 = self.block4(f3)
        return [f1, f2, f3, f4]

    def train(self, mode=True):
        return super().train(False)


class _ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


class LightweightStudentEncoder(nn.Module):
    def __init__(self, in_channels=6, student_channels=[32, 64, 128, 256], teacher_channels=[64, 128, 256, 512]):
        super().__init__()
        ch = student_channels
        
        self.enc1 = nn.Sequential(
            _ConvBnRelu(in_channels, ch[0]),
            _ConvBnRelu(ch[0], ch[0]),
            nn.MaxPool2d(2)
        )
        self.enc2 = nn.Sequential(
            _ConvBnRelu(ch[0], ch[1]),
            _ConvBnRelu(ch[1], ch[1]),
            nn.MaxPool2d(2)
        )
        self.enc3 = nn.Sequential(
            _ConvBnRelu(ch[1], ch[2]),
            _ConvBnRelu(ch[2], ch[2]),
            _ConvBnRelu(ch[2], ch[2]),
            nn.MaxPool2d(2)
        )
        self.enc4 = nn.Sequential(
            _ConvBnRelu(ch[2], ch[3]),
            _ConvBnRelu(ch[3], ch[3]),
            _ConvBnRelu(ch[3], ch[3]),
            nn.MaxPool2d(2)
        )

        # 1x1 projection for channel alignment with teacher
        self.proj = nn.ModuleList([
            nn.Conv2d(ch[i], teacher_channels[i], 1, bias=False)
            for i in range(4)
        ])
        
        # Latent Projection Layer: E(F_S^L) -> z_0 (256 ch, H/16 x W/16)
        self.latent_proj = nn.Conv2d(ch[3], 256, 3, padding=1)

    def forward(self, x):
        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        f4 = self.enc4(f3)
        
        student_feats = [f1, f2, f3, f4]
        projected_feats = [self.proj[i](student_feats[i]) for i in range(4)]
        
        z_0 = self.latent_proj(f4)
        
        return student_feats, projected_feats, z_0


class HierarchicalKDLoss(nn.Module):
    """
    Equation from Section 4.5:
    L_KD = sum \lambda_l || norm(F_T) - norm(\tilde{F}_S) ||_2^2
    """
    def __init__(self, lambdas=[1.0, 1.0, 1.0, 1.0], eps=1e-8):
        super().__init__()
        self.lambdas = lambdas
        self.eps = eps

    def forward(self, projected_student, teacher_feats):
        loss = 0.0
        for i, (s, t) in enumerate(zip(projected_student, teacher_feats)):
            t = t.detach()
            
            # Feature Normalization (Z-score)
            s_mean = s.mean(dim=[2,3], keepdim=True)
            s_std = s.std(dim=[2,3], keepdim=True) + self.eps
            s_norm = (s - s_mean) / s_std
            
            t_mean = t.mean(dim=[2,3], keepdim=True)
            t_std = t.std(dim=[2,3], keepdim=True) + self.eps
            t_norm = (t - t_mean) / t_std
            
            # L2 loss
            loss += self.lambdas[i] * F.mse_loss(s_norm, t_norm)
            
        return loss / len(projected_student)
