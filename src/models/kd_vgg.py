import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class TeacherVGG19(nn.Module):
    """Actual Paper Teacher: VGG-19, pretrained on ImageNet, completely frozen."""
    def __init__(self, in_channels=6):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)

        # Modify first layer for 6 channels (Paper approach: average RGB weights)
        orig_conv = vgg.features[0]
        new_conv = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        with torch.no_grad():
            new_conv.weight[:, :3] = orig_conv.weight
            new_conv.weight[:, 3:] = orig_conv.weight[:, :3] # Duplicate for structural bands
        vgg.features[0] = new_conv

        # Paper exact layer splits for VGG-19
        self.stage1 = vgg.features[0:5]    # Output: 64 ch
        self.stage2 = vgg.features[5:10]   # Output: 128 ch
        self.stage3 = vgg.features[10:19]  # Output: 256 ch
        self.stage4 = vgg.features[19:28]  # Output: 512 ch

        # Freeze all parameters
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        return [f1, f2, f3, f4]

class StudentVGG(nn.Module):
    """Lightweight Student: ~1/8th channels of Teacher."""
    def __init__(self, in_channels=6):
        super().__init__()
        def conv_block(ic, oc):
            return nn.Sequential(
                nn.Conv2d(ic, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
                nn.Conv2d(oc, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            )

        self.stage1 = conv_block(in_channels, 32)
        self.stage2 = conv_block(32, 64)
        self.stage3 = conv_block(64, 128)
        self.stage4 = conv_block(128, 256)

        # Projections to match Teacher's higher dimensions for MSE Loss
        self.proj1 = nn.Conv2d(32, 64, 1)
        self.proj2 = nn.Conv2d(64, 128, 1)
        self.proj3 = nn.Conv2d(128, 256, 1)
        self.proj4 = nn.Conv2d(256, 512, 1)

    def forward(self, x):
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        
        # Upsample logic handled in loss function now for exact alignment
        return [f1, f2, f3, f4], [self.proj1(f1), self.proj2(f2), self.proj3(f3), self.proj4(f4)]

class KDLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, student_projs, teacher_feats):
        loss = 0
        for s, t in zip(student_projs, teacher_feats):
            t_detach = t.detach()
            if s.shape != t_detach.shape:
                s = F.interpolate(s, size=t_detach.shape[-2:], mode='bilinear', align_corners=False)
            loss += F.mse_loss(s, t_detach)
        return loss / 4.0
