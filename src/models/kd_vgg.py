"""
KD-VGG — Knowledge Distillation VGG Feature Extractor
=======================================================
Paper: IIDM Section 3.1

Teacher : VGG16 pretrained (ImageNet), first conv modified 3→6 channels
          Weights frozen — only used for distillation signal

Student : Lightweight VGG, half channels at each block
          Trainable — learns to mimic teacher features

KD Loss : MSE between student and teacher features at 4 scales
          L_kd = (1/4) Σ MSE(student_feat_i, teacher_feat_i.detach())

Input   : (B, 6, H, W)  — B02, B03, B04, B08, DEM, Canopy  range [-1,1]
Output  : 4 feature maps per model
          f1: (B, C,   H,   W  )
          f2: (B, 2C,  H/2, W/2)
          f3: (B, 4C,  H/4, W/4)
          f4: (B, 8C,  H/8, W/8)
          Teacher C=64, Student C=32
"""

import torch
import torch.nn as nn
import torchvision.models as tvm
from typing import List, Tuple


# ══════════════════════════════════════════════════════════════════════════════
# TEACHER VGG
# ══════════════════════════════════════════════════════════════════════════════

class TeacherVGG(nn.Module):
    """
    VGG16 pretrained backbone modified for 6-channel satellite input.
    Weights are FROZEN — acts only as feature distillation target.

    4 feature blocks extracted:
        Block 1 → (B, 64,  H,   W  )
        Block 2 → (B, 128, H/2, W/2)
        Block 3 → (B, 256, H/4, W/4)
        Block 4 → (B, 512, H/8, W/8)
    """

    def __init__(self, in_channels: int = 6):
        super().__init__()

        # Load pretrained VGG16
        vgg = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1)
        features = list(vgg.features.children())

        # ── Modify first conv: 3 → 6 channels ─────────────────────────────────
        # Strategy: average original 3-channel weights, tile to 6 channels
        old_conv   = features[0]                         # Conv2d(3, 64, 3, 1, 1)
        new_conv   = nn.Conv2d(in_channels, 64, 3, padding=1, bias=True)

        with torch.no_grad():
            # Mean over input channel dim → (64, 1, 3, 3)
            avg_w = old_conv.weight.mean(dim=1, keepdim=True)
            # Tile to 6 channels → (64, 6, 3, 3)
            new_conv.weight.copy_(avg_w.repeat(1, in_channels, 1, 1))
            new_conv.bias.copy_(old_conv.bias)

        features[0] = new_conv

        # ── Split into 4 feature blocks ────────────────────────────────────────
        # VGG16 layer indices:
        #   Block1: 0-4   (conv-relu-conv-relu-pool)
        #   Block2: 5-9   (conv-relu-conv-relu-pool)
        #   Block3: 10-18 (conv-relu-conv-relu-conv-relu-conv-relu-pool) ← wait paper uses 3.1 blocks
        #   Block4: 19-27 (conv-relu-conv-relu-conv-relu-conv-relu-pool)
        # VGG16 exact indices:
        # [0]  Conv2d(6,64)   [1]  ReLU
        # [2]  Conv2d(64,64)  [3]  ReLU  [4]  MaxPool  → 64ch
        # [5]  Conv2d(64,128) [6]  ReLU
        # [7]  Conv2d(128,128)[8]  ReLU  [9]  MaxPool  → 128ch
        # [10] Conv2d(128,256)[11] ReLU
        # [12] Conv2d(256,256)[13] ReLU
        # [14] Conv2d(256,256)[15] ReLU  [16] MaxPool  → 256ch
        # [17] Conv2d(256,512)[18] ReLU
        # [19] Conv2d(512,512)[20] ReLU
        # [21] Conv2d(512,512)[22] ReLU  [23] MaxPool  → 512ch
        self.block1 = nn.Sequential(*features[0:5])    # → 64  ch, H,   W
        self.block2 = nn.Sequential(*features[5:10])   # → 128 ch, H/2, W/2
        self.block3 = nn.Sequential(*features[10:17])  # → 256 ch, H/4, W/4
        self.block4 = nn.Sequential(*features[17:24])  # → 512 ch, H/8, W/8

        # Freeze all weights
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        f1 = self.block1(x)
        f2 = self.block2(f1)
        f3 = self.block3(f2)
        f4 = self.block4(f3)
        return [f1, f2, f3, f4]

    def train(self, mode: bool = True):
        # Keep teacher always in eval mode — weights are frozen
        return super().train(False)


# ══════════════════════════════════════════════════════════════════════════════
# STUDENT VGG
# ══════════════════════════════════════════════════════════════════════════════

class _ConvBnRelu(nn.Module):
    """Conv2d → BatchNorm → ReLU (building block)."""
    def __init__(self, in_ch: int, out_ch: int,
                 kernel: int = 3, padding: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


class _VGGBlock(nn.Module):
    """2-conv VGG block followed by MaxPool."""
    def __init__(self, in_ch: int, out_ch: int, n_convs: int = 2):
        super().__init__()
        layers = [_ConvBnRelu(in_ch, out_ch)]
        for _ in range(n_convs - 1):
            layers.append(_ConvBnRelu(out_ch, out_ch))
        layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x): return self.block(x)


class StudentVGG(nn.Module):
    """
    Lightweight VGG-style student — half channels of Teacher at each block.
    Fully trainable. Distilled from TeacherVGG via KD loss.

        Block 1 → (B, 32,  H,   W  )
        Block 2 → (B, 64,  H/2, W/2)
        Block 3 → (B, 128, H/4, W/4)
        Block 4 → (B, 256, H/8, W/8)
    """

    # Channel config: teacher_ch // 2
    CHANNELS = [32, 64, 128, 256]

    def __init__(self, in_channels: int = 6):
        super().__init__()
        C = self.CHANNELS

        # Mirror VGG16 depth but half channels
        self.block1 = self._make_block(in_channels, C[0], n_convs=2)
        self.block2 = self._make_block(C[0],        C[1], n_convs=2)
        self.block3 = self._make_block(C[1],        C[2], n_convs=3)
        self.block4 = self._make_block(C[2],        C[3], n_convs=3)

        self._init_weights()

    @staticmethod
    def _make_block(in_ch, out_ch, n_convs):
        layers = [_ConvBnRelu(in_ch, out_ch)]
        for _ in range(n_convs - 1):
            layers.append(_ConvBnRelu(out_ch, out_ch))
        layers.append(nn.MaxPool2d(2, 2))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        s1 = self.block1(x)
        s2 = self.block2(s1)
        s3 = self.block3(s2)
        s4 = self.block4(s3)
        return [s1, s2, s3, s4]


# ══════════════════════════════════════════════════════════════════════════════
# KD LOSS
# ══════════════════════════════════════════════════════════════════════════════

class KDLoss(nn.Module):
    """
    Knowledge Distillation Loss — Paper Eq. (2)

    L_kd = (1/4) Σ_i MSE(student_feat_i, teacher_feat_i)

    Teacher features are DETACHED (no gradient flows to teacher).
    Student features need size-alignment with teacher — done via
    adaptive avg pooling to match teacher spatial dims.
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

        # 1×1 conv adapters: student_ch → teacher_ch at each scale
        # Needed because student has half the channels of teacher
        student_chs = StudentVGG.CHANNELS            # [32, 64, 128, 256]
        teacher_chs = [64, 128, 256, 512]            # VGG16 default
        self.adapters = nn.ModuleList([
            nn.Conv2d(s, t, kernel_size=1, bias=False)
            for s, t in zip(student_chs, teacher_chs)
        ])

    def forward(self,
                student_feats,
                teacher_feats):
        import torch.nn as nn
        assert len(student_feats) == len(teacher_feats) == 4
        total = 0.0
        for i, (s_feat, t_feat) in enumerate(zip(student_feats, teacher_feats)):
            t = t_feat.detach()
            s_adapted = self.adapters[i](s_feat)
            if s_adapted.shape[-2:] != t.shape[-2:]:
                s_adapted = nn.functional.adaptive_avg_pool2d(s_adapted, t.shape[-2:])
            s_norm = (s_adapted - s_adapted.mean()) / (s_adapted.std() + 1e-6)
            t_norm = (t - t.mean()) / (t.std() + 1e-6)
            total = total + self.mse(s_norm, t_norm)
        return total / 4.0


        return total / 4.0


# ══════════════════════════════════════════════════════════════════════════════
# QUICK SANITY CHECK
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 50)
    print("  KD-VGG Sanity Check")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device : {device}")

    dummy = torch.randn(2, 6, 256, 256).to(device)

    # Teacher
    teacher = TeacherVGG(in_channels=6).to(device)
    teacher.eval()
    with torch.no_grad():
        t_feats = teacher(dummy)

    print("\n  Teacher feature shapes:")
    for i, f in enumerate(t_feats):
        print(f"    Block {i+1}: {tuple(f.shape)}")

    # Student
    student = StudentVGG(in_channels=6).to(device)
    s_feats = student(dummy)

    print("\n  Student feature shapes:")
    for i, f in enumerate(s_feats):
        print(f"    Block {i+1}: {tuple(f.shape)}")

    # KD Loss
    kd_loss_fn = KDLoss().to(device)
    loss = kd_loss_fn(s_feats, t_feats)
    print(f"\n  KD Loss : {loss.item():.6f}")

    # Parameter counts
    teacher_params = sum(p.numel() for p in teacher.parameters()) / 1e6
    student_params = sum(p.numel() for p in student.parameters()
                         if p.requires_grad) / 1e6
    print(f"\n  Teacher params (frozen) : {teacher_params:.2f}M")
    print(f"  Student params (train)  : {student_params:.2f}M")
    print(f"\n  ✓ KD-VGG ready!")
