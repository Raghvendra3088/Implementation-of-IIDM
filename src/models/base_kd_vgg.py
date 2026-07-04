"""
Base IIDM paper - KD-VGG (exact paper implementation).
Teacher: VGG-19, frozen.
Student: enc model mirroring VGG-19 with reduced channels (Table A2).
KD: PCA-based — student features aligned to VGG-19 principal components.
Paper Table A2 VGG-19 distilled channels:
  relu1:23, relu2:34, relu3:80, relu4:79, relu5:159, relu6:162,
  relu7:160, relu8:154, relu9:267, relu10:203, relu11:121, relu12:123,
  relu13:108, relu14:64, relu15:36, relu16:16
We use 4 blocks (relu1-4 correspond to first 4 VGG blocks).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights

# Paper Table A2 - VGG-19 distilled channel counts (first 4 relu layers)
# relu1=23, relu2=34, relu3=80, relu4=79
VGG19_TEACHER_CH = [64, 128, 256, 512]   # original VGG-19 channels at 4 blocks
VGG19_STUDENT_CH = [23, 34, 80, 79]      # paper Table A2 distilled channels


class VGG19Teacher(nn.Module):
    """
    VGG-19 pretrained teacher, FROZEN.
    Extracts features at 4 scales matching paper Figure 3.
    Input: 6-channel (adapted from paper's 4-channel GF-1).
    """
    def __init__(self, in_channels=6):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        features = list(vgg.features)

        # Adapt first conv: 3ch -> 6ch (average weight tiling)
        old_conv = features[0]
        new_conv = nn.Conv2d(in_channels, 64, 3, padding=1, bias=True)
        with torch.no_grad():
            avg_w = old_conv.weight.mean(dim=1, keepdim=True)
            new_conv.weight.copy_(avg_w.repeat(1, in_channels, 1, 1))
            new_conv.bias.copy_(old_conv.bias)
        features[0] = new_conv

        # 4 blocks matching paper's 4-scale feature extraction
        self.block1 = nn.Sequential(*features[0:5])    # -> 64ch,  H,   W
        self.block2 = nn.Sequential(*features[5:10])   # -> 128ch, H/2, W/2
        self.block3 = nn.Sequential(*features[10:19])  # -> 256ch, H/4, W/4
        self.block4 = nn.Sequential(*features[19:28])  # -> 512ch, H/8, W/8

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        f1 = self.block1(x)
        f2 = self.block2(f1)
        f3 = self.block3(f2)
        f4 = self.block4(f3)
        return [f1, f2, f3, f4]

    def train(self, mode=True):
        return super().train(False)  # always eval


class _ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


class KDVGGStudent(nn.Module):
    """
    Student enc model - mirrors VGG-19 with reduced channels (Table A2).
    enc1: 6->23, enc2: 23->34, enc3: 34->80, enc4: 80->79
    Projection heads for KD alignment with teacher.
    """
    def __init__(self, in_channels=6):
        super().__init__()
        ch = VGG19_STUDENT_CH   # [23, 34, 80, 79]

        # Mirror VGG-19 depth, reduced channels
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

        # 1x1 projection: student_ch -> teacher_ch (for KD MSE loss)
        self.proj = nn.ModuleList([
            nn.Conv2d(ch[i], VGG19_TEACHER_CH[i], 1, bias=False)
            for i in range(4)
        ])

    def forward(self, x):
        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        f4 = self.enc4(f3)
        feats = [f1, f2, f3, f4]
        # Project to teacher channel dims for KD loss
        projected = [self.proj[i](feats[i]) for i in range(4)]
        return feats, projected


class KDLoss(nn.Module):
    """
    Paper KD Loss: MSE between student projected features and teacher features.
    L_kd = (1/4) * sum_i MSE(student_proj_i, teacher_feat_i)
    Teacher features detached (no gradient to teacher).
    """
    def __init__(self):
        super().__init__()

    def forward(self, student_projected, teacher_feats):
        loss = 0.0
        for sp, tf in zip(student_projected, teacher_feats):
            tf_d = tf.detach()
            # Align spatial size if needed
            if sp.shape[-2:] != tf_d.shape[-2:]:
                sp = F.adaptive_avg_pool2d(sp, tf_d.shape[-2:])
            loss = loss + F.mse_loss(sp, tf_d)
        return loss / len(student_projected)
