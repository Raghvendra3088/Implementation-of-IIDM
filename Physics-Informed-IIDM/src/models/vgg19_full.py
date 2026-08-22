"""
Paper-exact full VGG-19: 16 conv+relu layers.
Teacher channels (VGG-19 col, Table A2): standard VGG19 conv channels.
Student channels (mCEV col, Table A2): PCA-reduced, >85% variance retained.
"""
import torch
import torch.nn as nn
import torchvision.models as tvm

# Teacher: standard VGG-19 conv channel counts, 16 layers
VGG19_TEACHER_CH_16 = [64, 64, 128, 128, 256, 256, 256, 256,
                        512, 512, 512, 512, 512, 512, 512, 512]

# Student: Table A2, VGG-19 column, mCEV row (paper exact)
VGG19_STUDENT_CH_16 = [23, 34, 80, 79, 159, 162, 160, 154,
                        267, 203, 121, 123, 108, 64, 36, 16]

# Which layers are followed by 2x2 maxpool (after block ends): standard VGG19
POOL_AFTER = {1, 3, 7, 11, 15}   # 0-indexed layer positions (after relu2,4,8,12,16)


class VGG19Teacher16(nn.Module):
    """Full 16-layer VGG-19, ENC in paper. Uses ImageNet-pretrained weights
    if available (first conv adapted for in_channels), else random init."""
    def __init__(self, in_channels=4):
        super().__init__()
        try:
            backbone = tvm.vgg19(weights=tvm.VGG19_Weights.IMAGENET1K_V1).features
            pretrained = True
        except Exception:
            backbone = tvm.vgg19(weights=None).features
            pretrained = False

        convs = [m for m in backbone if isinstance(m, nn.Conv2d)]
        assert len(convs) == 16, f"Expected 16 convs, got {len(convs)}"

        self.layers = nn.ModuleList()
        for i, conv in enumerate(convs):
            new_conv = nn.Conv2d(conv.in_channels if i > 0 else in_channels,
                                  conv.out_channels, 3, padding=1, bias=True)
            if i == 0 and pretrained:
                with torch.no_grad():
                    avg_w = conv.weight.mean(dim=1, keepdim=True)
                    new_conv.weight.copy_(avg_w.repeat(1, in_channels, 1, 1))
                    new_conv.bias.copy_(conv.bias)
            elif pretrained:
                with torch.no_grad():
                    new_conv.weight.copy_(conv.weight)
                    new_conv.bias.copy_(conv.bias)
            self.layers.append(new_conv)

        self.pool = nn.MaxPool2d(2, 2)
        for p in self.parameters():
            p.requires_grad = False   # teacher frozen, paper: ENC is source, not trained

    def forward(self, x):
        feats = []
        for i, conv in enumerate(self.layers):
            x = torch.relu(conv(x))
            feats.append(x)                       # reluN feature, N=1..16
            if i in POOL_AFTER:
                x = self.pool(x)
        return feats                              # list of 16 tensors


class _ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class KDVGGStudent16(nn.Module):
    """enc in paper: 16 conv blocks, reduced channels (Table A2 mCEV)."""
    def __init__(self, in_channels=4):
        super().__init__()
        ch = VGG19_STUDENT_CH_16
        self.layers = nn.ModuleList()
        prev = in_channels
        for i, c in enumerate(ch):
            self.layers.append(_ConvBnRelu(prev, c))
            prev = c
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        feats = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            feats.append(x)                       # reluN_e feature, N=1..16
            if i in POOL_AFTER:
                x = self.pool(x)
        return feats                               # list of 16 tensors
