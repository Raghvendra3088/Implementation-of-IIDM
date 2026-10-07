import torch
import torch.nn as nn
import timm

class SwinEncoder(nn.Module):
    def __init__(self, in_chans=6, out_ch=256, pretrained=True):
        super().__init__()
        self.swin = timm.create_model(
            'swin_base_patch4_window7_224',
            pretrained=pretrained,
            in_chans=in_chans,
            img_size=256,
            num_classes=0,
            features_only=True,
            out_indices=(3,)
        )
        self.proj = nn.Sequential(
            nn.Conv2d(1024, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.GELU()
        )

    def forward(self, x):
        feats = self.swin(x)
        f = feats[0].permute(0, 3, 1, 2).contiguous()
        return [self.proj(f)]


class SwinEncoderWithSkip(nn.Module):
    def __init__(self, in_chans=6, pretrained=True):
        super().__init__()
        self.swin = timm.create_model(
            'swin_base_patch4_window7_224',
            pretrained=pretrained,
            in_chans=in_chans,
            img_size=256,
            num_classes=0,
            features_only=True,
            out_indices=(0, 1, 2, 3)
        )
        self.proj_last = nn.Sequential(
            nn.Conv2d(1024, 256, 1, bias=False),
            nn.GroupNorm(8, 256),
            nn.GELU()
        )

    def forward(self, x):
        feats = self.swin(x)
        feats = [f.permute(0, 3, 1, 2).contiguous() for f in feats]
        feats[-1] = self.proj_last(feats[-1])
        return feats
