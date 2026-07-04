"""
Paper Appendix B: decoder dec, mirrors encoder enc in reverse.
dec_N takes reluN_e features, reconstructs toward relu(N-1)_e, ..., down to image.
Blocks trained sequentially N=1..16 alongside matching encN (Eq. B.4-B.6).
"""
import torch
import torch.nn as nn
from src.models.vgg19_full import VGG19_STUDENT_CH_16, POOL_AFTER

# Which decoder blocks must upsample (mirrors POOL_AFTER, reversed)
# Fix: the 5th pool in POOL_AFTER (after relu16, i=15) never affects any
# captured feature in this 16-layer truncated encoder (no conv layer 17
# exists to consume it) -- verified against real VGG19 architecture via
# torchvision. Only 4 real spatial downsamplings occur across s_feats.
# Decoder must mirror only those 4, at the correct indices.
UPSAMPLE_BEFORE = {14 - i for i in POOL_AFTER if i != max(POOL_AFTER)}


class DecBlock(nn.Module):
    def __init__(self, in_ch, out_ch, upsample):
        super().__init__()
        self.upsample = upsample
        layers = []
        if upsample:
            layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        layers += [nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
        self.block = nn.Sequential(*layers)

    def forward(self, x): return self.block(x)


class VGGDecoder16(nn.Module):
    """dec16 -> dec15 -> ... -> dec1 -> image. Index i in self.decs corresponds to dec(16-i)."""
    def __init__(self, out_channels=4):
        super().__init__()
        ch = VGG19_STUDENT_CH_16          # [ch1..ch16], student channel sizes
        # decoder consumes reversed: dec16 input=ch16(=16), output ch15(=36), etc.
        rev_ch = list(reversed(ch))       # [ch16..ch1]
        self.decs = nn.ModuleList()
        for i in range(16):
            in_ch  = rev_ch[i]
            out_ch = rev_ch[i + 1] if i < 15 else out_channels
            up = (i in UPSAMPLE_BEFORE)
            self.decs.append(DecBlock(in_ch, out_ch, up))

    def forward_from(self, feat, start_N):
        """start_N: 1..16 (which reluN_e feature we're starting decode from).
        Returns dict {N-1: feature_at_that_stage, ..., 0: reconstructed_image}"""
        outputs = {}
        x = feat
        dec_idx_start = 16 - start_N     # dec16 is idx0, dec1 is idx15
        for idx in range(dec_idx_start, 16):
            x = self.decs[idx](x)
            level = 16 - (idx + 1)       # produces relu(level)_e ; level=0 -> image
            outputs[level] = x
        return outputs
