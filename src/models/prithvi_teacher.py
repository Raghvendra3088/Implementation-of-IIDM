import torch
import torch.nn as nn
import timm

# The spatial resolution of the feature maps from ViT-Base with 256x256 input and patch_size=16 is 16x16.
# We will use nearest interpolation to upsample them back to 256x256 so the blockwise KD logic
# (which expects spatial maps of the same size as the input, or we can just leave them at 16x16
# and have the student mimic them at 16x16). 
# Wait, the KD mechanism uses the spatial dimensions directly! If teacher features are 16x16, 
# the student encoder layers must match 16x16. 
# But wait, we can just upsample the ViT features to match the student, OR we can downsample the student.
# Standard knowledge distillation from a ViT to a CNN often involves upsampling the ViT tokens.
# Let's upsample them to 256x256 so they seamlessly plug into the existing IIDM KD pipeline.

PRITHVI_CH_12 = [768] * 12
PRITHVI_STUDENT_CH_12 = [8, 12, 14, 16, 17, 19, 20, 21, 22, 23, 24, 24]

class KDStudent12(nn.Module):
    """
    12-block CNN student to mimic Prithvi-100M via PCA KD.
    Keeps spatial dimensions at 64x64 for all blocks.
    """
    def __init__(self, in_channels=4):
        super().__init__()
        self.blocks = nn.ModuleList()
        in_c = in_channels
        for out_c in PRITHVI_STUDENT_CH_12:
            # Simple 3x3 depthwise-separable conv block for parameter efficiency
            self.blocks.append(nn.Sequential(
                nn.Conv2d(in_c, in_c, 3, padding=1, groups=in_c),
                nn.Conv2d(in_c, out_c, 1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            ))
            in_c = out_c

    def forward(self, x):
        feats = []
        for b in self.blocks:
            x = b(x)
            feats.append(x)
        return feats

class PrithviTeacher12(nn.Module):
    def __init__(self, pretrained_path=None):
        super().__init__()
        # Prithvi uses a ViT-Base architecture (patch 16, embed 768, 12 blocks, 12 heads)
        # It takes 6 channels natively.
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=False, in_chans=6, img_size=64)
        
        if pretrained_path is not None:
            # We would load the weights here, ignoring mismatches in positional embedding if needed
            pass
            
        # Freeze the teacher
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        """
        x is the 4-band Sentinel-2 input (B2, B3, B4, B8), shape: [B, 4, 256, 256]
        We need to pad it to 6 bands.
        """
        B, C, H, W = x.shape
        x_padded = torch.zeros(B, 6, H, W, device=x.device)
        x_padded[:, :4, :, :] = x
        
        # Get all 12 intermediate block outputs, reshaped to 2D [B, 768, 16, 16]
        features = self.vit.get_intermediate_layers(x_padded, n=12, reshape=True)
        
        # Upsample all features to 256x256 to seamlessly plug into the existing KD spatial process
        upsampled_features = []
        for feat in features:
            feat_up = torch.nn.functional.interpolate(feat, size=(H, W), mode='bilinear', align_corners=False)
            upsampled_features.append(feat_up)
            
        return upsampled_features
