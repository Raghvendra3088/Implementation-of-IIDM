import torch
from src.models.base_kd_unet import BaseKDUNet
m = BaseKDUNet(in_ch=5, cond_ch=16)
params = sum(p.numel() for p in m.parameters() if p.requires_grad)
print(f"BaseKDUNet parameters: {params:,}")
