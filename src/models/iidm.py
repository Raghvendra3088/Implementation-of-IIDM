import torch.nn as nn
from .kd_vgg import StudentVGG
from .kd_unet import KDUNet
from .diffusion import IIDM_Diffusion

class IIDM_Full(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = StudentVGG()
        self.unet = KDUNet()
        self.diffusion = IIDM_Diffusion(self.unet)
    
    def forward(self, optical, structural, carbon):
       
        pass
