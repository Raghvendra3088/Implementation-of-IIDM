import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBaseline(nn.Module):
    """
    Standard U-Net dense regression baseline (Model A)
    Input: (B, 6, 256, 256)
    Output: (B, 1, 256, 256)
    """
    def __init__(self, in_channels=6, out_channels=1, base_ch=64):
        super().__init__()
        
        def double_conv(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )
            
        self.down1 = double_conv(in_channels, base_ch)
        self.down2 = double_conv(base_ch, base_ch*2)
        self.down3 = double_conv(base_ch*2, base_ch*4)
        self.down4 = double_conv(base_ch*4, base_ch*8)
        
        self.bot = double_conv(base_ch*8, base_ch*16)
        
        self.up4 = nn.ConvTranspose2d(base_ch*16, base_ch*8, 2, stride=2)
        self.conv4 = double_conv(base_ch*16, base_ch*8)
        
        self.up3 = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, stride=2)
        self.conv3 = double_conv(base_ch*8, base_ch*4)
        
        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)
        self.conv2 = double_conv(base_ch*4, base_ch*2)
        
        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2)
        self.conv1 = double_conv(base_ch*2, base_ch)
        
        self.out = nn.Conv2d(base_ch, out_channels, 1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = F.max_pool2d(d1, 2)
        
        d2 = self.down2(p1)
        p2 = F.max_pool2d(d2, 2)
        
        d3 = self.down3(p2)
        p3 = F.max_pool2d(d3, 2)
        
        d4 = self.down4(p3)
        p4 = F.max_pool2d(d4, 2)
        
        b = self.bot(p4)
        
        u4 = self.up4(b)
        u4 = torch.cat([u4, d4], dim=1)
        c4 = self.conv4(u4)
        
        u3 = self.up3(c4)
        u3 = torch.cat([u3, d3], dim=1)
        c3 = self.conv3(u3)
        
        u2 = self.up2(c3)
        u2 = torch.cat([u2, d2], dim=1)
        c2 = self.conv2(u2)
        
        u1 = self.up1(c2)
        u1 = torch.cat([u1, d1], dim=1)
        c1 = self.conv1(u1)
        
        out = self.out(c1)
        return torch.sigmoid(out) # Normalised output is in [0, 1]
