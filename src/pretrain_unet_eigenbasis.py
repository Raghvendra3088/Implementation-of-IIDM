"""
Pre-train global eigenbasis W_N,g for UNet blocks, 200 epochs.
Extracts 9 feature maps from Teacher UNet.
"""
import os, sys, glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.vgg19_full import KDVGGStudent16
from src.models.base_kd_unet import TeacherUNet
from src.models.eigenbasis_unet import MultiLayerEigenbasisUNet

class PatchDataset(Dataset):
    def __init__(self, root, split):
        split_dir = os.path.join(root, split)
        self.files = sorted(glob.glob(os.path.join(split_dir, '*.npz')))
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        d = np.load(self.files[i])
        x = torch.from_numpy(d['image'])
        y = torch.from_numpy(d['carbon']).unsqueeze(0)
        return x, y

def make_schedule(T=1000, device='cpu'):
    betas     = torch.linspace(1e-4, 0.02, T, device=device)
    alpha_bar = torch.cumprod(1.0 - betas, dim=0)
    return betas, alpha_bar

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--patch_dir', default='data/processed/patches_v2')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--T', type=int, default=1000)
    p.add_argument('--save_path', default='checkpoints/eigenbasis_unet.pth')
    p.add_argument('--teacher_ckpt', default='checkpoints/teacher_unet/teacher_best.pth')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    ds = PatchDataset(args.patch_dir, 'train')
    ld = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Load Student VGG for f0 conditioning
    student = KDVGGStudent16(in_channels=4).to(device)
    student.load_state_dict(torch.load('checkpoints/blockwise_kd.pth', map_location=device)['student'])
    student.eval()

    # Load Teacher UNet
    teacher = TeacherUNet(in_ch=5, cond_ch=16).to(device)
    teacher.load_state_dict(torch.load(args.teacher_ckpt, map_location=device)['unet'])
    teacher.eval()

    eigenbasis = MultiLayerEigenbasisUNet().to(device)
    optimizer = torch.optim.Adam(eigenbasis.parameters(), lr=args.lr)
    
    betas, alpha_bar = make_schedule(args.T, device)

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for x, y0 in ld:
            x, y0 = x.to(device), y0.to(device)
            B, _, H, W = y0.shape

            with torch.no_grad():
                s_feats = student(x)
                f0 = s_feats[-1]
                f0_up = F.interpolate(f0, size=(H, W), mode='bilinear', align_corners=False)

                t_idx = torch.randint(1, args.T + 1, (B,), device=device)
                ab    = alpha_bar[t_idx - 1].view(B, 1, 1, 1)
                eps   = torch.randn_like(y0)
                y_t   = ab.sqrt() * y0 + (1 - ab).sqrt() * eps
                unet_in = torch.cat([x, y_t], dim=1)

                t_feats = teacher.forward_features(unet_in, t_idx, f0_up)

            _, _, recon_loss = eigenbasis(t_feats)

            optimizer.zero_grad()
            recon_loss.backward()
            optimizer.step()
            eigenbasis.orthonormalize()

            total_loss += recon_loss.item()

        avg = total_loss / len(ld)
        print(f"Epoch {epoch:3d}/{args.epochs} | Recon Loss: {avg:.6f}", flush=True)

    torch.save(eigenbasis.state_dict(), args.save_path)
    print(f"UNet eigenbasis saved: {args.save_path}")

if __name__ == '__main__':
    main()
