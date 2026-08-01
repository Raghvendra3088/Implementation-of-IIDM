"""
Paper Appendix B: train global eigenbasis W_N,g for N=1..16, 200 epochs, batch=8.
"""
import os, sys, glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.vgg19_full import VGG19Teacher16, VGG19_TEACHER_CH_16, VGG19_STUDENT_CH_16
from src.models.eigenbasis16 import MultiLayerEigenbasis16


class ImageOnlyDataset(Dataset):
    def __init__(self, patch_dir):
        self.files = sorted(glob.glob(os.path.join(patch_dir, 'train', '*.npz')))
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        d = np.load(self.files[i])
        return torch.from_numpy(d['image'])


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--patch_dir', default='data/processed/patches_v2')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--save_path', default='checkpoints/eigenbasis16.pth')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    ds = ImageOnlyDataset(args.patch_dir)
    ld = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    print(f"Pretraining 16-layer eigenbasis on {len(ds)} images")

    teacher = VGG19Teacher16(in_channels=4).to(device).eval()
    eigenbasis = MultiLayerEigenbasis16(VGG19_TEACHER_CH_16, VGG19_STUDENT_CH_16).to(device)

    optimizer = torch.optim.Adam(eigenbasis.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for x in ld:
            x = x.to(device)
            with torch.no_grad():
                t_feats = teacher(x)

            _, _, recon_loss = eigenbasis(t_feats)

            optimizer.zero_grad()
            recon_loss.backward()
            optimizer.step()
            eigenbasis.orthonormalize()

            total_loss += recon_loss.item()

        avg = total_loss / len(ld)
        print(f"Epoch {epoch:3d}/{args.epochs} | Recon Loss: {avg:.6f}", flush=True)

    torch.save(eigenbasis.state_dict(), args.save_path)
    print(f"16-layer eigenbasis saved: {args.save_path}")


if __name__ == '__main__':
    main()
