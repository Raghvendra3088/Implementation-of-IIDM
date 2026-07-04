"""
Read-only diagnostic for Stage 3 (blockwise KD) output quality.
No training, no methodology changes -- pure verification.

Checks:
  1. Eq B.4 encoder projection error, per layer N=1..16, on held-out data.
  2. Full image reconstruction quality: decode student's deepest feature
     (s_feats[-1], the SAME feature used as diffusion conditioning in
     Stage 5/6) all the way through the trained decoder back to the image,
     compare against ground truth input.

Loads: checkpoints/blockwise_kd.pth (student + decoder)
       checkpoints/eigenbasis16.pth (eigenbasis, for Eq B.4 check only)
Evaluates on: data/processed/patches_v2/<split>
"""
import os, sys, glob
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.vgg19_full import VGG19Teacher16, KDVGGStudent16, VGG19_STUDENT_CH_16, VGG19_TEACHER_CH_16
from src.models.vgg19_decoder import VGGDecoder16
from src.models.eigenbasis16 import MultiLayerEigenbasis16


class ImageOnlyDataset(Dataset):
    def __init__(self, patch_dir, split):
        self.files = sorted(glob.glob(os.path.join(patch_dir, split, '*.npz')))
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .npz files in {patch_dir}/{split}")
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        d = np.load(self.files[i])
        return torch.from_numpy(d['image'])


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--patch_dir', default='data/processed/patches_v2')
    p.add_argument('--split', default='test')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--n_batches', type=int, default=20,
                    help='limit batches for speed; set 0 for full split')
    p.add_argument('--blockwise_ckpt', default='checkpoints/blockwise_kd.pth')
    p.add_argument('--eigenbasis_ckpt', default='checkpoints/eigenbasis16.pth')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ds = ImageOnlyDataset(args.patch_dir, args.split)
    ld = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"{args.split}: {len(ds)} images")

    teacher = VGG19Teacher16(in_channels=4).to(device).eval()
    student = KDVGGStudent16(in_channels=4).to(device).eval()
    decoder = VGGDecoder16(out_channels=4).to(device).eval()
    eigenbasis = MultiLayerEigenbasis16(VGG19_TEACHER_CH_16, VGG19_STUDENT_CH_16).to(device).eval()

    ck = torch.load(args.blockwise_ckpt, map_location=device)
    student.load_state_dict(ck['student'])
    decoder.load_state_dict(ck['decoder'])
    eigenbasis.load_state_dict(torch.load(args.eigenbasis_ckpt, map_location=device))
    print(f"Loaded: {args.blockwise_ckpt}, {args.eigenbasis_ckpt}")

    enc_err_sum = np.zeros(16)
    enc_err_n   = 0
    img_rmse_sum = 0.0
    img_n = 0

    with torch.no_grad():
        for bi, x in enumerate(ld):
            if args.n_batches and bi >= args.n_batches:
                break
            x = x.to(device)
            B = x.shape[0]

            t_feats = teacher(x)
            s_feats = student(x)

            # --- Eq B.4 encoder projection error, per layer ---
            for N in range(1, 17):
                fN_e = s_feats[N-1]
                Bc, C, H, W = fN_e.shape
                f_bar_e = fN_e.view(Bc, C, H*W) - fN_e.view(Bc, C, H*W).mean(dim=2, keepdim=True)
                WN = eigenbasis.bases[N-1].W
                proj_up = torch.einsum('ec,bes->bcs', WN, f_bar_e)
                f_bar_teacher = t_feats[N-1].view(Bc, -1, H*W)
                f_bar_teacher = f_bar_teacher - f_bar_teacher.mean(dim=2, keepdim=True)
                err = F.mse_loss(proj_up, f_bar_teacher).item()
                enc_err_sum[N-1] += err
            enc_err_n += 1

            # --- Full image reconstruction from deepest student feature ---
            f16 = s_feats[-1]                      # same feature used as diffusion conditioning
            dec_out = decoder.forward_from(f16, 16)  # decode all the way to image
            img_rec = dec_out[0]
            rmse = torch.sqrt(F.mse_loss(img_rec, x)).item()
            img_rmse_sum += rmse * B
            img_n += B

            if (bi + 1) % 5 == 0:
                print(f"  batch {bi+1}", flush=True)

    print("\n" + "=" * 60)
    print("Eq B.4 encoder projection MSE per layer (lower = better fidelity)")
    print("=" * 60)
    for N in range(1, 17):
        print(f"  N={N:2d}: {enc_err_sum[N-1] / enc_err_n:.6f}")

    print("\n" + "=" * 60)
    print(f"Full image reconstruction (student f16 -> decoder -> image)")
    print("=" * 60)
    print(f"Mean pixel RMSE (normalized space): {img_rmse_sum / img_n:.6f}")
    print("(Compare: if this is large relative to input's own std (~1.0 in")
    print(" normalized space), the deepest student feature f16 is NOT carrying")
    print(" enough information to reconstruct the image -- meaning it is a")
    print(" weak signal for anything downstream, including diffusion conditioning.)")


if __name__ == '__main__':
    main()
