"""
Paper Appendix B: sequential blockwise PCA-KD training, N=1..16.
Eq B.4: L_enc^N = ||W_N^T F_bar_e_N - F_bar_N||^2
Eq B.5: L_dec^N = ||F_d_{N-1} - F_e_{N-1}||^2 + ||I_rec - I||^2 + ||F_{N,rec} - F_N||^2
Eq B.6: min over (encN, decN) of L_enc^N + L_dec^N, others frozen.
"""
import os, sys, glob
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.vgg19_full import VGG19Teacher16, KDVGGStudent16, VGG19_STUDENT_CH_16
from src.models.vgg19_decoder import VGGDecoder16
from src.models.eigenbasis16 import MultiLayerEigenbasis16, GlobalEigenbasis


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
    p.add_argument('--epochs_per_block', type=int, default=15)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--eigenbasis_ckpt', default='checkpoints/eigenbasis16.pth')
    p.add_argument('--save_path', default='checkpoints/blockwise_kd.pth')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    ds = ImageOnlyDataset(args.patch_dir)
    ld = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    print(f"Blockwise KD training on {len(ds)} images, 16 sequential stages")

    teacher    = VGG19Teacher16(in_channels=4).to(device).eval()
    student    = KDVGGStudent16(in_channels=4).to(device)
    decoder    = VGGDecoder16(out_channels=4).to(device)
    eigenbasis = MultiLayerEigenbasis16([64,64,128,128,256,256,256,256,512,512,512,512,512,512,512,512],
                                        VGG19_STUDENT_CH_16).to(device)
    eigenbasis.load_state_dict(torch.load(args.eigenbasis_ckpt, map_location=device))
    eigenbasis.eval()
    for pm in eigenbasis.parameters(): pm.requires_grad = False

    # Freeze everything initially; unfreeze per-block during its stage
    for pm in student.parameters(): pm.requires_grad = False
    for pm in decoder.parameters(): pm.requires_grad = False

    for N in range(1, 17):   # paper: sequential N=1..16
        print(f"\n{'='*50}\nStage N={N}/16\n{'='*50}")

        # Unfreeze only encN (student.layers[N-1]) and decN (decoder.decs[16-N])
        for pm in student.layers[N-1].parameters(): pm.requires_grad = True
        for pm in decoder.decs[16-N].parameters():  pm.requires_grad = True

        params = (list(student.layers[N-1].parameters()) +
                  list(decoder.decs[16-N].parameters()))
        optimizer = torch.optim.Adam(params, lr=args.lr)

        for epoch in range(1, args.epochs_per_block + 1):
            total_loss = 0.0
            for x in ld:
                x = x.to(device)

                with torch.no_grad():
                    t_feats = teacher(x)                    # 16 teacher features
                    t_proj, t_mean, _ = eigenbasis(t_feats)  # W_N F_bar_N per layer

                s_feats = student(x)                         # 16 student features (grad only on encN)
                fN_e = s_feats[N-1]                           # F^e_N,k
                B, C, H, W = fN_e.shape
                f_bar_e = fN_e.view(B, C, H*W) - fN_e.view(B, C, H*W).mean(dim=2, keepdim=True)

                # Eq B.4: encoder distillation loss
                WN = eigenbasis.bases[N-1].W                  # (C_N_e, C_N)
                proj_up = torch.einsum('ec,bes->bcs', WN, f_bar_e)  # W_N^T F_bar_e_N  (B, C_N, HW)
                f_bar_teacher = t_feats[N-1].view(B, -1, H*W)
                f_bar_teacher = f_bar_teacher - f_bar_teacher.mean(dim=2, keepdim=True)
                loss_enc = F.mse_loss(proj_up, f_bar_teacher)

                # Decoder: reconstruct from fN_e down to relu(N-1)_e / image
                dec_out = decoder.forward_from(fN_e, N)

                loss_dec = torch.tensor(0.0, device=device)
                # Term 1: F^d_{N-1} vs F^e_{N-1} (feature reconstruction), skip if N==1 (paper note)
                if N > 1 and (N-2) in dec_out:
                    target_feat = s_feats[N-2].detach()
                    if dec_out[N-1].shape == target_feat.shape:
                        loss_dec = loss_dec + F.mse_loss(dec_out[N-1], target_feat)

                # Term 2: image reconstruction (only meaningful once decoder reaches level 0)
                if 0 in dec_out:
                    img_rec = dec_out[0]
                    if img_rec.shape == x.shape:
                        loss_dec = loss_dec + F.mse_loss(img_rec, x)
                        # Term 3: perceptual round-trip — re-encode reconstructed image
                        with torch.no_grad():
                            t_feats_rec = teacher(img_rec.detach())
                        loss_dec = loss_dec + F.mse_loss(t_feats_rec[N-1], t_feats[N-1])

                loss = loss_enc + loss_dec

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg = total_loss / len(ld)
            print(f"  N={N} Epoch {epoch:2d}/{args.epochs_per_block} | Loss: {avg:.6f}", flush=True)

        # Re-freeze this block before moving to next N
        for pm in student.layers[N-1].parameters(): pm.requires_grad = False
        for pm in decoder.decs[16-N].parameters():  pm.requires_grad = False

    torch.save({'student': student.state_dict(), 'decoder': decoder.state_dict()}, args.save_path)
    print(f"\nBlockwise-distilled student saved: {args.save_path}")


if __name__ == '__main__':
    main()
