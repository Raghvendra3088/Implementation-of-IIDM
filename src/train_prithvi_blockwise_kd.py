import os, sys, glob
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.prithvi_teacher import PrithviTeacher12, KDStudent12, PRITHVI_CH_12, PRITHVI_STUDENT_CH_12
from src.models.eigenbasis12 import MultiLayerEigenbasis12

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
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--save_path', default='checkpoints/prithvi_blockwise_kd.pth')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = False # Bypass cuDNN error on this server
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    ds = ImageOnlyDataset(args.patch_dir)
    ld = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    print(f"Blockwise KD training on {len(ds)} images, 12 sequential stages")

    teacher    = PrithviTeacher12().to(device).eval()
    student    = KDStudent12(in_channels=4).to(device)
    
    # We use a completely new eigenbasis that is learned simultaneously during KD
    # The original paper pre-trains it, but we can co-adapt it.
    eigenbasis = MultiLayerEigenbasis12(PRITHVI_CH_12, PRITHVI_STUDENT_CH_12).to(device)

    # Freeze everything initially
    for pm in student.parameters(): pm.requires_grad = False

    for N in range(1, 13):   # 12 blocks for Prithvi
        print(f"\n{'='*50}\nStage N={N}/12\n{'='*50}")

        # Unfreeze encN and the corresponding eigenbasis W_N
        for pm in student.blocks[N-1].parameters(): pm.requires_grad = True
        for pm in eigenbasis.bases[N-1].parameters(): pm.requires_grad = True

        params = (list(student.blocks[N-1].parameters()) +
                  list(eigenbasis.bases[N-1].parameters()))
        optimizer = torch.optim.Adam(params, lr=args.lr)

        for epoch in range(1, args.epochs_per_block + 1):
            total_loss = 0.0
            for x in ld:
                x = x.to(device)

                with torch.no_grad():
                    t_feats = teacher(x)  # 12 teacher features [B, 768, 64, 64]
                
                s_feats = student(x)      # 12 student features [B, C_student, 64, 64]
                fN_e = s_feats[N-1]
                B, C_e, H, W = fN_e.shape
                
                # Mean-center student features
                f_bar_e = fN_e.view(B, C_e, H*W) - fN_e.view(B, C_e, H*W).mean(dim=2, keepdim=True)

                # W_N^T * F_bar_e
                WN = eigenbasis.bases[N-1].W                  # (C_e, 768)
                proj_up = torch.einsum('ec,bes->bcs', WN, f_bar_e)  # (B, 768, HW)
                
                # Mean-center teacher features
                f_bar_teacher = t_feats[N-1].view(B, -1, H*W)
                f_bar_teacher = f_bar_teacher - f_bar_teacher.mean(dim=2, keepdim=True)
                
                # Variance matching
                loss_var = F.mse_loss(proj_up, f_bar_teacher)
                
                # Mean matching
                f_mean_e = fN_e.view(B, C_e, H*W).mean(dim=2)  # (B, C_e)
                t_mean   = t_feats[N-1].view(B, -1, H*W).mean(dim=2) # (B, 768)
                f_mean_proj = torch.einsum('ec,be->bc', WN, f_mean_e)
                loss_mean = F.mse_loss(f_mean_proj, t_mean)
                
                loss_enc = loss_var + loss_mean

                optimizer.zero_grad()
                loss_enc.backward()
                optimizer.step()
                eigenbasis.bases[N-1].orthonormalize()
                
                total_loss += loss_enc.item()

            avg = total_loss / len(ld)
            print(f"  N={N} Epoch {epoch:2d}/{args.epochs_per_block} | Loss: {avg:.6f}", flush=True)

        # Re-freeze this block
        for pm in student.blocks[N-1].parameters(): pm.requires_grad = False
        for pm in eigenbasis.bases[N-1].parameters(): pm.requires_grad = False

    torch.save({'student': student.state_dict()}, args.save_path)
    print(f"\nBlockwise-distilled student saved: {args.save_path}")

if __name__ == '__main__':
    main()
