"""
Sequential Blockwise PCA Distillation for KD-UNet.
Matches Student UNet features to Teacher UNet features, stage by stage.
"""
import os, sys, glob
import torch
torch.backends.cudnn.enabled = False
import torch.nn.functional as F
torch.backends.cudnn.enabled = False
import numpy as np
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.vgg19_full import KDVGGStudent16, VGG19_STUDENT_CH_16
from src.models.base_kd_unet import TeacherUNet, BaseKDUNet
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
    p.add_argument('--epochs_per_block', type=int, default=15)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--T', type=int, default=1000)
    p.add_argument('--eigenbasis_ckpt', default='checkpoints/eigenbasis_unet.pth')
    p.add_argument('--teacher_ckpt', default='checkpoints/teacher_unet/teacher_best.pth')
    p.add_argument('--save_path', default='checkpoints/blockwise_kd_unet.pth')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    ds = PatchDataset(args.patch_dir, 'train')
    ld = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # 1. Load VGG
    vgg = KDVGGStudent16(in_channels=4).to(device)
    vgg.load_state_dict(torch.load('checkpoints/blockwise_kd.pth', map_location=device)['student'])
    vgg.eval()
    for pm in vgg.parameters(): pm.requires_grad = False

    # 2. Load Teacher
    COND_CHS = [VGG19_STUDENT_CH_16[i] for i in [1, 3, 7, 11, 15]]
    teacher = TeacherUNet(in_ch=5, cond_chs=COND_CHS).to(device)
    teacher.load_state_dict(torch.load(args.teacher_ckpt, map_location=device)['unet'])
    teacher.eval()
    for pm in teacher.parameters(): pm.requires_grad = False

    # 3. Load Eigenbasis
    eigenbasis = MultiLayerEigenbasisUNet().to(device)
    eigenbasis.load_state_dict(torch.load(args.eigenbasis_ckpt, map_location=device))
    eigenbasis.eval()
    for pm in eigenbasis.parameters(): pm.requires_grad = False

    # 4. Initialize Student
    student = BaseKDUNet(in_ch=5, cond_chs=COND_CHS).to(device)
    for pm in student.parameters(): pm.requires_grad = False
    
    # Enable time embeddings since they are used across all layers
    for pm in student.t_emb.parameters(): pm.requires_grad = True
    for pm in student.t_proj.parameters(): pm.requires_grad = True

    blocks = [
        [student.enc1, student.cross_attn[0]],
        [student.enc2, student.cross_attn[1]],
        [student.enc3, student.cross_attn[2]],
        [student.enc4, student.cross_attn[3]],
        [student.bot,  student.cross_attn[4]],
        [student.up4,  student.dec4],
        [student.up3,  student.dec3],
        [student.up2,  student.dec2],
        [student.up1,  student.dec1]
    ]

    betas, alpha_bar = make_schedule(args.T, device)

    for N in range(9):
        print(f"\n{'='*50}\nStage N={N+1}/9\n{'='*50}")
        
        # Unfreeze current block
        for mod in blocks[N]:
            for pm in mod.parameters():
                pm.requires_grad = True

        params = list(student.t_emb.parameters()) + list(student.t_proj.parameters())
        for mod in blocks[N]:
            params += list(mod.parameters())
            
        optimizer = torch.optim.Adam(params, lr=args.lr)

        for epoch in range(1, args.epochs_per_block + 1):
            total_loss = 0.0
            for x, y0 in ld:
                x, y0 = x.to(device), y0.to(device)
                B, _, H, W = y0.shape

                with torch.no_grad():
                    s_feats = vgg(x)
                    f_multi = [s_feats[i] for i in [1, 3, 7, 11, 15]]

                    t_idx = torch.randint(1, args.T + 1, (B,), device=device)
                    ab    = alpha_bar[t_idx - 1].view(B, 1, 1, 1)
                    eps   = torch.randn_like(y0)
                    y_t   = ab.sqrt() * y0 + (1 - ab).sqrt() * eps
                    unet_in = torch.cat([x, y_t], dim=1)

                    t_feats = teacher.forward_features(unet_in, t_idx, f_multi)

                # Student forward
                student.train()
                s_feats_all = student.forward_features(unet_in, t_idx, f_multi)
                fN_e = s_feats_all[N]

                # Eq B.4 PCA Loss + Mean Matching Fix
                B_s, C_s, H_s, W_s = fN_e.shape
                f_e_flat = fN_e.view(B_s, C_s, H_s*W_s)
                s_mean = f_e_flat.mean(dim=2, keepdim=True)
                f_bar_e = f_e_flat - s_mean
                
                WN = eigenbasis.bases[N].W
                proj_up = torch.einsum('ec,bes->bcs', WN, f_bar_e)
                
                f_t_flat = t_feats[N].view(B_s, -1, H_s*W_s)
                t_mean = f_t_flat.mean(dim=2, keepdim=True)
                f_bar_t = f_t_flat - t_mean
                
                loss_pca = F.mse_loss(proj_up, f_bar_t)
                
                # Critical Fix: Match spatial means
                s_mean_proj = torch.einsum('ec,bes->bcs', WN, s_mean)
                loss_mean = F.mse_loss(s_mean_proj, t_mean)
                
                loss = loss_pca + loss_mean

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg = total_loss / len(ld)
            print(f"  Stage {N+1}/9 Epoch {epoch:2d}/{args.epochs_per_block} | Loss: {avg:.6f}", flush=True)

        # Freeze current block after training
        for mod in blocks[N]:
            for pm in mod.parameters():
                pm.requires_grad = False

    # Save final distilled student
    torch.save({'unet': student.state_dict()}, args.save_path)
    print(f"\nBlockwise-distilled UNet student saved: {args.save_path}")

if __name__ == '__main__':
    main()
