"""
Train Teacher UNet (Full Channels) for PCA Blockwise Distillation
"""
import os, sys, json
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class PatchDataset(Dataset):
    def __init__(self, root, split):
        import glob
        split_dir = os.path.join(root, split)
        self.files = sorted(glob.glob(os.path.join(split_dir, '*.npz')))
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .npz files in {split_dir}")
        print(f"  {split}: {len(self.files)} patches")

    def __len__(self): return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i])
        x = torch.from_numpy(d['image'])
        y = torch.from_numpy(d['carbon']).unsqueeze(0)
        m = torch.from_numpy(d['mask']).unsqueeze(0)
        return x, y, m

def make_schedule(T=1000, device='cpu'):
    betas     = torch.linspace(1e-4, 0.02, T, device=device)
    alpha_bar = torch.cumprod(1.0 - betas, dim=0)
    return betas, alpha_bar

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--patch_dir',  default='data/processed/patches_v2')
    p.add_argument('--epochs',     type=int,   default=100)
    p.add_argument('--batch_size', type=int,   default=4)
    p.add_argument('--lr',         type=float, default=2e-4)
    p.add_argument('--T',          type=int,   default=1000)
    p.add_argument('--save_dir',   default='checkpoints/teacher_unet')
    p.add_argument('--log_path',   default='logs/teacher_unet_train.log')
    p.add_argument('--resume',     action='store_true')
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_ds = PatchDataset(args.patch_dir, 'train')
    val_ds   = PatchDataset(args.patch_dir, 'val')
    train_ld = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True,  num_workers=4, pin_memory=True)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch_size,
                          shuffle=False, num_workers=2, pin_memory=True)

    from src.models.vgg19_full   import KDVGGStudent16, VGG19_STUDENT_CH_16
    from src.models.base_kd_unet import TeacherUNet

    student = KDVGGStudent16(in_channels=4).to(device)
    block_ckpt = 'checkpoints/blockwise_kd.pth'
    student.load_state_dict(torch.load(block_ckpt, map_location=device)['student'])
    student.eval()
    for pm in student.parameters(): pm.requires_grad = False

    COND_CHS = [VGG19_STUDENT_CH_16[i] for i in [1, 3, 7, 11, 15]]
    unet = TeacherUNet(in_ch=5, cond_chs=COND_CHS).to(device)

    optimizer = torch.optim.Adam(unet.parameters(), lr=args.lr)
    betas, alpha_bar = make_schedule(args.T, device)

    C_MIN = 4.816495895385742
    C_MAX = 129.18380737304688

    best_rmse   = float('inf')
    start_epoch = 1
    log         = []

    resume_ckpt = os.path.join(args.save_dir, 'teacher_resume.pth')
    if args.resume and os.path.exists(resume_ckpt):
        ckpt = torch.load(resume_ckpt, map_location=device)
        unet.load_state_dict(ckpt['unet'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_rmse   = ckpt['best_rmse']
        log         = ckpt.get('log', [])
        print(f"Resumed epoch {ckpt['epoch']}, best RMSE={best_rmse:.4f}")

    for epoch in range(start_epoch, args.epochs + 1):
        unet.train()
        train_loss = 0.0

        for x, y0, mask in train_ld:
            x, y0, mask = x.to(device), y0.to(device), mask.to(device)
            B, _, H, W = y0.shape

            with torch.no_grad():
                s_feats = student(x)
                f_multi = [s_feats[i] for i in [1, 3, 7, 11, 15]]

            t_idx = torch.randint(1, args.T + 1, (B,), device=device)
            ab    = alpha_bar[t_idx - 1].view(B, 1, 1, 1)
            eps   = torch.randn_like(y0)
            y_t   = ab.sqrt() * y0 + (1 - ab).sqrt() * eps

            unet_in  = torch.cat([x, y_t], dim=1)
            eps_pred = unet(unet_in, t_idx, f_multi)

            diff = (eps_pred - eps).abs() * mask
            loss = diff.sum() / (mask.sum() + 1e-8)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_ld)

        unet.eval()
        all_gt, all_pred = [], []
        t_val = args.T // 2

        with torch.no_grad():
            for x, y0, mask in val_ld:
                x, y0, mask = x.to(device), y0.to(device), mask.to(device)
                B, _, H, W = y0.shape
                with torch.no_grad():
                    s_feats = student(x)
                    f_multi = [s_feats[i] for i in [1, 3, 7, 11, 15]]
                
                torch.manual_seed(42)
                eps_v = torch.randn_like(y0)
                ab_v  = alpha_bar[t_val - 1]
                y_t_v = ab_v.sqrt() * y0 + (1 - ab_v).sqrt() * eps_v
                
                t_tensor = torch.full((B,), t_val, device=device, dtype=torch.long)
                unet_in  = torch.cat([x, y_t_v], dim=1)
                eps_pred = unet(unet_in, t_tensor, f_multi)

                y0_pred = (y_t_v - (1 - ab_v).sqrt() * eps_pred) / (ab_v.sqrt() + 1e-8)
                y0_pred = y0_pred.clamp(-1, 1)

                pred_mg = (y0_pred.cpu().numpy() * 0.5 + 0.5) * (C_MAX - C_MIN) + C_MIN
                gt_mg   = (y0.cpu().numpy() * 0.5 + 0.5) * (C_MAX - C_MIN) + C_MIN
                m_np    = mask.cpu().numpy().astype(bool)

                all_pred.append(pred_mg[m_np])
                all_gt.append(gt_mg[m_np])

        all_pred = np.concatenate(all_pred)
        all_gt   = np.concatenate(all_gt)
        rmse = float(np.sqrt(np.mean((all_pred - all_gt)**2)))

        print(f"Epoch {epoch:3d}/{args.epochs} | Loss: {train_loss:.4f} | RMSE: {rmse:.4f}", flush=True)

        if rmse < best_rmse:
            best_rmse = rmse
            torch.save({
                'epoch': epoch, 'rmse': rmse,
                'unet': unet.state_dict(),
                }, os.path.join(args.save_dir, 'teacher_best.pth'))
            print(f"  ✓ Best saved (RMSE={rmse:.4f})", flush=True)

        torch.save({
            'epoch': epoch, 'best_rmse': best_rmse,
            'unet': unet.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, resume_ckpt)

if __name__ == '__main__':
    main()
