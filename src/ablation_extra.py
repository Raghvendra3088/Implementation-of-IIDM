"""
3 Additional Ablations for Base IIDM:
  --mode student_only      : VGG19 student features → direct carbon pred (no KD, no diff, no INR)
  --mode student_kd_inr    : Student + KD + INR (no diffusion)
  --mode student_kd_diff   : Student + KD + diffusion (no INR)

Run on base Implementation-of-IIDM — do NOT run in Physics-Informed-IIDM.
"""
import os, sys, json, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Dataset (same as train_base.py) ──────────────────────────────────────────
class PatchDataset(Dataset):
    def __init__(self, root, split):
        self.files = sorted([
            os.path.join(root, split, f)
            for f in os.listdir(os.path.join(root, split))
            if f.endswith('.npz')
        ])
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        d = np.load(self.files[i])
        x = torch.from_numpy(d['image']).float()            # (6, H, W)
        y = torch.from_numpy(d['carbon']).float().unsqueeze(0)  # (1, H, W)
        m = torch.from_numpy(d['mask']).float().unsqueeze(0)    # (1, H, W)
        return x, y, m

# ── Simple carbon head (for student_only and student_kd_inr) ─────────────────
class CarbonHead(nn.Module):
    """1x1 conv: projects student features → 1ch carbon map"""
    def __init__(self, in_ch):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, 64, 1), nn.ReLU(),
            nn.Conv2d(64, 1, 1), nn.Tanh()
        )
    def forward(self, f, H, W):
        f_up = F.interpolate(f, size=(H, W), mode='bilinear', align_corners=False)
        return self.head(f_up)

# ── Diffusion schedule ────────────────────────────────────────────────────────
def make_schedule(T, device):
    betas     = torch.linspace(1e-4, 0.02, T, device=device)
    alpha_bar = torch.cumprod(1 - betas, dim=0)
    return betas, alpha_bar

# ── RMSE util ─────────────────────────────────────────────────────────────────
def compute_rmse(pred, gt, mask, C_MIN, C_MAX):
    p = ((pred.clamp(-1,1) + 1)/2 * (C_MAX - C_MIN) + C_MIN)
    g = ((gt.clamp(-1,1)   + 1)/2 * (C_MAX - C_MIN) + C_MIN)
    err = ((p - g)**2 * mask).sum() / (mask.sum() + 1e-8)
    return err.sqrt().item()

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', required=True,
                   choices=['student_only', 'student_kd_inr', 'student_kd_diff'],
                   help='Which ablation to run')
    p.add_argument('--patch_dir',  default='data/processed/patches')
    p.add_argument('--epochs',     type=int,   default=60)
    p.add_argument('--batch_size', type=int,   default=4)
    p.add_argument('--lr',         type=float, default=2e-4)
    p.add_argument('--T',          type=int,   default=1000)
    p.add_argument('--save_dir',   default='checkpoints/')
    p.add_argument('--log_dir',    default='logs/')
    args = p.parse_args()

    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    C_MIN, C_MAX = 0.04, 207.97
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir,  exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  ABLATION: {args.mode}")
    print(f"{'='*55}")

    # Dataset
    train_ds = PatchDataset(args.patch_dir, 'train')
    val_ds   = PatchDataset(args.patch_dir, 'val')
    train_ld = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True,  num_workers=4, pin_memory=True)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch_size,
                          shuffle=False, num_workers=2, pin_memory=True)
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Models — always need student
    from src.models.vgg19_full   import KDVGGStudent16, VGG19_STUDENT_CH_16
    from src.models.base_kd_unet import BaseKDUNet
    from src.models.inr          import SIRENINR

    block_ckpt = 'checkpoints/blockwise_kd.pth'
    assert os.path.exists(block_ckpt), f"Missing: {block_ckpt}"

    student = KDVGGStudent16(in_channels=6).to(device)
    student.load_state_dict(torch.load(block_ckpt, map_location=device)['student'])
    COND_CH = VGG19_STUDENT_CH_16[-1]

    # Mode-specific components
    use_diff = args.mode == 'student_kd_diff'
    use_inr  = args.mode == 'student_kd_inr'
    use_head = args.mode in ['student_only', 'student_kd_inr']

    if use_diff:
        unet = BaseKDUNet(in_ch=COND_CH + 1, cond_ch=COND_CH).to(device)
        betas, alpha_bar = make_schedule(args.T, device)
        head = CarbonHead(COND_CH).to(device)   # final carbon from denoised
    elif use_inr:
        from src.models.inr import SIRENINR
        inr  = SIRENINR().to(device)
        head = CarbonHead(COND_CH).to(device)
    else:  # student_only
        head = CarbonHead(COND_CH).to(device)

    # Optimizer — student frozen after blockwise KD, only new components train
    if use_diff:
        params = list(unet.parameters()) + list(head.parameters())
    elif use_inr:
        params = list(inr.parameters())  + list(head.parameters())
    else:
        params = list(head.parameters())

    optimizer = torch.optim.Adam(params, lr=args.lr)

    best_rmse = float('inf')
    log = []

    for epoch in range(1, args.epochs + 1):
        student.eval()   # student frozen
        if use_diff: unet.train()
        if use_inr:  inr.train()
        head.train() if use_head else None

        train_loss = 0.0
        for x, y0, mask in train_ld:
            x, y0, mask = x.to(device), y0.to(device), mask.to(device)
            B, _, H, W  = y0.shape

            with torch.no_grad():
                s_feats = student(x)
                f0      = s_feats[-1]
                f0_up   = F.interpolate(f0, size=(H,W), mode='bilinear', align_corners=False)

            optimizer.zero_grad()

            if args.mode == 'student_only':
                pred = head(f0, H, W)
                loss = ((pred - y0).abs() * mask).sum() / (mask.sum() + 1e-8)

            elif args.mode == 'student_kd_inr':
                pred = inr([s_feats[-1]], H=H, W=W)
                loss = ((pred - y0).abs() * mask).sum() / (mask.sum() + 1e-8)

            elif args.mode == 'student_kd_diff':
                t_idx = torch.randint(1, args.T+1, (B,), device=device)
                ab    = alpha_bar[t_idx-1].view(B,1,1,1)
                eps   = torch.randn_like(y0)
                y_t   = ab.sqrt() * y0 + (1-ab).sqrt() * eps
                unet_in  = torch.cat([f0_up, y_t], dim=1)
                eps_pred = unet(unet_in, t_idx, f0_up)
                diff = (eps_pred - eps).abs() * mask
                loss = diff.sum() / (mask.sum() + 1e-8)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_ld)

        # Val RMSE
        val_rmses = []
        with torch.no_grad():
            for x, y0, mask in val_ld:
                x, y0, mask = x.to(device), y0.to(device), mask.to(device)
                B, _, H, W  = y0.shape
                s_feats = student(x)
                f0      = s_feats[-1]

                if args.mode == 'student_only':
                    pred = head(f0, H, W)
                elif args.mode == 'student_kd_inr':
                    pred = inr([s_feats[-1]], H=H, W=W)
                elif args.mode == 'student_kd_diff':
                    # Single step denoise at t=T//2
                    t_mid = torch.full((B,), args.T//2, device=device)
                    ab_m  = alpha_bar[t_mid-1].view(B,1,1,1)
                    eps   = torch.randn_like(y0)
                    y_t   = ab_m.sqrt() * y0 + (1-ab_m).sqrt() * eps
                    f0_up = F.interpolate(f0, size=(H,W), mode='bilinear', align_corners=False)
                    unet_in  = torch.cat([f0_up, y_t], dim=1)
                    eps_pred = unet(unet_in, t_mid, f0_up)
                    pred = (y_t - (1-ab_m).sqrt()*eps_pred) / (ab_m.sqrt() + 1e-8)

                val_rmses.append(compute_rmse(pred, y0, mask, C_MIN, C_MAX))

        val_rmse = float(np.mean(val_rmses))
        print(f"Epoch {epoch:3d}/{args.epochs} | Loss: {train_loss:.4f} | Val RMSE: {val_rmse:.4f}")

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            ckpt_path = os.path.join(args.save_dir, f'ablation_{args.mode}_best.pth')
            torch.save({'epoch': epoch, 'val_rmse': best_rmse,
                        'student': student.state_dict(),
                        'head': head.state_dict() if use_head else None,
                        'unet': unet.state_dict() if use_diff else None,
                        'inr':  inr.state_dict()  if use_inr  else None},
                       ckpt_path)
            print(f"  ✓ Saved best: {best_rmse:.4f}")

        log.append({'epoch': epoch, 'train_loss': train_loss, 'val_rmse': val_rmse})

    # Save log
    log_path = os.path.join(args.log_dir, f'ablation_{args.mode}.json')
    with open(log_path, 'w') as f:
        json.dump({'mode': args.mode, 'best_val_rmse': best_rmse, 'log': log}, f, indent=2)
    print(f"\nBest Val RMSE: {best_rmse:.4f} | Log: {log_path}")

if __name__ == '__main__':
    main()
