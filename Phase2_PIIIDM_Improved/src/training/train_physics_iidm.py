"""
Base IIDM paper training script (exact methodology).
Paper Eq.4: L1 noise prediction loss.
Training eval: direct y0_pred from single noisy step (t=T//2).
Final test eval: full DDIM 20 steps (paper Table A3).
"""
import os, sys, json
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, '/DATA/raghvendra3088/Physics-Informed-IIDM')
from src.models.physics_loss import PhysicsInformedLoss


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


@torch.no_grad()
def ddim_sample(unet, f0_up, alpha_bar, T, device, n_steps=20, seed=42):
    """
    Paper Table A3: DDIM inference, n_steps=20.
    Used for final test evaluation only.
    """
    B, _, H, W = f0_up.shape
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    y_t = torch.randn(B, 1, H, W, device=device, generator=gen)

    step_size = T // n_steps
    timesteps = list(range(T, 0, -step_size))
    if timesteps[-1] != 1:
        timesteps.append(1)

    for i, t in enumerate(timesteps):
        t_next = timesteps[i + 1] if i + 1 < len(timesteps) else 0

        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
        ab_t   = alpha_bar[t - 1]
        ab_tm1 = alpha_bar[t_next - 1] if t_next > 0 else torch.ones(1, device=device)

        unet_in  = torch.cat([f0_up, y_t], dim=1)
        eps_pred = unet(unet_in, t_tensor, f0_up)

        y0_pred = (y_t - (1 - ab_t).sqrt() * eps_pred) / (ab_t.sqrt() + 1e-8)
        y0_pred = y0_pred.clamp(-1, 1)

        if t_next > 0:
            y_t = ab_tm1.sqrt() * y0_pred + (1 - ab_tm1).sqrt() * eps_pred
        else:
            y_t = y0_pred

    return y_t


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--patch_dir',  default='data/processed/patches')
    p.add_argument('--epochs',     type=int,   default=100)
    p.add_argument('--batch_size', type=int,   default=4)
    p.add_argument('--lr',         type=float, default=2e-4)
    p.add_argument('--T',          type=int,   default=1000)
    p.add_argument('--n_steps',    type=int,   default=20)
    p.add_argument('--save_dir',   default='checkpoints/base_paper')
    p.add_argument('--log_path',   default='logs/base_paper_train.log')
    p.add_argument('--resume',     action='store_true')
    p.add_argument('--data_root',      default='data_base_readonly/processed/patches')
    p.add_argument('--use_swin',       action='store_true')
    p.add_argument('--use_physics_loss', action='store_true')
    p.add_argument('--lambda_phys',    type=float, default=0.05)
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
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    from src.models.vgg19_full   import KDVGGStudent16, VGG19_STUDENT_CH_16
    from src.models.base_kd_unet import BaseKDUNet, UNET_CH

    if args.use_swin:
        from src.models.swin_moe_encoder import SwinMoEEncoder
        student = SwinMoEEncoder(in_chans=6, out_ch=256, pretrained=True, n_experts=4).to(device)
        COND_CH = 256
        print("Encoder: SwinV2-Base + MoE-4 (Physics-Informed IIDM)")
    else:
        student = KDVGGStudent16(in_channels=6).to(device)
        block_ckpt = 'checkpoints/blockwise_kd.pth'
        assert os.path.exists(block_ckpt), f"Run train_blockwise_kd.py first! Missing: {block_ckpt}"
        student.load_state_dict(torch.load(block_ckpt, map_location=device)['student'])
        COND_CH = VGG19_STUDENT_CH_16[-1]
        print(f"Encoder: VGG19 Student (COND_CH={COND_CH})")
    unet = BaseKDUNet(in_ch=COND_CH + 1, cond_ch=COND_CH).to(device)
    phys_criterion = PhysicsInformedLoss(lambda_phys=args.lambda_phys).to(device)

    params = (list(student.parameters()) +
              list(unet.parameters()))
    optimizer = torch.optim.Adam(params, lr=args.lr)

    betas, alpha_bar = make_schedule(args.T, device)

    # Carbon normalization from norm_stats.json
    # mean=36.20, std=39.78 Mg C/ha
    C_MIN = 4.816495895385742    # base IIDM norm_stats match
    C_MAX = 129.18380737304688   # base IIDM norm_stats match


    best_rmse   = float('inf')
    start_epoch = 1
    log         = []

    resume_ckpt = os.path.join(args.save_dir, 'base_resume.pth')
    if args.resume and os.path.exists(resume_ckpt):
        ckpt = torch.load(resume_ckpt, map_location=device)
        student.load_state_dict(ckpt['student'])
        unet.load_state_dict(ckpt['unet'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_rmse   = ckpt['best_rmse']
        log         = ckpt.get('log', [])
        print(f"Resumed epoch {ckpt['epoch']}, best RMSE={best_rmse:.4f}")

    for epoch in range(start_epoch, args.epochs + 1):
        student.train(); unet.train()
        train_loss = 0.0

        for x, y0, mask in train_ld:
            x, y0, mask = x.to(device), y0.to(device), mask.to(device)
            B, _, H, W = y0.shape

            s_feats = student(x)
            f0      = s_feats[-1]
            f0_up   = F.interpolate(f0, size=(H, W), mode='bilinear',
                                    align_corners=False)

            # DDPM forward: paper Eq.4
            t_idx = torch.randint(1, args.T + 1, (B,), device=device)
            ab    = alpha_bar[t_idx - 1].view(B, 1, 1, 1)
            eps   = torch.randn_like(y0)
            y_t   = ab.sqrt() * y0 + (1 - ab).sqrt() * eps

            unet_in  = torch.cat([f0_up, y_t], dim=1)
            eps_pred = unet(unet_in, t_idx, f0_up)

            diff = (eps_pred - eps).abs() * mask
            loss = diff.sum() / (mask.sum() + 1e-8)

            # ── Physics-Informed Loss (PIDM-style, optional) ──────────────
            if args.use_physics_loss:
                canopy_h  = x[:, 5:6, :, :]          # ch5 = canopy height
                sigma_t   = (1 - ab).sqrt().mean().item()  # noise level at t
                phys_loss = phys_criterion(y0, canopy_h, sigma_t)
                loss      = loss + phys_loss
            # ─────────────────────────────────────────────────────────────

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_ld)

        # Validation: direct y0_pred at fixed t=500 (mid-noise)
        # This monitors training progress stably
        student.eval(); unet.eval()
        all_gt, all_pred = [], []
        t_val = args.T // 2   # fixed t=500

        with torch.no_grad():
            for x, y0, mask in val_ld:
                x, y0, mask = x.to(device), y0.to(device), mask.to(device)
                B, _, H, W = y0.shape

                s_feats = student(x)
                f0      = s_feats[-1]
                f0_up = F.interpolate(f0, size=(H, W), mode='bilinear',
                                      align_corners=False)

                torch.manual_seed(42)
                eps_v = torch.randn_like(y0)
                ab_v  = alpha_bar[t_val - 1]
                y_t_v = ab_v.sqrt() * y0 + (1 - ab_v).sqrt() * eps_v

                t_tensor = torch.full((B,), t_val, device=device, dtype=torch.long)
                unet_in  = torch.cat([f0_up, y_t_v], dim=1)
                eps_p    = unet(unet_in, t_tensor, f0_up)

                y0_pred = (y_t_v - (1 - ab_v).sqrt() * eps_p) / (ab_v.sqrt() + 1e-8)
                y0_pred = y0_pred.clamp(-1, 1)

                mask_np = mask.cpu().numpy() > 0          # (B,1,H,W) bool
                pred_mg = (y0_pred.cpu().numpy() * 0.5 + 0.5) * (C_MAX - C_MIN) + C_MIN
                gt_mg   = (y0.cpu().numpy()      * 0.5 + 0.5) * (C_MAX - C_MIN) + C_MIN
                pred_mg = pred_mg[mask_np]        # only valid pixels
                gt_mg   = gt_mg[mask_np]
                all_pred.append(pred_mg)
                all_gt.append(gt_mg)

        all_pred = np.concatenate(all_pred)
        all_gt   = np.concatenate(all_gt)

        rmse = float(np.sqrt(np.mean((all_pred - all_gt)**2)))
        mae  = float(np.mean(np.abs(all_pred - all_gt)))
        ss_r = np.sum((all_gt - all_pred)**2)
        ss_t = np.sum((all_gt - all_gt.mean())**2)
        r2   = float(1 - ss_r / (ss_t + 1e-8))

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}",
              flush=True)

        log.append(dict(epoch=epoch, train_loss=train_loss,
                        rmse=rmse, mae=mae, r2=r2))

        if rmse < best_rmse:
            best_rmse = rmse
            torch.save({
                'epoch': epoch, 'rmse': rmse, 'mae': mae, 'r2': r2,
                'student': student.state_dict(),
                'unet':    unet.state_dict(),
                }, os.path.join(args.save_dir, 'base_best.pth'))
            print(f"  ✓ Best saved (RMSE={rmse:.4f})", flush=True)

        torch.save({
            'epoch': epoch, 'best_rmse': best_rmse,
            'student': student.state_dict(),
            'unet': unet.state_dict(),
            'optimizer': optimizer.state_dict(),
            'log': log,
        }, resume_ckpt)

        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'student': student.state_dict(),
                'unet': unet.state_dict(),
            }, os.path.join(args.save_dir, f'base_epoch_{epoch}.pth'))

        with open(args.log_path, 'w') as f:
            json.dump(log, f, indent=2)

    print(f"\nBase paper training done.")
    print(f"Best Val RMSE : {best_rmse:.4f} Mg C/ha")
    print(f"Paper reports : 12.17 Mg C/ha")


if __name__ == '__main__':
    main()
