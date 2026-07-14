"""
Paper-exact test-set evaluation for the base IIDM model.
Paper Table A3: full DDIM inference, n_steps=20.
No shortcuts, no deviations from paper methodology.

Loads: checkpoints/base_paper/base_best.pth  (student + unet state_dicts)
Evaluates on: data/processed/patches_v2/test
"""
import os, sys
import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train_base import PatchDataset, make_schedule, ddim_sample


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--patch_dir',  default='data/processed/patches')
    p.add_argument('--ckpt',       default='checkpoints/base_paper/base_best.pth')
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--T',          type=int, default=1000)
    p.add_argument('--n_steps',    type=int, default=20)
    p.add_argument('--seed',       type=int, default=42)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    test_ds = PatchDataset(args.patch_dir, 'test')
    test_ld = DataLoader(test_ds, batch_size=args.batch_size,
                          shuffle=False, num_workers=2, pin_memory=True)
    print(f"Test: {len(test_ds)} patches")

    from src.models.vgg19_full   import KDVGGStudent16, VGG19_STUDENT_CH_16
    from src.models.base_kd_unet import BaseKDUNet, UNET_CH
    import torch.nn.functional as F

    student = KDVGGStudent16(in_channels=6).to(device)
    COND_CH = VGG19_STUDENT_CH_16[-1]
    unet = BaseKDUNet(in_ch=COND_CH + 1, cond_ch=COND_CH).to(device)

    assert os.path.exists(args.ckpt), f"Checkpoint not found: {args.ckpt}"
    ckpt = torch.load(args.ckpt, map_location=device)
    student.load_state_dict(ckpt['student'])
    unet.load_state_dict(ckpt['unet'])
    print(f"Loaded checkpoint: epoch={ckpt.get('epoch')}, "
          f"val_rmse={ckpt.get('rmse', ckpt.get('val_rmse', 'N/A'))}")

    student.eval(); unet.eval()

    betas, alpha_bar = make_schedule(args.T, device)

    # Same normalization constants as train_base.py
    C_MIN = 4.816495895385742
    C_MAX = 129.18380737304688
    

    all_gt, all_pred = [], []
    with torch.no_grad():
        for bi, (x, y0, mask) in enumerate(test_ld):
            x, y0, mask = x.to(device), y0.to(device), mask.to(device)
            B, _, H, W = y0.shape

            s_feats = student(x)
            f0      = s_feats[-1]
            f0_up = F.interpolate(f0, size=(H, W), mode='bilinear',
                                  align_corners=False)

            y0_pred = ddim_sample(unet, f0_up, alpha_bar, args.T,
                                   device, n_steps=args.n_steps, seed=args.seed)

            pred_mg = (y0_pred.cpu().numpy() * 0.5 + 0.5) * (C_MAX - C_MIN) + C_MIN
            gt_mg   = (y0.cpu().numpy() * 0.5 + 0.5) * (C_MAX - C_MIN) + C_MIN
            m_np    = mask.cpu().numpy().astype(bool)

            all_pred.append(pred_mg[m_np])
            all_gt.append(gt_mg[m_np])

            if (bi + 1) % 50 == 0:
                print(f"  batch {bi+1}/{len(test_ld)}", flush=True)

    all_pred = np.concatenate(all_pred)
    all_gt   = np.concatenate(all_gt)

    rmse = float(np.sqrt(np.mean((all_pred - all_gt) ** 2)))
    mae  = float(np.mean(np.abs(all_pred - all_gt)))
    ss_r = np.sum((all_gt - all_pred) ** 2)
    ss_t = np.sum((all_gt - all_gt.mean()) ** 2)
    r2   = float(1 - ss_r / (ss_t + 1e-8))

    print("\n" + "=" * 50)
    print(f"TEST SET RESULTS (DDIM, n_steps={args.n_steps}, paper Table A3)")
    print("=" * 50)
    print(f"Test RMSE : {rmse:.4f} Mg C/ha")
    print(f"Test MAE  : {mae:.4f} Mg C/ha")
    print(f"Test R2   : {r2:.4f}")
    print("-" * 50)
    print("Paper RMSE: 12.17 | Paper MAE: 9.91")
    print("=" * 50)


if __name__ == '__main__':
    main()
