"""
Test-set evaluation for the base IIDM paper model.
Uses the SAME methodology as training-time validation:
  direct y0_pred from a single fixed noisy step at t = T // 2 (t=500).
No DDIM multi-step sampling — intentionally, per user instruction, since
validation used direct t=500 prediction and test must match it exactly.

Loads: checkpoints/base_paper/base_best.pth  (student + unet state_dicts)
Evaluates on: data/processed/patches_v2/test
"""
import os, sys
import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train_base import PatchDataset, make_schedule


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--patch_dir',  default='data/processed/patches_v2')
    p.add_argument('--ckpt',       default='checkpoints/base_paper/base_best.pth')
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--T',          type=int, default=1000)
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

    student = KDVGGStudent16(in_channels=4).to(device)
    COND_CH = VGG19_STUDENT_CH_16[-1]
    unet = BaseKDUNet(in_ch=COND_CH + 1, cond_ch=COND_CH).to(device)

    assert os.path.exists(args.ckpt), f"Checkpoint not found: {args.ckpt}"
    ckpt = torch.load(args.ckpt, map_location=device)
    student.load_state_dict(ckpt['student'])
    unet.load_state_dict(ckpt['unet'])
    print(f"Loaded checkpoint: epoch={ckpt.get('epoch')}, "
          f"val_rmse={ckpt.get('rmse'):.4f}")

    student.eval(); unet.eval()

    betas, alpha_bar = make_schedule(args.T, device)

    # Same normalization constants as train_base.py
    C_MEAN = 33.41
    C_STD  = 38.21

    t_val = args.T // 2   # fixed t=500, same as validation

    all_gt, all_pred = [], []
    with torch.no_grad():
        for x, y0, mask in test_ld:
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
            y0_pred = y0_pred.clamp(-3, 3)

            pred_mg = y0_pred.cpu().numpy() * C_STD + C_MEAN
            gt_mg   = y0.cpu().numpy()      * C_STD + C_MEAN
            m_np    = mask.cpu().numpy().astype(bool)

            all_pred.append(pred_mg[m_np])
            all_gt.append(gt_mg[m_np])

    all_pred = np.concatenate(all_pred)
    all_gt   = np.concatenate(all_gt)

    rmse = float(np.sqrt(np.mean((all_pred - all_gt) ** 2)))
    mae  = float(np.mean(np.abs(all_pred - all_gt)))
    ss_r = np.sum((all_gt - all_pred) ** 2)
    ss_t = np.sum((all_gt - all_gt.mean()) ** 2)
    r2   = float(1 - ss_r / (ss_t + 1e-8))

    print("\n" + "=" * 50)
    print("TEST SET RESULTS (t=500 direct prediction)")
    print("=" * 50)
    print(f"Test RMSE : {rmse:.4f} Mg C/ha")
    print(f"Test MAE  : {mae:.4f} Mg C/ha")
    print(f"Test R2   : {r2:.4f}")
    print("-" * 50)
    print("Paper RMSE: 12.17 | Paper MAE: 9.91")
    print("=" * 50)


if __name__ == '__main__':
    main()
