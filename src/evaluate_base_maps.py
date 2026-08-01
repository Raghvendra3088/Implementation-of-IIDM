"""
Base IIDM Map Generation and Evaluation Script
==============================================
Paper Step-4: Output Estimation Analysis
This script evaluates the base IIDM model using the 20-step DDIM inference,
computes metrics, and generates visualizations (Scatter plots & Carbon Density Maps).
"""

import os, sys, json, argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train_base import PatchDataset, make_schedule, ddim_sample
from src.models.vgg19_full import KDVGGStudent16, VGG19_STUDENT_CH_16
from src.models.base_kd_unet import BaseKDUNet, UNET_CH

def denorm(arr, vmin, vmax):
    """Denormalize from [-1, 1] back to physical carbon range (Mg C/ha)"""
    return (arr * 0.5 + 0.5) * (vmax - vmin) + vmin

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--patch_dir',  default='data/processed/patches_v2')
    p.add_argument('--ckpt',       default='checkpoints/base_paper/base_best.pth')
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--T',          type=int, default=1000)
    p.add_argument('--n_steps',    type=int, default=20)
    p.add_argument('--seed',       type=int, default=42)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create directories for results
    fig_dir = Path("results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    test_ds = PatchDataset(args.patch_dir, 'test')
    test_ld = DataLoader(test_ds, batch_size=args.batch_size,
                          shuffle=False, num_workers=2, pin_memory=True)
    print(f"Test: {len(test_ds)} patches")

    student = KDVGGStudent16(in_channels=4).to(device)
    COND_CHS = [VGG19_STUDENT_CH_16[i] for i in [1, 3, 7, 11, 15]]
    unet = BaseKDUNet(in_ch=5, cond_chs=COND_CHS).to(device)

    assert os.path.exists(args.ckpt), f"Checkpoint not found: {args.ckpt}"
    ckpt = torch.load(args.ckpt, map_location=device)
    student.load_state_dict(ckpt['student'])
    unet.load_state_dict(ckpt['unet'])
    print(f"Loaded checkpoint: epoch={ckpt.get('epoch')}")

    student.eval(); unet.eval()
    betas, alpha_bar = make_schedule(args.T, device)

    # Normalization constants used during training
    C_MIN = 4.816495895385742
    C_MAX = 129.18380737304688

    all_gt, all_pred = [], []

    print("\nRunning inference on test set to generate maps & scatter plots...")
    
    # Store a sample prediction for visualization
    sample_inp, sample_gt, sample_pred = None, None, None

    with torch.no_grad():
        for bi, (x, y0, mask) in enumerate(test_ld):
            x, y0, mask = x.to(device), y0.to(device), mask.to(device)
            B, _, H, W = y0.shape

            s_feats = student(x)
            f_multi = [s_feats[i] for i in [1, 3, 7, 11, 15]]

            y0_pred = ddim_sample(unet, x, f_multi, alpha_bar, args.T,
                                   device, n_steps=args.n_steps, seed=args.seed)

            pred_mg = denorm(y0_pred.cpu().numpy(), C_MIN, C_MAX)
            gt_mg   = denorm(y0.cpu().numpy(), C_MIN, C_MAX)
            m_np    = mask.cpu().numpy().astype(bool)

            # Store the first batch's first item for visualization
            if sample_inp is None:
                sample_inp = x[0].cpu().numpy()
                sample_gt = gt_mg[0, 0]
                sample_pred = pred_mg[0, 0]

            for b in range(B):
                mask_b = m_np[b, 0]
                all_pred.append(pred_mg[b, 0][mask_b])
                all_gt.append(gt_mg[b, 0][mask_b])

            if (bi + 1) % 50 == 0:
                print(f"  Processed batch {bi+1}/{len(test_ld)}", flush=True)

    all_pred = np.concatenate(all_pred)
    all_gt   = np.concatenate(all_gt)

    rmse = float(np.sqrt(np.mean((all_pred - all_gt) ** 2)))
    mae  = float(np.mean(np.abs(all_pred - all_gt)))
    ss_r = np.sum((all_gt - all_pred) ** 2)
    ss_t = np.sum((all_gt - all_gt.mean()) ** 2)
    r2   = float(1 - ss_r / (ss_t + 1e-8))

    print("\n" + "=" * 50)
    print("TEST METRICS (Base IIDM)")
    print(f"RMSE : {rmse:.4f} Mg C/ha")
    print(f"MAE  : {mae:.4f} Mg C/ha")
    print(f"R2   : {r2:.4f}")
    print("=" * 50)

    # ── 1. Scatter Plot ───────────────────────────────────────────────────────
    print("\nGenerating scatter plot...")
    n_pts = min(5000, len(all_pred))
    idx   = np.random.choice(len(all_pred), n_pts, replace=False)
    px, gx = all_pred[idx], all_gt[idx]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(gx, px, alpha=0.3, s=5, color="steelblue", label="Test Patches")
    lim = [min(gx.min(), px.min()), max(gx.max(), px.max())]
    ax.plot(lim, lim, "r--", linewidth=1.5, label="1:1 line")
    ax.set_xlabel("Ground Truth (Mg C/ha)", fontsize=12)
    ax.set_ylabel("Predicted (Mg C/ha)",    fontsize=12)
    ax.set_title(f"Predicted vs Ground Truth (Base IIDM)\nR²={r2:.3f}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "base_scatter_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {fig_dir / 'base_scatter_plot.png'}")

    # ── 2. Sample Prediction Visualization (Carbon Map) ───────────────────────
    print("Generating Carbon Stock Density Map (sample patch)...")
    err_s = np.abs(sample_pred - sample_gt)

    # RGB from Sentinel bands (Assumes B04=R, B03=G, B02=B -> channels 2, 1, 0)
    # The first 4 channels are typically B, G, R, NIR, so B,G,R are 0,1,2.
    rgb = sample_inp[[2, 1, 0]].transpose(1, 2, 0)
    rgb = np.clip((rgb + 1) / 2 * 3, 0, 1)   # denorm [-1,1] -> [0,1], brighten

    vmin_c = min(sample_pred.min(), sample_gt.min())
    vmax_c = max(sample_pred.max(), sample_gt.max())

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"Base IIDM Output Estimation Analysis", fontsize=13, fontweight="bold")

    axes[0].imshow(rgb)
    axes[0].set_title("Sentinel-2 RGB", fontsize=11)
    axes[0].axis("off")

    im1 = axes[1].imshow(sample_gt, cmap="YlGn", vmin=vmin_c, vmax=vmax_c)
    axes[1].set_title("Ground Truth (Mg C/ha)", fontsize=11)
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(sample_pred, cmap="YlGn", vmin=vmin_c, vmax=vmax_c)
    axes[2].set_title("Base IIDM Prediction (Mg C/ha)", fontsize=11)
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    im3 = axes[3].imshow(err_s, cmap="Reds")
    axes[3].set_title("Absolute Error (Mg C/ha)", fontsize=11)
    axes[3].axis("off")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(fig_dir / "base_carbon_map_sample.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {fig_dir / 'base_carbon_map_sample.png'}")

    print("=" * 50)
    print("Step 4: Output Estimation Analysis successfully completed!")

if __name__ == '__main__':
    main()
