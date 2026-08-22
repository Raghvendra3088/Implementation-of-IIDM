"""
IIDM Evaluation Script
=======================
Paper metrics: RMSE, MAE, SSIM, R²

Loads best checkpoint → runs DDIM inference on test set →
denormalizes predictions → computes metrics → saves results

Outputs:
    results/metrics.json              — all metrics
    results/figures/sample_pred.png   — prediction vs GT visualization
    results/figures/loss_curve.png    — training loss curve
    results/figures/scatter_plot.png  — predicted vs actual scatter
    results/figures/carbon_pred.tif   — GeoTIFF carbon map (first test patch)

Run:
    python src/evaluate.py
    python src/evaluate.py --ckpt checkpoints/best_model.pth --ddim_steps 50
"""

import os, json, argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim_fn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.iidm  import IIDM

from src.train        import IIDMDataset


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def rmse(pred, gt):
    return float(np.sqrt(np.mean((pred - gt) ** 2)))

def mae(pred, gt):
    return float(np.mean(np.abs(pred - gt)))

def r2(pred, gt):
    ss_res = np.sum((gt - pred) ** 2)
    ss_tot = np.sum((gt - gt.mean()) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-8))

def ssim(pred, gt):
    # pred, gt: (H, W) float32
    data_range = gt.max() - gt.min()
    if data_range < 1e-6:
        return 1.0
    return float(ssim_fn(pred, gt, data_range=float(data_range)))

def denorm(arr, vmin, vmax):
    """[-1,1] → Mg C/ha"""
    return (arr + 1.0) / 2.0 * (vmax - vmin) + vmin


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION LOOP
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"\n{'='*55}")
    print(f"  IIDM Evaluation")
    print(f"{'='*55}")
    print(f"  Device      : {device}")
    print(f"  Checkpoint  : {args.ckpt}")
    print(f"  DDIM steps  : {args.ddim_steps}")

    # ── Dirs ──────────────────────────────────────────────────────────────────
    fig_dir = Path("results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── Load norm stats ───────────────────────────────────────────────────────
    stats_path = Path("data/processed/norm_stats_final.json")
    if stats_path.exists():
        with open(stats_path) as f:
            norm_stats = json.load(f)
        C_MIN = norm_stats["carbon"]["min"]
        C_MAX = norm_stats["carbon"]["max"]
    else:
        C_MIN, C_MAX = 0.04, 207.97   # from our preprocessing output
    print(f"  Carbon range: [{C_MIN:.2f}, {C_MAX:.2f}] Mg C/ha")

    # ── Dataset ───────────────────────────────────────────────────────────────
    test_ds = IIDMDataset("test", augment=False, patch_dir=args.patch_dir)
    loader  = DataLoader(test_ds, batch_size=1,
                         shuffle=False, num_workers=0)
    print(f"  Test patches: {len(test_ds)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = IIDM(in_channels=6, T=1000, device=device).to(device)

    if not Path(args.ckpt).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.ckpt}\n"
            "Run src/train.py first."
        )

    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.student_vgg.load_state_dict(ckpt["student_vgg"])
    model.unet.load_state_dict(ckpt["unet"])

    model.inr.load_state_dict(ckpt["inr"])
    if "teacher_cond" in ckpt:
        model.teacher_cond.load_state_dict(ckpt["teacher_cond"])
    model.eval()
    print(f"  Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # ══════════════════════════════════════════════════════════════════════════
    # INFERENCE LOOP
    # ══════════════════════════════════════════════════════════════════════════

    all_rmse, all_mae, all_ssim, all_r2 = [], [], [], []
    all_preds, all_gts = [], []

    print(f"\n  Running inference on {len(test_ds)} patches ...")

    with torch.no_grad():
        for i, (inp, tgt) in enumerate(loader):
            inp = inp.to(device)

            # DDIM inference
            pred = model.predict(inp, steps=args.ddim_steps)  # (1,1,256,256)

            pred_np = pred.cpu().numpy()[0, 0]   # (256, 256) normalized
            gt_np   = tgt.numpy()[0, 0]          # (256, 256) normalized

            # Denormalize → Mg C/ha
            pred_real = denorm(pred_np, C_MIN, C_MAX)
            gt_real   = denorm(gt_np,   C_MIN, C_MAX)

            # Metrics in real units
            all_rmse.append(rmse(pred_real, gt_real))
            all_mae.append( mae( pred_real, gt_real))
            all_r2.append(  r2(  pred_real, gt_real))
            all_ssim.append(ssim(pred_np,   gt_np))  # SSIM on normalized

            all_preds.append(pred_real.flatten())
            all_gts.append(  gt_real.flatten())

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(test_ds)}]  "
                      f"RMSE={all_rmse[-1]:.3f}  "
                      f"MAE={all_mae[-1]:.3f}  "
                      f"SSIM={all_ssim[-1]:.3f}")

    # ══════════════════════════════════════════════════════════════════════════
    # AGGREGATE METRICS
    # ══════════════════════════════════════════════════════════════════════════

    metrics = {
        "RMSE" : float(np.mean(all_rmse)),
        "MAE"  : float(np.mean(all_mae)),
        "SSIM" : float(np.mean(all_ssim)),
        "R2"   : float(np.mean(all_r2)),
        "RMSE_std" : float(np.std(all_rmse)),
        "MAE_std"  : float(np.std(all_mae)),
        "n_patches": len(test_ds),
        "ddim_steps": args.ddim_steps,
        "carbon_vmin": C_MIN,
        "carbon_vmax": C_MAX,
    }
    
   # NEW: RMSE% / MAE% relative to mean carbon density
    carbon_mean = float(np.mean(np.concatenate(all_gts)))
    metrics["carbon_mean"]   = carbon_mean
    metrics["RMSE_percent"]  = (metrics["RMSE"] / carbon_mean) * 100
    metrics["MAE_percent"]   = (metrics["MAE"]  / carbon_mean) * 100

    print(f"\n{'─'*45}")
    print(f"  TEST RESULTS ({len(test_ds)} patches)")
    print(f"{'─'*45}")
    print(f"  RMSE : {metrics['RMSE']:.4f} ± {metrics['RMSE_std']:.4f}  Mg C/ha")
    print(f"  MAE  : {metrics['MAE']:.4f} ± {metrics['MAE_std']:.4f}  Mg C/ha")
    print(f"  SSIM : {metrics['SSIM']:.4f}")
    print(f"  R²   : {metrics['R2']:.4f}")
    print(f"  RMSE%: {metrics['RMSE_percent']:.2f}%  (Paper: 12.17%)")
    print(f"  MAE% : {metrics['MAE_percent']:.2f}%")
    print(f"{'─'*45}")


    out_path = Path("results/metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics saved → {out_path}")

    # ══════════════════════════════════════════════════════════════════════════
    # VISUALIZATIONS
    # ══════════════════════════════════════════════════════════════════════════

    preds_all = np.concatenate(all_preds)
    gts_all   = np.concatenate(all_gts)

    # ── 1. Sample prediction visualization ───────────────────────────────────
    print("\n  Generating visualizations ...")
    with torch.no_grad():
        inp_s, tgt_s = test_ds[0]
        pred_s = model.predict(
            inp_s.unsqueeze(0).to(device), steps=args.ddim_steps
        ).cpu().numpy()[0, 0]
        gt_s   = tgt_s.numpy()[0]

    pred_real_s = denorm(pred_s, C_MIN, C_MAX)
    gt_real_s   = denorm(gt_s,   C_MIN, C_MAX)
    err_s       = np.abs(pred_real_s - gt_real_s)

    # RGB from sentinel bands (B04=R, B03=G, B02=B → channels 2,1,0)
    rgb = inp_s.numpy()[[2, 1, 0]].transpose(1, 2, 0)
    rgb = np.clip((rgb + 1) / 2 * 3, 0, 1)   # denorm [-1,1] → [0,1], brighten

    vmin_c = min(pred_real_s.min(), gt_real_s.min())
    vmax_c = max(pred_real_s.max(), gt_real_s.max())

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(
        f"IIDM Carbon Stock Prediction\n"
        f"RMSE={metrics['RMSE']:.2f} | MAE={metrics['MAE']:.2f} | "
        f"SSIM={metrics['SSIM']:.3f} | R²={metrics['R2']:.3f}",
        fontsize=13, fontweight="bold"
    )

    axes[0].imshow(rgb)
    axes[0].set_title("Sentinel-2 RGB", fontsize=11)
    axes[0].axis("off")

    im1 = axes[1].imshow(gt_real_s, cmap="YlGn",
                          vmin=vmin_c, vmax=vmax_c)
    axes[1].set_title("Ground Truth (Mg C/ha)", fontsize=11)
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(pred_real_s, cmap="YlGn",
                          vmin=vmin_c, vmax=vmax_c)
    axes[2].set_title("IIDM Prediction (Mg C/ha)", fontsize=11)
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    im3 = axes[3].imshow(err_s, cmap="Reds")
    axes[3].set_title("Absolute Error (Mg C/ha)", fontsize=11)
    axes[3].axis("off")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(fig_dir / "sample_pred.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → results/figures/sample_pred.png")

    # ── 2. Scatter plot: predicted vs actual ──────────────────────────────────
    # Subsample for clarity
    n_pts = min(5000, len(preds_all))
    idx   = np.random.choice(len(preds_all), n_pts, replace=False)
    px, gx = preds_all[idx], gts_all[idx]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(gx, px, alpha=0.3, s=5, color="steelblue", label="Patches")
    lim = [min(gx.min(), px.min()), max(gx.max(), px.max())]
    ax.plot(lim, lim, "r--", linewidth=1.5, label="1:1 line")
    ax.set_xlabel("Ground Truth (Mg C/ha)", fontsize=12)
    ax.set_ylabel("Predicted (Mg C/ha)",    fontsize=12)
    ax.set_title(f"Predicted vs Ground Truth\nR²={metrics['R2']:.3f}",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "scatter_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → results/figures/scatter_plot.png")

    # ── 3. Training loss curve ────────────────────────────────────────────────
    log_path = Path("results/train_log.json")
    if log_path.exists():
        with open(log_path) as f:
            log = json.load(f)

        epochs     = [e["epoch"]              for e in log]
        tr_total   = [e["train"]["L_total"]   for e in log]
        val_total  = [e["val"]["L_total"]     for e in log]
        tr_diff    = [e["train"]["L_diff"]    for e in log]
        tr_recon   = [e["train"]["L_recon"]   for e in log]
        val_rmse_l = [e.get("val_rmse", 0)   for e in log]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("IIDM Training Curves", fontsize=13, fontweight="bold")

        axes[0].plot(epochs, tr_total,  label="Train Total", color="steelblue")
        axes[0].plot(epochs, val_total, label="Val Total",   color="tomato")
        axes[0].set_title("Total Loss"); axes[0].set_xlabel("Epoch")
        axes[0].legend(); axes[0].grid(alpha=0.3)

        axes[1].plot(epochs, tr_diff,  label="L_diff",  color="purple")
        axes[1].plot(epochs, tr_recon, label="L_recon", color="green")
        axes[1].set_title("Loss Components"); axes[1].set_xlabel("Epoch")
        axes[1].legend(); axes[1].grid(alpha=0.3)

        axes[2].plot(epochs, val_rmse_l, color="darkorange", label="Val RMSE")
        axes[2].set_title("Val RMSE (normalized)"); axes[2].set_xlabel("Epoch")
        axes[2].legend(); axes[2].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(fig_dir / "loss_curve.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved → results/figures/loss_curve.png")

    # ── 4. Save first prediction as GeoTIFF ───────────────────────────────────
    try:
        import rasterio
        from rasterio.transform import from_bounds
        # Huize County approximate bounds (WGS84)
        transform = from_bounds(103.0, 25.8, 103.8, 26.6, 256, 256)
        meta = {
            "driver"   : "GTiff",
            "dtype"    : "float32",
            "width"    : 256,
            "height"   : 256,
            "count"    : 1,
            "crs"      : "EPSG:4326",
            "transform": transform,
            "compress" : "lzw",
        }
        tif_path = fig_dir / "carbon_pred.tif"
        with rasterio.open(tif_path, "w", **meta) as dst:
            dst.write(pred_real_s.astype(np.float32)[np.newaxis])
        print(f"  Saved → results/figures/carbon_pred.tif")
    except Exception as e:
        print(f"  GeoTIFF skipped: {e}")

    print(f"\n{'='*55}")
    print(f"  Evaluation complete!")
    print(f"  Results → results/")
    print(f"{'='*55}")

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# ARGS
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate IIDM")
    p.add_argument("--ckpt",       type=str, default="checkpoints/best_model.pth")
    p.add_argument("--ddim_steps", type=int, default=50)
    p.add_argument("--patch_dir",  type=str,
                   default="data/processed/patches_final")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
