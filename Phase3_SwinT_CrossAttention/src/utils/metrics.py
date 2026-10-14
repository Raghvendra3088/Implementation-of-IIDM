import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim


def calculate_metrics(pred, gt):
    """
    Computes patch-level regression metrics with strict zero-variance protection.
    """
    mse = F.mse_loss(pred, gt)
    rmse = torch.sqrt(mse)
    mae = torch.mean(torch.abs(pred - gt))

    # SSIM requires numpy array
    pred_np = pred.detach().cpu().numpy()[0, 0]
    gt_np = gt.detach().cpu().numpy()[0, 0]
    drange = float(max(1e-4, gt_np.max() - gt_np.min()))
    ssim_val = float(ssim(gt_np, pred_np, data_range=drange))

    # R^2 calculation with zero-variance protection
    ss_res = torch.sum((gt - pred) ** 2)
    ss_tot = torch.sum((gt - torch.mean(gt)) ** 2)
    if ss_tot.item() < 1e-4:
        r2 = torch.tensor(0.0)
    else:
        r2 = 1.0 - (ss_res / ss_tot)
        r2 = torch.clamp(r2, min=-10.0, max=1.0)

    return {"RMSE": rmse.item(), "MAE": mae.item(), "SSIM": ssim_val, "R2": r2.item()}


def calculate_dataset_metrics(preds, targets, inputs=None, scale_factor=130.0, stand_aggregation_scale=None):
    """
    Computes statistically robust dataset-wide metrics across all test pixels.
    Avoids patch-averaging artifacts and optionally evaluates at stand-level spatial support (e.g. 100m).
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    if inputs is not None and isinstance(inputs, torch.Tensor):
        inputs = inputs.detach().cpu().numpy()

    p_norm = preds.astype(np.float64)
    t_norm = targets.astype(np.float64)

    # Multi-scale spatial support aggregation (e.g. 100m / 1ha support = 10x10 pooling on 10m pixels)
    if stand_aggregation_scale is not None and stand_aggregation_scale > 1:
        k = int(stand_aggregation_scale)
        B, C, H, W = p_norm.shape
        H_pad = (H // k) * k
        W_pad = (W // k) * k
        p_sub = p_norm[:, :, :H_pad, :W_pad].reshape(B, C, H_pad // k, k, W_pad // k, k).mean(axis=(3, 5))
        t_sub = t_norm[:, :, :H_pad, :W_pad].reshape(B, C, H_pad // k, k, W_pad // k, k).mean(axis=(3, 5))
        p_flat = p_sub.ravel()
        t_flat = t_sub.ravel()
    else:
        p_flat = p_norm.ravel()
        t_flat = t_norm.ravel()

    # Global pixel statistics
    mae_norm = float(np.mean(np.abs(p_flat - t_flat)))
    rmse_norm = float(np.sqrt(np.mean((p_flat - t_flat) ** 2)))

    ss_res = np.sum((t_flat - p_flat) ** 2)
    ss_tot = np.sum((t_flat - np.mean(t_flat)) ** 2)
    r2_global = float(1.0 - (ss_res / (ss_tot + 1e-10)))

    # Global SSIM
    ssim_sum = 0.0
    n_patches = p_norm.shape[0]
    for i in range(n_patches):
        gt_patch = t_norm[i, 0]
        pr_patch = p_norm[i, 0]
        dr = float(max(1e-4, gt_patch.max() - gt_patch.min()))
        ssim_sum += ssim(gt_patch, pr_patch, data_range=dr)
    ssim_mean = float(ssim_sum / max(1, n_patches))

    # Biophysical Consistency: PVR and MND
    pvr, mnd = 0.0, 0.0
    if inputs is not None and inputs.shape[1] >= 8:
        chm_mean = inputs[:, 7].mean(axis=(1, 2))
        bio_mean = preds[:, 0].mean(axis=(1, 2))
        idx = np.argsort(chm_mean)
        diffs = np.diff(bio_mean[idx])
        pvr = float(np.mean(diffs < 0))
        mnd = float(np.mean(np.clip(-diffs, 0, None)))

    # Physical scale conversion
    mae_scaled = float(mae_norm * scale_factor)
    rmse_scaled = float(rmse_norm * scale_factor)

    return {
        "MAE_norm": mae_norm,
        "RMSE_norm": rmse_norm,
        "MAE_Mg_C_ha": mae_scaled,
        "RMSE_Mg_C_ha": rmse_scaled,
        "R2_global": r2_global,
        "SSIM": ssim_mean,
        "PVR": pvr,
        "MND": mnd,
        "Relative_RMSE_percent": float((rmse_scaled / scale_factor) * 100.0)
    }
