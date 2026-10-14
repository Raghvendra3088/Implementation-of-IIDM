import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim

def calculate_metrics(pred, gt, eps=1e-6):
    """
    Computes per-patch metrics safely.
    For R2, guards against exploding negative numbers when patch variance is near zero.
    """
    mse = F.mse_loss(pred, gt)
    rmse = torch.sqrt(mse)
    mae = torch.mean(torch.abs(pred - gt))
    
    # SSIM requires numpy array
    pred_np = pred.detach().cpu().numpy()[0, 0]
    gt_np = gt.detach().cpu().numpy()[0, 0]
    data_range = float(gt_np.max() - gt_np.min())
    if data_range < 1e-4:
        ssim_val = 1.0
    else:
        ssim_val = float(ssim(gt_np, pred_np, data_range=data_range))
    
    # Safe patch-level R2 calculation
    ss_res = torch.sum((gt - pred) ** 2)
    ss_tot = torch.sum((gt - torch.mean(gt)) ** 2)
    if ss_tot < 1e-4:
        r2 = torch.tensor(0.0)
    else:
        r2 = 1.0 - (ss_res / (ss_tot + eps))
    
    return {"RMSE": rmse.item(), "MAE": mae.item(), "SSIM": ssim_val, "R2": r2.item()}

def calculate_dataset_metrics(all_preds, all_gts, all_inputs=None, scale_factor=130.0, stand_aggregate_size=10):
    """
    Computes dataset-level global metrics (accumulated over all test pixels)
    as well as stand-level aggregated metrics (e.g. 10x10 pooling = 100m stand = 1 ha).
    Directly addresses scale-support and eliminates patch-level R2 averaging artifacts.
    """
    preds_norm = np.asarray(all_preds, dtype=np.float64)
    gts_norm = np.asarray(all_gts, dtype=np.float64)
    
    p_flat = preds_norm.ravel()
    t_flat = gts_norm.ravel()
    
    mae_norm = float(np.mean(np.abs(p_flat - t_flat)))
    rmse_norm = float(np.sqrt(np.mean((p_flat - t_flat) ** 2)))
    
    ss_res_global = np.sum((t_flat - p_flat) ** 2)
    ss_tot_global = np.sum((t_flat - np.mean(t_flat)) ** 2)
    r2_global = float(1.0 - ss_res_global / (ss_tot_global + 1e-10))
    
    # Stand-level aggregation (100m support, 1 hectare)
    # Reduces sub-pixel diffusion noise and matches forestry inventory plot scales
    if stand_aggregate_size is not None and preds_norm.ndim == 4:
        k = stand_aggregate_size
        p_tensor = torch.from_numpy(preds_norm)
        t_tensor = torch.from_numpy(gts_norm)
        p_stand = F.avg_pool2d(p_tensor, kernel_size=k, stride=k).numpy().ravel()
        t_stand = F.avg_pool2d(t_tensor, kernel_size=k, stride=k).numpy().ravel()
        
        rmse_stand_norm = float(np.sqrt(np.mean((t_stand - p_stand) ** 2)))
        mae_stand_norm = float(np.mean(np.abs(t_stand - p_stand)))
        ss_res_stand = np.sum((t_stand - p_stand) ** 2)
        ss_tot_stand = np.sum((t_stand - np.mean(t_stand)) ** 2)
        r2_stand = float(1.0 - ss_res_stand / (ss_tot_stand + 1e-10))
    else:
        rmse_stand_norm = rmse_norm
        mae_stand_norm = mae_norm
        r2_stand = r2_global

    return {
        "MAE_norm": mae_norm,
        "RMSE_norm": rmse_norm,
        "MAE_MgCha": mae_norm * scale_factor,
        "RMSE_MgCha": rmse_norm * scale_factor,
        "R2_global": r2_global,
        "Stand_100m_RMSE_MgCha": rmse_stand_norm * scale_factor,
        "Stand_100m_MAE_MgCha": mae_stand_norm * scale_factor,
        "Stand_100m_R2": r2_stand
    }
