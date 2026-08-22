import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim

def calculate_metrics(pred, gt):
    mse = F.mse_loss(pred, gt)
    rmse = torch.sqrt(mse)
    mae = torch.mean(torch.abs(pred - gt))
    
    # SSIM requires numpy array
    pred_np = pred.detach().cpu().numpy()[0, 0]
    gt_np = gt.detach().cpu().numpy()[0, 0]
    ssim_val = ssim(gt_np, pred_np, data_range=gt_np.max() - gt_np.min())
    
    # R^2 calculation
    ss_res = torch.sum((gt - pred)**2)
    ss_tot = torch.sum((gt - torch.mean(gt))**2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {"RMSE": rmse.item(), "MAE": mae.item(), "SSIM": ssim_val, "R2": r2.item()}
