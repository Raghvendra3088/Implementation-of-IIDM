import os
import glob
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.models.cnn_baseline import CNNBaseline

def rmse(pred, gt):
    return float(np.sqrt(np.mean((pred - gt) ** 2)))

def mae(pred, gt):
    return float(np.mean(np.abs(pred - gt)))

def denorm(arr, vmin, vmax):
    return arr * (vmax - vmin) + vmin

def calculate_physics_violation_rate(pred, chm):
    dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    
    dy_chm = chm[:, :, 1:, :] - chm[:, :, :-1, :]
    dx_chm = chm[:, :, :, 1:] - chm[:, :, :, :-1]
    
    vy = np.logical_and(dy_chm > 0.05, dy_pred < -0.01)
    vx = np.logical_and(dx_chm > 0.05, dx_pred < -0.01)
    
    total_valid = np.sum(dy_chm > 0.05) + np.sum(dx_chm > 0.05)
    total_violations = np.sum(vy) + np.sum(vx)
    
    if total_valid == 0:
        return 0.0
    return float(total_violations / total_valid) * 100.0

class PILdmTestDataset(Dataset):
    def __init__(self):
        self.data_dir = ROOT / "data" / "processed" / "patches_6ch" / "test"
        self.input_files = sorted(glob.glob(str(self.data_dir / "input" / "*.npz")))
        self.target_files = sorted(glob.glob(str(self.data_dir / "target" / "*.npz")))
            
    def __len__(self):
        return len(self.input_files)
        
    def __getitem__(self, idx):
        with np.load(self.input_files[idx]) as d_in, np.load(self.target_files[idx]) as d_tgt:
            inp = d_in['image']
            tgt = d_tgt['image']
        if len(tgt.shape) == 2:
            tgt = np.expand_dims(tgt, axis=0)
        return torch.from_numpy(inp).float(), torch.from_numpy(tgt).float()

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Evaluating {args.ckpt_dir} on {device}")
    
    with open(ROOT / "configs" / "norm_stats.json") as f:
        stats = json.load(f)
        C_MIN, C_MAX = stats["target"]["min"], stats["target"]["max"]
        
    test_dataset = PILdmTestDataset()
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)
    
    model = CNNBaseline(in_channels=8).to(device)
    
    ckpt_path = Path(args.ckpt_dir) / "unet_best.pt"
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return
        
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    all_preds = []
    all_gts = []
    all_chms = []
    
    with torch.no_grad():
        for inp, tgt in test_loader:
            inp = inp.to(device)
            tgt = tgt.to(device)
            
            y_pred = model(inp)
            
            all_preds.append(y_pred.cpu().numpy())
            all_gts.append(tgt.cpu().numpy())
            # CHM is channel 7
            all_chms.append(inp[:, 7:8, :, :].cpu().numpy())
            
    all_preds = np.concatenate(all_preds, axis=0)
    all_gts = np.concatenate(all_gts, axis=0)
    all_chms = np.concatenate(all_chms, axis=0)
    
    pred_denorm = denorm(all_preds, C_MIN, C_MAX)
    gt_denorm = denorm(all_gts, C_MIN, C_MAX)
    
    r = rmse(pred_denorm, gt_denorm)
    m = mae(pred_denorm, gt_denorm)
    nrmse = (r / (C_MAX - C_MIN)) * 100.0
    viol = calculate_physics_violation_rate(all_preds, all_chms)
    
    print(f"Results for {args.ckpt_dir}:")
    print(f"  RMSE: {r:.2f} Mg C/ha")
    print(f"  MAE:  {m:.2f} Mg C/ha")
    print(f"  nRMSE:{nrmse:.2f}%")
    print(f"  PhysViol: {viol:.2f}%")
    
    # save to json
    res_path = Path(args.ckpt_dir) / "metrics.json"
    with open(res_path, "w") as f:
        json.dump({"rmse": r, "mae": m, "nrmse": nrmse, "phys_viol": viol}, f, indent=2)
    
    return r, m, nrmse, viol

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    args = parser.parse_args()
    
    evaluate(args)
