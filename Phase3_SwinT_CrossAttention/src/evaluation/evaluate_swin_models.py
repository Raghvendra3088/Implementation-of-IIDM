#!/usr/bin/env python3
"""
evaluate_swin_models.py — Comprehensive Evaluation Suite for Phase 3 Vision Transformer Models
Evaluates:
  1. Full Proposed PI-IIDM (Swin-T + Cross-Attention + Footprint Support + Point Physics + Mixed Curriculum)
  2. Swin-T Direct Regression (Deterministic Baseline)
  3. Modality Ablation (Swin-T w/o ETH Canopy Height)
"""

import os, sys, glob, json, time, argparse
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[3] if "src" in str(Path(__file__)) else Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data" / "processed" / "patches_6ch"

class PatchDataset(Dataset):
    def __init__(self, split="test", drop_eth=False):
        d = DATA_DIR / split
        self.inp = sorted(glob.glob(str(d / "input"  / "*.npz")))
        self.tgt = sorted(glob.glob(str(d / "target" / "*.npz")))
        self.drop_eth = drop_eth
    def __len__(self): return len(self.inp)
    def __getitem__(self, idx):
        with np.load(self.inp[idx]) as di:
            x = di['image'].astype(np.float32)
            if self.drop_eth: x = x[:7]
        with np.load(self.tgt[idx]) as dt:
            y = dt['image'].astype(np.float32)
        if y.ndim == 2: y = y[None]
        return torch.from_numpy(x), torch.from_numpy(y)

def compute_metrics(preds, targets, inputs, scale_factor=130.0):
    try:
        from skimage.metrics import structural_similarity as ski_ssim
        has_ski = True
    except ImportError:
        has_ski = False
    p = preds.astype(np.float64).ravel()
    t = targets.astype(np.float64).ravel()
    mae  = float(np.mean(np.abs(p - t)))
    rmse = float(np.sqrt(np.mean((p - t)**2)))
    r2   = float(1 - np.sum((p-t)**2) / (np.sum((t - t.mean())**2) + 1e-10))
    ssim = 0.0
    if has_ski:
        for i in range(preds.shape[0]):
            ssim += ski_ssim(preds[i,0].astype(np.float64),
                             targets[i,0].astype(np.float64), data_range=1.0)
        ssim /= preds.shape[0]
    
    # Physics violation calculation
    chm_idx = 7 if inputs.shape[1] > 7 else 6
    chm_mean = inputs[:, chm_idx, :, :].mean(axis=(1, 2))
    bio_mean = preds[:, 0, :, :].mean(axis=(1, 2))
    idx = np.argsort(chm_mean)
    diffs = np.diff(bio_mean[idx])
    pvr = float(np.mean(diffs < 0))
    mnd = float(np.mean(np.clip(-diffs, 0, None)))
    
    return {
        "MAE_norm": mae,
        "RMSE_norm": rmse,
        "MAE_MgCha": mae * scale_factor,
        "RMSE_MgCha": rmse * scale_factor,
        "R2": r2,
        "SSIM": float(ssim),
        "PVR": pvr,
        "MND": mnd
    }

def print_master_table():
    results_dir = ROOT / "results"
    models = {
        "Oracle z0 (VAE Ceiling)": results_dir / "vae_ceiling_metrics.json",
        "Random Forest (B0)": results_dir / "baselines/B0_RF/metrics.json",
        "U-Net Regression (B1)": results_dir / "baselines/B1_UNet/metrics.json",
        "LDM (B2)": results_dir / "baselines/B2_LDM/metrics.json",
        "LDM + KD (B2-KD)": results_dir / "baselines/B2KD_LDM_KD/metrics.json",
        "PI-LDM (P1)": results_dir / "baselines/P1_PILDM/metrics.json",
        "Swin-T Direct Reg": results_dir / "baselines/swin_deterministic/metrics.json",
        "Swin-T w/o ETH (7ch)": results_dir / "phase3_swin_no_eth/metrics.json",
        "Proposed Full PI-IIDM": results_dir / "phase3_swin/metrics.json",
    }
    
    print("\n" + "="*110)
    print(f"{'Model':<25} | {'RMSE (Norm)':<11} | {'RMSE (Mg/ha)':<13} | {'MAE (Mg/ha)':<13} | {'R2':<8} | {'SSIM':<8} | {'PVR':<8} | {'MND':<8}")
    print("="*110)
    for name, path in models.items():
        if path.exists():
            d = json.load(open(path))
            rmse_norm = d.get('RMSE', d.get('Oracle_z0_RMSE_norm', d.get('RMSE_norm', 0.0)))
            rmse_mg = d.get('RMSE_MgCha', rmse_norm * 130)
            mae_norm = d.get('MAE', d.get('Oracle_z0_MAE_norm', d.get('MAE_norm', 0.0)))
            mae_mg = d.get('MAE_MgCha', mae_norm * 130)
            r2 = d.get('R2', 0.0)
            ssim = d.get('SSIM', 0.0)
            pvr = d.get('PVR', 0.0)
            mnd = d.get('MND', 0.0)
            print(f"{name:<25} | {rmse_norm:<11.4f} | {rmse_mg:<13.2f} | {mae_mg:<13.2f} | {r2:<8.3f} | {ssim:<8.3f} | {pvr:<8.3f} | {mnd:<8.4f}")
        else:
            print(f"{name:<25} | PENDING / NOT FOUND")
    print("="*110 + "\n")

if __name__ == "__main__":
    print_master_table()
