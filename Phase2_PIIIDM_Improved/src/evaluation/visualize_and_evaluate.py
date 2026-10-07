import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from skimage.metrics import structural_similarity as ssim

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.models.cnn_baseline import CNNBaseline
from src.models.vae import CarbonVAE
from src.models.kd_vgg import LightweightStudentEncoder
from src.models.kd_unet import KDUNet
from src.models.diffusion import DiffusionScheduler
from src.models.pi_ldm import PILDM

C_MIN = 5.051
C_MAX = 198.261

def denorm(arr, vmin=C_MIN, vmax=C_MAX):
    return arr * (vmax - vmin) + vmin

def calculate_physics_metrics(pred, chm):
    dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dy_chm = chm[:, :, 1:, :] - chm[:, :, :-1, :]
    
    # -1.93 Mg/ha is exactly equivalent to -0.01 in the normalized [0,1] space
    vy = np.logical_and(dy_chm > 2.075, dy_pred < -1.93)
    
    dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dx_chm = chm[:, :, :, 1:] - chm[:, :, :, :-1]
    
    vx = np.logical_and(dx_chm > 2.075, dx_pred < -1.93)
    
    total_pixels = dy_pred.size + dx_pred.size
    total_violations = np.sum(vy) + np.sum(vx)
    pvr = (total_violations / total_pixels) * 100 if total_pixels > 0 else 0.0
    
    violating_dy = dy_pred[vy]
    violating_dx = dx_pred[vx]
    all_violating = np.concatenate([violating_dy, violating_dx])
    mnd = np.mean(np.abs(all_violating)) if len(all_violating) > 0 else 0.0
    
    return pvr, mnd

def calc_r2(pred, gt):
    # Ignore zero variance
    ss_tot = np.sum((gt - np.mean(gt))**2)
    if ss_tot < 1e-4:
        return np.nan
    ss_res = np.sum((gt - pred)**2)
    return 1 - (ss_res / ss_tot)

def calc_ssim(pred, gt):
    # Fixed data_range to global max-min for consistency
    return ssim(gt[0], pred[0], data_range=193.21)

class TestDataset(Dataset):
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

def load_pi_ldm(ckpt_dir, device, use_vgg=False):
    vae = CarbonVAE(in_channels=1, latent_channels=4, base_channels=64)
    if use_vgg:
        from src.models.kd_vgg import VGG19_Adapter
        encoder = VGG19_Adapter(in_channels=8)
    else:
        encoder = LightweightStudentEncoder(in_channels=8)
    unet = KDUNet(in_channels=4, out_channels=4, context_dim=256)
    scheduler = DiffusionScheduler(T=1000, device=device)
    
    vae_ckpt = ROOT / "results" / "checkpoints" / "vae_best.pt"
    if vae_ckpt.exists():
        vae.load_state_dict(torch.load(vae_ckpt, map_location="cpu"))
        
    model = PILDM(vae, encoder, unet, scheduler, freeze_encoder=True).to(device)
    
    if use_vgg:
        from src.models.implicit_repr import SIRENINR
        model.inr = SIRENINR(student_chs=[64, 128, 256, 512, 4]).to(device)
    else:
        from src.models.implicit_repr import SIRENINR
        model.inr = SIRENINR(student_chs=[32, 64, 128, 256, 4]).to(device)
    
    unet_ckpt = Path(ckpt_dir) / "unet_best.pt"
    if unet_ckpt.exists():
        model.unet.load_state_dict(torch.load(unet_ckpt, map_location=device))
        
    encoder_ckpt = Path(ckpt_dir) / "encoder_best.pt"
    if encoder_ckpt.exists():
        model.encoder.load_state_dict(torch.load(encoder_ckpt, map_location=device))
        
    inr_ckpt = Path(ckpt_dir) / "inr_best.pt"
    if inr_ckpt.exists():
        model.inr.load_state_dict(torch.load(inr_ckpt, map_location=device))
    
    model.eval()
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running on {device}")
    
    ds = TestDataset()
    # Use smaller batch size for speed and to avoid OOM
    batch_size = 4
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    
    print("Loading models...")
    cnn = CNNBaseline(in_channels=8).to(device)
    cnn.load_state_dict(torch.load(ROOT / "results" / "checkpoints" / "unet_baseline" / "unet_best.pt", map_location=device))
    cnn.eval()
    
    iidm = load_pi_ldm(ROOT / "results" / "checkpoints" / "pildm_seed42_L0.0_noKD", device, use_vgg=False)
    pi_iidm = load_pi_ldm(ROOT / "results" / "checkpoints" / "pildm_robust_v4", device, use_vgg=True)
    
    metrics = {
        "cnn": {"mae": [], "rmse": [], "r2": [], "ssim": [], "pvr": [], "mnd": []},
        "iidm": {"mae": [], "rmse": [], "r2": [], "ssim": [], "pvr": [], "mnd": []},
        "pi_iidm": {"mae": [], "rmse": [], "r2": [], "ssim": [], "pvr": [], "mnd": []}
    }
    
    print("Starting evaluation across all test patches...")
    saved_plot = False
    
    out_dir = ROOT / "results" / "figures"
    out_dir.mkdir(exist_ok=True, parents=True)
    
    for i, (inp, tgt) in enumerate(loader):
        inp = inp.to(device)
        B = inp.shape[0]
        
        chm = inp[:, 7:8, :, :].cpu().numpy() * 40.0
        gt_np = denorm(tgt.numpy())
        
        with torch.no_grad():
            cnn_pred = denorm(cnn(inp).cpu().numpy())
            
            # Monte Carlo Ensemble Inference for PI-IIDM (10 samples)
            ensemble_preds = []
            for _ in range(10):
                ensemble_preds.append(pi_iidm.predict(inp, num_steps=20).cpu().numpy())
            pi_iidm_pred = denorm(np.mean(ensemble_preds, axis=0))
            
            # For pure LDM (iidm), we must pass zero condition to avoid KD
            f_enc, f_proj, _ = iidm.encoder(inp)
            lat_hw = (32, 32)
            cond_zero = torch.zeros(B, 256, *lat_hw, device=device)
            z_T = torch.randn(B, 4, *lat_hw, device=device)
            z_0_pred = iidm.scheduler.ddim_sample(iidm.unet, z_T.shape, cond_zero, steps=20)
            iidm_pred = denorm(iidm.vae.decode(z_0_pred).cpu().numpy())
            
        preds = {"cnn": cnn_pred, "iidm": iidm_pred, "pi_iidm": pi_iidm_pred}
        
        # Calculate per-patch metrics
        for name, p in preds.items():
            err = np.abs(p - gt_np)
            for b in range(B):
                metrics[name]["mae"].append(np.mean(err[b]))
                metrics[name]["rmse"].append(np.sqrt(np.mean(err[b]**2)))
                metrics[name]["r2"].append(calc_r2(p[b], gt_np[b]))
                metrics[name]["ssim"].append(calc_ssim(p[b], gt_np[b]))
                pvr, mnd = calculate_physics_metrics(np.expand_dims(p[b], axis=0), np.expand_dims(chm[b], axis=0))
                metrics[name]["pvr"].append(pvr)
                metrics[name]["mnd"].append(mnd)
                
        # Generate Visualization Map
        if not saved_plot and i * batch_size >= 40:
            saved_plot = True
            b_idx = 0 
            
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            
            axes[0, 0].imshow(gt_np[b_idx,0], cmap='viridis', vmin=C_MIN, vmax=C_MAX)
            axes[0, 0].set_title("Ground Truth")
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(cnn_pred[b_idx,0], cmap='viridis', vmin=C_MIN, vmax=C_MAX)
            axes[0, 1].set_title("CNN Baseline")
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(iidm_pred[b_idx,0], cmap='viridis', vmin=C_MIN, vmax=C_MAX)
            axes[0, 2].set_title("IIDM Prediction")
            axes[0, 2].axis('off')
            
            axes[0, 3].imshow(pi_iidm_pred[b_idx,0], cmap='viridis', vmin=C_MIN, vmax=C_MAX)
            axes[0, 3].set_title("PI-IIDM Prediction")
            axes[0, 3].axis('off')
            
            err_vmax = 50.0
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(np.abs(cnn_pred[b_idx,0]-gt_np[b_idx,0]), cmap='magma', vmin=0, vmax=err_vmax)
            axes[1, 1].set_title("CNN Absolute Error")
            axes[1, 1].axis('off')
            
            axes[1, 2].imshow(np.abs(iidm_pred[b_idx,0]-gt_np[b_idx,0]), cmap='magma', vmin=0, vmax=err_vmax)
            axes[1, 2].set_title("IIDM Absolute Error")
            axes[1, 2].axis('off')
            
            axes[1, 3].imshow(np.abs(pi_iidm_pred[b_idx,0]-gt_np[b_idx,0]), cmap='magma', vmin=0, vmax=err_vmax)
            axes[1, 3].set_title("PI-IIDM Absolute Error")
            axes[1, 3].axis('off')
            
            plt.tight_layout()
            plt.savefig(out_dir / "visualization_maps.png", dpi=300)
            plt.close()
            print("Saved visualization maps.")
            
        if (i+1) % 5 == 0:
            print(f"Processed {(i+1)*batch_size} patches...")
            
    print("\n--- STATISTICAL REPORT ---")
    report = []
    
    for name in metrics:
        report.append(f"\n[{name.upper()}]")
        for m in ["mae", "rmse", "r2", "ssim", "pvr", "mnd"]:
            report.append(f"{m.upper()}: {np.nanmean(metrics[name][m]):.4f}")
            
    report.append("\n--- STATISTICAL SIGNIFICANCE (PI-IIDM vs IIDM) ---")
    
    for m in ["mae", "rmse", "r2", "ssim"]:
        arr_iidm = np.array(metrics["iidm"][m])
        arr_pi = np.array(metrics["pi_iidm"][m])
        
        t_stat, p_val = stats.ttest_rel(arr_pi, arr_iidm)
        diff = arr_pi - arr_iidm
        mean_diff = np.mean(diff)
        sem = stats.sem(diff)
        ci_lower = mean_diff - 1.96 * sem
        ci_upper = mean_diff + 1.96 * sem
        
        std_diff = np.std(diff, ddof=1)
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0
        
        report.append(f"\nMetric: {m.upper()}")
        report.append(f"Mean Difference (PI-IIDM - IIDM): {mean_diff:.4f}")
        report.append(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        report.append(f"p-value: {p_val:.4e}")
        report.append(f"Effect Size (Cohen's d): {cohens_d:.4f}")
        
    full_report = "\n".join(report)
    print(full_report)
    with open(out_dir / "statistical_analysis.txt", "w") as f:
        f.write(full_report)

if __name__ == "__main__":
    main()
