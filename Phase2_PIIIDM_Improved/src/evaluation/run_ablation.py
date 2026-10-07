import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from scipy import stats

import sys
import os
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from torch.utils.data import DataLoader, Dataset
import glob

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
from src.models.cnn_baseline import CNNBaseline
from src.models.vae import CarbonVAE
from src.models.kd_vgg import LightweightStudentEncoder, VGG19_Adapter
from src.models.kd_unet import KDUNet
from src.models.diffusion import DiffusionScheduler
from src.models.pi_ldm import PILDM
from src.models.implicit_repr import SIRENINR
from skimage.metrics import structural_similarity as ssim

def compute_pvr_mnd(pred, chm):
    dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dy_chm = chm[:, :, 1:, :] - chm[:, :, :-1, :]
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
    ss_tot = np.sum((gt - np.mean(gt))**2)
    if ss_tot < 1e-4:
        return np.nan
    ss_res = np.sum((gt - pred)**2)
    return 1 - (ss_res / ss_tot)

def calc_ssim(pred, gt):
    return ssim(gt, pred, data_range=193.21)

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
        vae.load_state_dict(torch.load(vae_ckpt, map_location="cpu", weights_only=True))
        
    model = PILDM(vae, encoder, unet, scheduler, freeze_encoder=True).to(device)
    
    if use_vgg:
        model.inr = SIRENINR(student_chs=[64, 128, 256, 512, 4]).to(device)
    else:
        model.inr = SIRENINR(student_chs=[32, 64, 128, 256, 4]).to(device)
    
    unet_ckpt = Path(ckpt_dir) / "unet_best.pt"
    if unet_ckpt.exists():
        model.unet.load_state_dict(torch.load(unet_ckpt, map_location=device, weights_only=True))
        
    encoder_ckpt = Path(ckpt_dir) / "encoder_best.pt"
    if encoder_ckpt.exists():
        model.encoder.load_state_dict(torch.load(encoder_ckpt, map_location=device, weights_only=True))
        
    inr_ckpt = Path(ckpt_dir) / "inr_best.pt"
    if inr_ckpt.exists():
        model.inr.load_state_dict(torch.load(inr_ckpt, map_location=device, weights_only=True))
    
    model.eval()
    return model

C_MIN = 5.051
C_MAX = 198.261

def denorm(arr, vmin=C_MIN, vmax=C_MAX):
    return arr * (vmax - vmin) + vmin

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
    
    ds = TestDataset()
    batch_size = 4
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    
    print("Loading models...")
    models = {}
    
    # A. No diffusion + no physics (CNN Baseline)
    cnn = CNNBaseline(in_channels=8).to(device)
    cnn_ckpt = ROOT / "results" / "checkpoints" / "unet_baseline" / "unet_best.pt"
    if cnn_ckpt.exists():
        cnn.load_state_dict(torch.load(cnn_ckpt, map_location=device, weights_only=True))
    cnn.eval()
    models['A_CNN'] = cnn
    
    # B. Diffusion without physics (IIDM baseline)
    models['B_IIDM'] = load_pi_ldm(ROOT / "results" / "checkpoints" / "pildm_seed42_L0.0_noKD", device, use_vgg=False)
    
    # C. Diffusion + physics (PI-IIDM, no KD)
    models['C_PI_IIDM_noKD'] = load_pi_ldm(ROOT / "results" / "checkpoints" / "pildm_seed42_L0.5_noKD", device, use_vgg=False)
    
    # E. Full PI-LDM + KD (The massive 26hr run)
    models['E_Full_PI_IIDM'] = load_pi_ldm(ROOT / "results" / "checkpoints" / "pildm_robust_v4", device, use_vgg=True)
    
    metrics = {k: {"mae": [], "rmse": [], "r2": [], "ssim": [], "pvr": [], "mnd": []} for k in models.keys()}
    
    out_dir = ROOT / "results" / "figures"
    out_dir.mkdir(exist_ok=True, parents=True)
    
    saved_plot = False
    
    with torch.no_grad():
        for i, (inp, tgt) in enumerate(loader):
            if i % 10 == 0:
                print(f"Processing batch {i}...")
            inp = inp.to(device)
            B = inp.shape[0]
            chm = inp[:, 7:8, :, :].cpu().numpy() * 40.0
            gt = denorm(tgt.cpu().numpy())
            
            preds = {}
            for k, model in models.items():
                if 'CNN' in k:
                    pred = denorm(model(inp).cpu().numpy())
                elif 'B_IIDM' == k:
                    f_enc, f_proj, _ = model.encoder(inp)
                    lat_hw = (32, 32)
                    cond_zero = torch.zeros(B, 256, *lat_hw, device=device)
                    z_T = torch.randn(B, 4, *lat_hw, device=device)
                    z_0_pred = model.scheduler.ddim_sample(model.unet, z_T.shape, cond_zero, steps=20)
                    pred = denorm(model.vae.decode(z_0_pred).cpu().numpy())
                else:
                    pred = denorm(model.predict(inp, num_steps=20).cpu().numpy())
                preds[k] = pred
                
                for b in range(B):
                    p = pred[b, 0]
                    g = gt[b, 0]
                    c = chm[b, 0]
                    
                    m_mae = np.mean(np.abs(p - g))
                    m_rmse = np.sqrt(np.mean((p - g)**2))
                    m_r2 = calc_r2(p, g)
                    m_ssim = calc_ssim(p, g)
                    
                    p_pvr, p_mnd = compute_pvr_mnd(np.expand_dims(np.expand_dims(p, 0), 0), np.expand_dims(np.expand_dims(c, 0), 0))
                    
                    metrics[k]["mae"].append(m_mae)
                    metrics[k]["rmse"].append(m_rmse)
                    metrics[k]["r2"].append(m_r2)
                    metrics[k]["ssim"].append(m_ssim)
                    metrics[k]["pvr"].append(p_pvr)
                    metrics[k]["mnd"].append(p_mnd)
                    
            if not saved_plot:
                saved_plot = True
                fig, axes = plt.subplots(3, 4, figsize=(20, 15))
                
                b_idx = 0
                gt_img = gt[b_idx, 0]
                
                # Top Row: Ground Truth and Predictions
                axes[0, 0].imshow(gt_img, cmap='viridis')
                axes[0, 0].set_title('1. Ground Truth')
                
                axes[0, 1].imshow(preds['A_CNN'][b_idx, 0], cmap='viridis')
                axes[0, 1].set_title('2. CNN (No Diff, No Phys)')
                
                axes[0, 2].imshow(preds['B_IIDM'][b_idx, 0], cmap='viridis')
                axes[0, 2].set_title('3. IIDM (Diff, No Phys)')
                
                axes[0, 3].imshow(preds['E_Full_PI_IIDM'][b_idx, 0], cmap='viridis')
                axes[0, 3].set_title('4. Full PI-IIDM')
                
                # Middle Row: Error Maps
                vmin, vmax = 0, np.max(np.abs(gt_img - preds['A_CNN'][b_idx, 0]))
                
                axes[1, 0].axis('off')
                
                err_cnn = np.abs(gt_img - preds['A_CNN'][b_idx, 0])
                im = axes[1, 1].imshow(err_cnn, cmap='hot', vmin=0, vmax=vmax)
                axes[1, 1].set_title('5. CNN Error')
                
                err_iidm = np.abs(gt_img - preds['B_IIDM'][b_idx, 0])
                axes[1, 2].imshow(err_iidm, cmap='hot', vmin=0, vmax=vmax)
                axes[1, 2].set_title('6. IIDM Error')
                
                err_pi = np.abs(gt_img - preds['E_Full_PI_IIDM'][b_idx, 0])
                axes[1, 3].imshow(err_pi, cmap='hot', vmin=0, vmax=vmax)
                axes[1, 3].set_title('7. Full PI-IIDM Error')
                
                # Bottom Row: C vs E Error comparison (just to fill space if needed)
                axes[2, 0].axis('off')
                axes[2, 1].axis('off')
                err_c = np.abs(gt_img - preds['C_PI_IIDM_noKD'][b_idx, 0])
                axes[2, 2].imshow(err_c, cmap='hot', vmin=0, vmax=vmax)
                axes[2, 2].set_title('PI-IIDM (no KD) Error')
                axes[2, 3].axis('off')
                
                for ax in axes.flatten():
                    if ax.has_data():
                        ax.axis('off')
                        
                plt.tight_layout()
                plt.savefig(out_dir / "ablation_7pane_maps.png", dpi=150, bbox_inches='tight')
                plt.close()
                
    report = "=== ABLATION STUDY RESULTS ===\n\n"
    for k in models.keys():
        report += f"[{k}]\n"
        for metric in ["mae", "rmse", "r2", "ssim", "pvr", "mnd"]:
            report += f"{metric.upper()}: {np.nanmean(metrics[k][metric]):.4f}\n"
        report += "\n"
        
    report += "=== STATISTICAL SIGNIFICANCE (Independent Spatial Units) ===\n\n"
    
    baseline = 'B_IIDM'
    proposed = 'E_Full_PI_IIDM'
    report += f"Comparing {proposed} vs {baseline}\n\n"
    
    for metric in ["mae", "rmse", "ssim"]:
        arr_b = np.array(metrics[baseline][metric])
        arr_p = np.array(metrics[proposed][metric])
        
        diff = arr_p - arr_b
        mean_diff = np.nanmean(diff)
        
        t_stat, p_val = stats.ttest_rel(arr_p, arr_b, nan_policy='omit')
        
        std_diff = np.nanstd(diff, ddof=1)
        n = len(diff)
        ci_margin = stats.t.ppf(0.975, n-1) * (std_diff / np.sqrt(n))
        ci_lower = mean_diff - ci_margin
        ci_upper = mean_diff + ci_margin
        
        sd_pooled = np.sqrt((np.nanstd(arr_p, ddof=1)**2 + np.nanstd(arr_b, ddof=1)**2) / 2)
        cohen_d = mean_diff / sd_pooled
        
        report += f"Metric: {metric.upper()}\n"
        report += f"Mean Difference ({proposed} - {baseline}): {mean_diff:.4f}\n"
        report += f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]\n"
        report += f"p-value: {p_val:.4e}\n"
        report += f"Effect Size (Cohen's d): {cohen_d:.4f}\n\n"

    print(report)
    with open(out_dir / "ablation_stats.txt", "w") as f:
        f.write(report)

if __name__ == "__main__":
    main()
