import os, sys, json
import torch
torch.backends.cudnn.enabled = False
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train_v2_ldm import PatchDataset, make_schedule
from src.models.vae import CarbonVAE
from src.models.prithvi_teacher import KDStudent12
from src.models.latent_kd_unet import LatentKDUNet12

@torch.no_grad()
def latent_ddim_sample(unet, vae, student_feats, scale_factor, alpha_bar, T, device, n_steps=20, seed=42):
    B = student_feats[0].shape[0]
    # For 64x64 input patches, Latent space is 8x8 with 4 channels
    H_z, W_z = 8, 8 
    
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    z_t = torch.randn(B, 4, H_z, W_z, device=device, generator=gen)

    step_size = T // n_steps
    timesteps = list(range(T, 0, -step_size))
    if timesteps[-1] != 1:
        timesteps.append(1)

    for i, t in enumerate(timesteps):
        t_next = timesteps[i + 1] if i + 1 < len(timesteps) else 0

        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
        ab_t   = alpha_bar[t - 1]
        ab_tm1 = alpha_bar[t_next - 1] if t_next > 0 else torch.ones(1, device=device)

        eps_pred = unet(z_t, t_tensor, student_feats)

        z0_pred = (z_t - (1 - ab_t).sqrt() * eps_pred) / (ab_t.sqrt() + 1e-8)

        if t_next > 0:
            z_t = ab_tm1.sqrt() * z0_pred + (1 - ab_tm1).sqrt() * eps_pred
        else:
            z_t = z0_pred

    # Decode latent z back to pixel y
    y0_pred = vae.decode(z_t / scale_factor)
    return y0_pred.clamp(-1, 1)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--patch_dir',  default='data/processed/patches_v2')
    p.add_argument('--unet_ckpt',  default='checkpoints/latent_kd_unet.pth')
    p.add_argument('--student_ckpt', default='checkpoints/prithvi_blockwise_kd.pth')
    p.add_argument('--vae_ckpt',   default='checkpoints/vae.pth')
    p.add_argument('--vae_scale_file', default='checkpoints/vae_scale_factor.txt')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--T',          type=int, default=1000)
    p.add_argument('--n_steps',    type=int, default=20)
    p.add_argument('--seed',       type=int, default=42)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    test_ds = PatchDataset(args.patch_dir, 'test')
    test_ld = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"Test: {len(test_ds)} patches")

    print("Loading Models...")
    vae = CarbonVAE().to(device)
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location=device))
    
    try:
        with open(args.vae_scale_file, 'r') as f:
            scale_factor = float(f.read().strip())
    except:
        scale_factor = 1.0
        
    student = KDStudent12(in_channels=4).to(device)
    s_ckpt = torch.load(args.student_ckpt, map_location=device)
    student.load_state_dict(s_ckpt['student'])
    
    unet = LatentKDUNet12(in_channels=4).to(device)
    unet.load_state_dict(torch.load(args.unet_ckpt, map_location=device))

    vae.eval(); student.eval(); unet.eval()
    
    for model in [vae, student, unet]:
        for param in model.parameters():
            param.requires_grad = False

    betas, alpha_bar = make_schedule(args.T, device)

    C_MIN = 0.0
    C_MAX = 130.0

    all_gt, all_pred = [], []
    with torch.no_grad():
        for bi, (x, y0, mask) in enumerate(test_ld):
            x, y0, mask = x.to(device), y0.to(device), mask.to(device)
            B, _, H, W = y0.shape

            s_feats = student(x)

            y0_pred = latent_ddim_sample(unet, vae, s_feats, scale_factor, alpha_bar, args.T,
                                         device, n_steps=args.n_steps, seed=args.seed)

            pred_mg = ((y0_pred.cpu().numpy() + 1) / 2) * (C_MAX - C_MIN) + C_MIN
            gt_mg   = ((y0.cpu().numpy() + 1) / 2) * (C_MAX - C_MIN) + C_MIN
            m_np    = mask.cpu().numpy().astype(bool)

            all_pred.append(pred_mg[m_np])
            all_gt.append(gt_mg[m_np])

            if (bi + 1) % 10 == 0:
                print(f"  batch {bi+1}/{len(test_ld)}", flush=True)

    all_pred = np.concatenate(all_pred)
    all_gt   = np.concatenate(all_gt)

    rmse = float(np.sqrt(np.mean((all_pred - all_gt) ** 2)))
    mae  = float(np.mean(np.abs(all_pred - all_gt)))
    ss_r = np.sum((all_gt - all_pred) ** 2)
    ss_t = np.sum((all_gt - all_gt.mean()) ** 2)
    r2   = float(1 - ss_r / (ss_t + 1e-8))

    print("\n" + "=" * 50)
    print(f"IIDM-V2 LATENT TEST RESULTS (DDIM, n_steps={args.n_steps})")
    print("=" * 50)
    print(f"Test RMSE : {rmse:.4f} Mg C/ha")
    print(f"Test MAE  : {mae:.4f} Mg C/ha")
    print(f"Test R2   : {r2:.4f}")
    print("=" * 50)

if __name__ == '__main__':
    main()
