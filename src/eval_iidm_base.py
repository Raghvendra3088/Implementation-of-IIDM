import os
import sys
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

torch.backends.cudnn.enabled = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train_iidm_base import PatchDataset
from src.models.iidm_encoder import VGG16Teacher, LightweightStudentEncoder
from src.models.iidm_diffusion import ConditionalLatentDiffusion
from src.models.iidm_inr import ImplicitNeuralDecoder, generate_grid_coords

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--patch_dir', default='data/processed/patches_v2')
    p.add_argument('--ckpt', default='checkpoints/base_iidm_paper.pth')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--n_steps', type=int, default=20)
    p.add_argument('--ablation', type=str, default='none', choices=['none', 'no_kd', 'no_diffusion', 'no_inr'])
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    test_ds = PatchDataset(args.patch_dir, 'test')
    test_ld = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"Test: {len(test_ds)} patches")

    print(f"Loading Models... [Ablation: {args.ablation}]")
    teacher = VGG16Teacher(in_channels=4).to(device)
    student = LightweightStudentEncoder(in_channels=4).to(device)
    diffusion = ConditionalLatentDiffusion(T=1000).to(device)
    
    if args.ablation == 'no_inr':
        from src.models.iidm_inr import StandardDecoder
        inr_decoder = StandardDecoder(latent_dim=256).to(device)
    else:
        inr_decoder = ImplicitNeuralDecoder(latent_dim=256, num_freqs=10).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    student.load_state_dict(ckpt['student'])
    diffusion.load_state_dict(ckpt['diffusion'])
    inr_decoder.load_state_dict(ckpt['inr'])
    print(f"Loaded checkpoint from {args.ckpt}")

    teacher.eval(); student.eval(); diffusion.eval(); inr_decoder.eval()
    
    for m in [teacher, student, diffusion, inr_decoder]:
        for param in m.parameters(): param.requires_grad = False

    grid_coords = generate_grid_coords(64, 64, device)

    C_MIN = 4.816495895385742
    C_MAX = 129.18380737304688

    all_gt, all_pred = [], []
    with torch.no_grad():
        for bi, (x, y0, mask) in enumerate(test_ld):
            x, y0, mask = x.to(device), y0.to(device), mask.to(device)
            B = x.shape[0]

            t_feats = teacher(x)
            _, _, z_0 = student(x)
            cond = t_feats[-1]

            # 20-step DDIM sample or skip
            if args.ablation == 'no_diffusion':
                z_star = z_0
            else:
                z_star = diffusion.sample(z_0, cond, ddim_steps=args.n_steps)

            coords_b = grid_coords.expand(B, -1, -1)
            y0_pred = inr_decoder(z_star, coords_b)
            y0_pred = y0_pred.view(B, 1, 64, 64).clamp(-1, 1)

            pred_mg = (y0_pred.cpu().numpy() * 0.5 + 0.5) * (C_MAX - C_MIN) + C_MIN
            gt_mg   = (y0.cpu().numpy() * 0.5 + 0.5) * (C_MAX - C_MIN) + C_MIN
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
    print(f"BASE IIDM PAPER TEST RESULTS (DDIM, n_steps={args.n_steps})")
    print("=" * 50)
    print(f"Test RMSE : {rmse:.4f} Mg C/ha")
    print(f"Test MAE  : {mae:.4f} Mg C/ha")
    print(f"Test R2   : {r2:.4f}")
    print("=" * 50)

if __name__ == '__main__':
    main()
