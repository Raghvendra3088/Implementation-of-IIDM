import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

torch.backends.cudnn.enabled = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.iidm_encoder import VGG16Teacher, LightweightStudentEncoder, HierarchicalKDLoss
from src.models.iidm_diffusion import ConditionalLatentDiffusion
from src.models.iidm_inr import ImplicitNeuralDecoder, StandardDecoder, generate_grid_coords

class PatchDataset(Dataset):
    def __init__(self, root, split):
        import glob
        self.split = split
        split_dir = os.path.join(root, split)
        self.files = sorted(glob.glob(os.path.join(split_dir, '*.npz')))
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .npz files in {split_dir}")
        print(f"  {split}: {len(self.files)} patches")

    def __len__(self): return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i])
        x = torch.from_numpy(d['image'])
        y = torch.from_numpy(d['carbon']).unsqueeze(0)
        m = torch.from_numpy(d['mask']).unsqueeze(0)
        
        if self.split == 'train':
            if torch.rand(1) > 0.5:
                x = torch.flip(x, dims=[2])
                y = torch.flip(y, dims=[2])
                m = torch.flip(m, dims=[2])
            if torch.rand(1) > 0.5:
                x = torch.flip(x, dims=[1])
                y = torch.flip(y, dims=[1])
                m = torch.flip(m, dims=[1])
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                x = torch.rot90(x, k, dims=[1, 2])
                y = torch.rot90(y, k, dims=[1, 2])
                m = torch.rot90(m, k, dims=[1, 2])

        return x, y, m

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--patch_dir', default='data/processed/patches_v2')
    p.add_argument('--epochs', type=int, default=250)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--save_path', default='checkpoints/base_iidm_paper.pth')
    p.add_argument('--lambda_kd', type=float, default=1.0)
    p.add_argument('--lambda_diff', type=float, default=1.0)
    p.add_argument('--lambda_recon', type=float, default=10.0)
    p.add_argument('--ablation', type=str, default='none', choices=['none', 'no_kd', 'no_diffusion', 'no_inr'])
    args = p.parse_args()

    # Apply ablation overrides
    if args.ablation == 'no_kd':
        args.lambda_kd = 0.0
    if args.ablation == 'no_diffusion':
        args.lambda_diff = 0.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    print(f"Loading Models... [Ablation: {args.ablation}]")
    teacher = VGG16Teacher(in_channels=4).to(device)
    student = LightweightStudentEncoder(in_channels=4).to(device)
    kd_loss_fn = HierarchicalKDLoss().to(device)
    diffusion = ConditionalLatentDiffusion(T=1000).to(device)
    
    if args.ablation == 'no_inr':
        inr_decoder = StandardDecoder(latent_dim=256).to(device)
    else:
        inr_decoder = ImplicitNeuralDecoder(latent_dim=256, num_freqs=10).to(device)

    # Only optimize student, diffusion, and INR
    optimizer = torch.optim.Adam(
        list(student.parameters()) + list(diffusion.parameters()) + list(inr_decoder.parameters()),
        lr=args.lr
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    train_ds = PatchDataset(args.patch_dir, 'train')
    val_ds   = PatchDataset(args.patch_dir, 'val')
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_ld   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    best_val_loss = float('inf')

    # Pre-generate coordinates for H=64, W=64
    grid_coords = generate_grid_coords(64, 64, device)

    print(f"Starting End-to-End Training for {args.epochs} epochs")

    for epoch in range(1, args.epochs + 1):
        student.train()
        diffusion.train()
        inr_decoder.train()
        
        train_loss = 0.0
        train_recon = 0.0
        train_diff = 0.0
        train_kd = 0.0

        for x, y, m in train_ld:
            x, y, m = x.to(device), y.to(device), m.to(device)
            B = x.shape[0]

            optimizer.zero_grad()

            # 1. Forward Teacher & Student
            with torch.no_grad():
                t_feats = teacher(x)
            
            s_feats, p_feats, z_0 = student(x)
            
            # The condition for diffusion is the deep semantic feature from the teacher
            cond = t_feats[-1].detach()

            # 2. Hierarchical KD Loss
            l_kd = kd_loss_fn(p_feats, t_feats)

            # 3. Diffusion Loss
            l_diff = diffusion(z_0, cond)

            # 4. INR Reconstruction Loss
            # coords: (B, N, 2)
            coords_b = grid_coords.expand(B, -1, -1)
            # Predict using clean z_0 during training (acts as the generator target)
            y_pred = inr_decoder(z_0, coords_b) # (B, N, 1)
            
            # Reshape y and m to (B, N, 1)
            y_flat = y.view(B, -1, 1)
            m_flat = m.view(B, -1, 1)
            
            # Compute MSE only on valid pixels
            valid_mask = m_flat > 0
            if valid_mask.sum() > 0:
                l_recon = F.mse_loss(y_pred[valid_mask], y_flat[valid_mask])
            else:
                l_recon = torch.tensor(0.0, device=device)

            # Total Loss
            loss = args.lambda_recon * l_recon + args.lambda_diff * l_diff + args.lambda_kd * l_kd
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_recon += l_recon.item()
            train_diff += l_diff.item()
            train_kd += l_kd.item()

        avg_loss = train_loss / len(train_ld)
        avg_recon = train_recon / len(train_ld)
        avg_diff = train_diff / len(train_ld)
        avg_kd = train_kd / len(train_ld)
        
        # Step the learning rate scheduler
        scheduler.step()

        # Validation (using exact DDIM sampling + INR decoding)
        student.eval()
        diffusion.eval()
        inr_decoder.eval()
        
        val_loss = 0.0
        val_rmse = 0.0
        
        C_MIN = 4.816495895385742
        C_MAX = 129.18380737304688

        with torch.no_grad():
            for x, y, m in val_ld:
                x, y, m = x.to(device), y.to(device), m.to(device)
                B = x.shape[0]
                
                t_feats = teacher(x)
                s_feats, p_feats, z_0 = student(x)
                cond = t_feats[-1]
                
                # Sample z* using DDIM (20 steps for speed during validation)
                if args.ablation == 'no_diffusion':
                    z_star = z_0
                else:
                    z_star = diffusion.sample(z_0, cond, ddim_steps=20)
                
                # Decode using INR
                coords_b = grid_coords.expand(B, -1, -1)
                y_pred = inr_decoder(z_star, coords_b) # (B, N, 1)
                y_pred = y_pred.view(B, 1, 64, 64)
                
                # Compute Loss
                valid_mask = m > 0
                if valid_mask.sum() > 0:
                    v_loss = F.mse_loss(y_pred[valid_mask], y[valid_mask])
                    val_loss += v_loss.item()
                    
                    # Compute Est RMSE
                    pred_mg = (y_pred * 0.5 + 0.5) * (C_MAX - C_MIN) + C_MIN
                    gt_mg   = (y * 0.5 + 0.5) * (C_MAX - C_MIN) + C_MIN
                    
                    se = (pred_mg[valid_mask] - gt_mg[valid_mask])**2
                    val_rmse += torch.sqrt(se.mean()).item()

        avg_val = val_loss / len(val_ld)
        avg_rmse = val_rmse / len(val_ld)

        print(f"Epoch {epoch:3d} | TrLoss: {avg_loss:.4f} (Rec:{avg_recon:.4f} Diff:{avg_diff:.4f} KD:{avg_kd:.4f}) | ValLoss: {avg_val:.4f} | Val RMSE: {avg_rmse:.2f} Mg/ha", flush=True)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                'student': student.state_dict(),
                'diffusion': diffusion.state_dict(),
                'inr': inr_decoder.state_dict()
            }, args.save_path)

    print(f"\nTraining Complete. Best Model saved to {args.save_path}")

if __name__ == '__main__':
    main()
