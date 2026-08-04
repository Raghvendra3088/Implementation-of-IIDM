import os, sys, json
import torch
torch.backends.cudnn.enabled = False
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.vae import CarbonVAE
from src.models.prithvi_teacher import KDStudent12
from src.models.latent_kd_unet import LatentKDUNet12

class PatchDataset(Dataset):
    def __init__(self, root, split):
        import glob
        self.split = split
        split_dir = os.path.join(root, split)
        self.files = sorted(glob.glob(os.path.join(split_dir, '*.npz')))
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


def make_schedule(T=1000, device='cpu'):
    betas     = torch.linspace(1e-4, 0.02, T, device=device)
    alpha_bar = torch.cumprod(1.0 - betas, dim=0)
    return betas, alpha_bar


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--patch_dir', default='data/processed/patches_v2')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--student_ckpt', default='checkpoints/prithvi_blockwise_kd.pth')
    p.add_argument('--vae_ckpt', default='checkpoints/vae.pth')
    p.add_argument('--vae_scale_file', default='checkpoints/vae_scale_factor.txt')
    p.add_argument('--save_path', default='checkpoints/latent_kd_unet.pth')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    print("Loading Models...")
    
    # Load VAE
    vae = CarbonVAE().to(device)
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location=device))
    vae.eval()
    for param in vae.parameters(): param.requires_grad = False
    
    # Load Latent Scale Factor
    try:
        with open(args.vae_scale_file, 'r') as f:
            scale_factor = float(f.read().strip())
    except:
        scale_factor = 1.0
        print("WARNING: Could not load VAE scale factor. Defaulting to 1.0")
    print(f"VAE Scale Factor: {scale_factor}")

    # Load Student (Conditioner)
    student = KDStudent12(in_channels=4).to(device)
    ckpt = torch.load(args.student_ckpt, map_location=device)
    student.load_state_dict(ckpt['student'])
    student.eval()
    for param in student.parameters(): param.requires_grad = False
    
    # Latent Diffusion UNet
    unet = LatentKDUNet12(in_channels=4).to(device)
    
    optimizer = torch.optim.Adam(unet.parameters(), lr=args.lr)

    print("Loading Data...")
    train_ds = PatchDataset(args.patch_dir, 'train')
    val_ds   = PatchDataset(args.patch_dir, 'val')
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_ld   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    T = 1000
    betas, alpha_bar = make_schedule(T, device)
    
    print(f"Starting Latent Diffusion Training on {len(train_ds)} images for {args.epochs} epochs")

    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        unet.train()
        train_loss = 0.0

        for x, y, m in train_ld:
            x, y, m = x.to(device), y.to(device), m.to(device)
            B = x.shape[0]

            with torch.no_grad():
                # Encode target y to latent z
                mu, _ = vae.encode(y)
                z = mu * scale_factor
                
                # Get student features
                s_feats = student(x)
                
            # Diffusion forward process
            t = torch.randint(1, T + 1, (B,), device=device)
            noise = torch.randn_like(z)
            ab_t = alpha_bar[t - 1].view(B, 1, 1, 1)
            z_t = (ab_t ** 0.5) * z + ((1 - ab_t) ** 0.5) * noise

            # Predict noise
            noise_pred = unet(z_t, t, s_feats)

            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train = train_loss / len(train_ld)

        # Validation (Single Step Denoising at T//2)
        unet.eval()
        val_loss = 0.0
        val_rmse = 0.0
        with torch.no_grad():
            for x, y, m in val_ld:
                x, y, m = x.to(device), y.to(device), m.to(device)
                B = x.shape[0]
                
                mu, _ = vae.encode(y)
                z = mu * scale_factor
                s_feats = student(x)

                t = torch.full((B,), T // 2, device=device)
                noise = torch.randn_like(z)
                ab_t = alpha_bar[t - 1].view(B, 1, 1, 1)
                z_t = (ab_t ** 0.5) * z + ((1 - ab_t) ** 0.5) * noise

                noise_pred = unet(z_t, t, s_feats)
                loss = F.mse_loss(noise_pred, noise)
                val_loss += loss.item()
                
                # Predict z0 and decode
                z0_pred = (z_t - ((1 - ab_t) ** 0.5) * noise_pred) / (ab_t ** 0.5)
                y0_pred = vae.decode(z0_pred / scale_factor)
                
                y0_pred = y0_pred.clamp(-1, 1)
                
                # Unnormalize carbon to Mg/ha
                # c_min and c_max depend on dataset. Let's approximate using max 130
                # Using 0 to 130 Mg/ha mapping
                c_min = 0.0
                c_max = 130.0
                
                y_mg = ((y + 1) / 2) * (c_max - c_min) + c_min
                pred_mg = ((y0_pred + 1) / 2) * (c_max - c_min) + c_min
                
                valid_mask = (m > 0)
                if valid_mask.sum() > 0:
                    se = (y_mg[valid_mask] - pred_mg[valid_mask])**2
                    val_rmse += torch.sqrt(se.mean()).item()
                else:
                    val_rmse += 0.0

        avg_val = val_loss / len(val_ld)
        avg_rmse = val_rmse / len(val_ld)

        print(f"Epoch {epoch:3d}/{args.epochs} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Val Est. RMSE: {avg_rmse:.2f} Mg/ha", flush=True)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(unet.state_dict(), args.save_path)

    print(f"\nTraining Complete. Best Model saved to {args.save_path}")

if __name__ == '__main__':
    main()
