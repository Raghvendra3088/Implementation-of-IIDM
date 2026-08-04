import os
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from src.dataset import PatchDataset
from src.models.vae import CarbonVAE

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--patch_dir',  default='data/processed/patches_v2')
    p.add_argument('--epochs',     type=int,   default=100)
    p.add_argument('--batch_size', type=int,   default=16)
    p.add_argument('--lr',         type=float, default=1e-4)
    p.add_argument('--save_dir',   default='checkpoints')
    p.add_argument('--kl_weight',  type=float, default=1e-5)
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_ds = PatchDataset(args.patch_dir, 'train')
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    print(f"Train patches: {len(train_ds)}")

    vae = CarbonVAE().to(device)
    optimizer = optim.AdamW(vae.parameters(), lr=args.lr)

    print("Starting VAE Training...")
    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        vae.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0

        for x, y, m in train_ld:
            # y is the carbon map, shape [B, 1, 256, 256]
            y = y.to(device)
            
            optimizer.zero_grad()
            recon, mu, logvar = vae(y)
            
            # Reconstruction loss (L1)
            recon_loss = F.l1_loss(recon, y)
            
            # KL Divergence loss
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            
            loss = recon_loss + args.kl_weight * kl_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()

        N = len(train_ld)
        avg_loss = total_loss / N
        avg_recon = total_recon / N
        avg_kl = total_kl / N
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{args.epochs} | Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(vae.state_dict(), os.path.join(args.save_dir, 'vae.pth'))

    print("Training finished! Saved checkpoints/vae.pth")

    # Now calculate latent_scale_factor
    print("Calculating Latent Scale Factor...")
    vae.load_state_dict(torch.load(os.path.join(args.save_dir, 'vae.pth'), map_location=device))
    vae.eval()
    
    all_latents = []
    with torch.no_grad():
        for x, y, m in train_ld:
            y = y.to(device)
            mu, logvar = vae.encode(y)
            # Use deterministic mu for scaling factor
            all_latents.append(mu.cpu())
            
    all_latents = torch.cat(all_latents, dim=0)
    std = torch.std(all_latents).item()
    scale_factor = 1.0 / (std + 1e-8)
    
    print(f"Latent Std Dev: {std:.6f}")
    print(f"Latent Scale Factor: {scale_factor:.6f}")
    
    # Save the scale factor so the diffusion script can load it
    with open(os.path.join(args.save_dir, 'vae_scale_factor.txt'), 'w') as f:
        f.write(str(scale_factor))

if __name__ == '__main__':
    main()
