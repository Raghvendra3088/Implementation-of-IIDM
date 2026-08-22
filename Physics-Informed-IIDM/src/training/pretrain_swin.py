"""
Phase 0: Swin-Base warm-up on carbon prediction (direct supervision).
No diffusion, no physics loss — sirf Swin features → carbon map.
Target: RMSE < 15 before diffusion training.
"""
import os, sys, json, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class PatchDataset(Dataset):
    def __init__(self, root, split):
        import glob
        self.files = sorted(glob.glob(os.path.join(root, split, '*.npz')))
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        d = np.load(self.files[i])
        x = torch.from_numpy(d['image']).float()
        y = torch.from_numpy(d['carbon']).float().unsqueeze(0)
        m = torch.from_numpy(d['mask']).float().unsqueeze(0)
        return x, y, m

class SwinCarbonHead(nn.Module):
    """Swin + direct carbon prediction head — warm-up model."""
    def __init__(self, pretrained=True):
        super().__init__()
        from src.models.swin_encoder import SwinEncoder
        self.encoder = SwinEncoder(in_chans=6, out_ch=256, pretrained=pretrained)
        self.head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.GroupNorm(8,128), nn.GELU(),
            nn.Conv2d(128, 64,  3, padding=1), nn.GroupNorm(8,64),  nn.GELU(),
            nn.Conv2d(64,  1,   1), nn.Tanh()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        feats = self.encoder(x)          # [(B,256,8,8)]
        f = F.interpolate(feats[-1], size=(H,W), mode='bilinear', align_corners=False)
        return self.head(f)              # (B,1,H,W) in [-1,1]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--patch_dir',  default='data/processed/patches')
    p.add_argument('--epochs',     type=int,   default=40)
    p.add_argument('--batch_size', type=int,   default=4)
    p.add_argument('--lr',         type=float, default=1e-4)
    p.add_argument('--save_dir',   default='checkpoints/')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    C_MIN, C_MAX = 0.04, 207.97
    os.makedirs(args.save_dir, exist_ok=True)

    train_ds = PatchDataset(args.patch_dir, 'train')
    val_ds   = PatchDataset(args.patch_dir, 'val')
    train_ld = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, num_workers=4, pin_memory=True)
    val_ld   = DataLoader(val_ds, batch_size=args.batch_size,
                          shuffle=False, num_workers=2, pin_memory=True)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    model = SwinCarbonHead(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_rmse = float('inf')
    log = []

    for epoch in range(1, args.epochs+1):
        model.train()
        train_loss = 0.0
        for x, y0, mask in train_ld:
            x, y0, mask = x.to(device), y0.to(device), mask.to(device)
            optimizer.zero_grad()
            pred = model(x)
            # Only valid pixels
            loss = ((pred - y0).abs() * mask).sum() / (mask.sum() + 1e-8)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_ld)
        scheduler.step()

        # Val RMSE
        model.eval()
        rmses = []
        with torch.no_grad():
            for x, y0, mask in val_ld:
                x, y0, mask = x.to(device), y0.to(device), mask.to(device)
                pred = model(x)
                valid = (mask > 0).float()
                n = valid.sum()
                if n < 1: continue
                p_r = ((pred.clamp(-1,1)+1)/2)*(C_MAX-C_MIN)+C_MIN
                g_r = ((y0.clamp(-1,1)+1)/2)*(C_MAX-C_MIN)+C_MIN
                rmse = (((p_r-g_r)**2 * valid).sum() / n).sqrt().item()
                rmses.append(rmse)

        val_rmse = float(np.mean(rmses)) if rmses else 999.0
        print(f"Epoch {epoch:3d}/{args.epochs} | Loss: {train_loss:.4f} | Val RMSE: {val_rmse:.4f} Mg C/ha")

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            ckpt = os.path.join(args.save_dir, 'swin_warmup_best.pth')
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'encoder': model.encoder.state_dict(),
                        'best_rmse': best_rmse}, ckpt)
            print(f"  ✓ Saved (RMSE={best_rmse:.4f})")

        log.append({'epoch': epoch, 'loss': train_loss, 'val_rmse': val_rmse})

    with open(os.path.join(args.save_dir, 'swin_warmup_log.json'), 'w') as f:
        json.dump({'best_rmse': best_rmse, 'log': log}, f, indent=2)
    print(f"\nWarm-up done. Best RMSE: {best_rmse:.4f} Mg C/ha")
    print(f"Checkpoint: {args.save_dir}/swin_warmup_best.pth")

if __name__ == '__main__':
    main()
