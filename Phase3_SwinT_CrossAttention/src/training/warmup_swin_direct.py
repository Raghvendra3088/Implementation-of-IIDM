"""
Stage 1: Direct carbon supervision — Swin ko carbon predict karna seekhao.
No diffusion, no physics loss. Sirf: image → Swin → head → carbon.
Target: RMSE < 12 before diffusion training.
"""
import os, sys, json, argparse, glob
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

torch.backends.cudnn.enabled = False
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class PatchDataset(Dataset):
    def __init__(self, root, split):
        self.files = sorted(glob.glob(os.path.join(root, split, '*.npz')))
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        d = np.load(self.files[i])
        return (torch.from_numpy(d['image']).float(),
                torch.from_numpy(d['carbon']).float().unsqueeze(0),
                torch.from_numpy(d['mask']).float().unsqueeze(0))

class SwinDirectHead(nn.Module):
    def __init__(self):
        super().__init__()
        from src.models.swin_moe_encoder import SwinMoEEncoder
        self.enc = SwinMoEEncoder(in_chans=6, out_ch=256, pretrained=True, n_experts=4)
        self.dec = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.GELU(),
            nn.Conv2d(128, 64,  3, padding=1), nn.GroupNorm(8, 64),  nn.GELU(),
            nn.Conv2d(64,  1,   1), nn.Tanh()
        )
    def forward(self, x):
        f = self.enc(x)[0]                                    # (B,256,8,8)
        f = F.interpolate(f, x.shape[-2:], mode='bilinear', align_corners=False)
        return self.dec(f)                                    # (B,1,H,W)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--patch_dir',  default='data/processed/patches')
    p.add_argument('--epochs',     type=int,   default=40)
    p.add_argument('--batch_size', type=int,   default=4)
    p.add_argument('--lr',         type=float, default=5e-5)
    p.add_argument('--save_dir',   default='checkpoints/')
    args = p.parse_args()

    device = torch.device('cuda')
    C_MIN, C_MAX = 4.816, 129.18
    os.makedirs(args.save_dir, exist_ok=True)

    train_ds = PatchDataset(args.patch_dir, 'train')
    val_ds   = PatchDataset(args.patch_dir, 'val')
    train_ld = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, num_workers=2, pin_memory=False)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch_size,
                          shuffle=False, num_workers=2, pin_memory=False)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    model = SwinDirectHead().to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    best_rmse = float('inf')

    for epoch in range(1, args.epochs+1):
        model.train()
        tloss = 0.0
        for x, y, m in train_ld:
            x, y, m = x.to(device), y.to(device), m.to(device)
            opt.zero_grad()
            pred = model(x)
            loss = ((pred - y).abs() * m).sum() / (m.sum() + 1e-8)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tloss += loss.item()
        sched.step()
        tloss /= len(train_ld)

        # Val RMSE — only valid pixels
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for x, y, m in val_ld:
                x, y, m = x.to(device), y.to(device), m.to(device)
                pred = model(x)
                valid = m.bool()
                if valid.sum() == 0: continue
                p_r = ((pred[valid].clamp(-1,1) + 1)/2) * (C_MAX-C_MIN) + C_MIN
                g_r = ((y[valid].clamp(-1,1)    + 1)/2) * (C_MAX-C_MIN) + C_MIN
                preds.append(p_r.cpu().numpy())
                gts.append(g_r.cpu().numpy())

        if preds:
            p_all = np.concatenate(preds)
            g_all = np.concatenate(gts)
            rmse  = float(np.sqrt(np.mean((p_all - g_all)**2)))
            r2    = 1 - np.sum((g_all-p_all)**2) / (np.sum((g_all-g_all.mean())**2)+1e-8)
        else:
            rmse, r2 = 999.0, -99.0

        print(f"Epoch {epoch:3d}/{args.epochs} | Loss: {tloss:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}", flush=True)

        if rmse < best_rmse:
            best_rmse = rmse
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'encoder': model.enc.state_dict(), 'best_rmse': best_rmse},
                       f'{args.save_dir}/swin_warmup_best.pth')
            print(f"  ✓ Saved (RMSE={best_rmse:.4f})", flush=True)

    print(f"\nWarmup done. Best RMSE: {best_rmse:.4f} Mg C/ha")
    print("Next: use swin_warmup_best.pth encoder in diffusion training")

if __name__ == '__main__': main()
