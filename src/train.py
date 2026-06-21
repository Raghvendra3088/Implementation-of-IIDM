"""
IIDM Training Script
====================
Paper: IIDM — full training pipeline

Optimizer  : AdamW, lr=1e-4, weight_decay=1e-5
Scheduler  : CosineAnnealingLR, T_max=100
Epochs     : 100
Batch size : 4 (Mac CPU), 16 (GPU)
Grad clip  : max_norm=1.0
AMP        : enabled on CUDA

Loss weights (paper):
    lambda_kd    = 0.1
    lambda_recon = 1.0

Logs every step to console.
Saves checkpoint every 10 epochs + best val loss.

Run:
    source venv/bin/activate
    python src/train.py
    python src/train.py --epochs 100 --batch_size 16  # GPU
"""

import os, json, time, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp   import GradScaler, autocast

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.iidm import IIDM


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════

class IIDMDataset(Dataset):
    """
    Loads preprocessed patches from data/processed/patches_final/
    Input  : (6, 256, 256) float32, range [-1, 1]
    Target : (1, 256, 256) float32, range [-1, 1]
    """

    def __init__(self, split: str = "train", augment: bool = False,
                 patch_dir: str = "data/processed/patches_final"):
        assert split in ("train", "val", "test")
        base          = Path(patch_dir) / split
        self.inp_dir  = base / "input"
        self.tgt_dir  = base / "target"
        self.files    = sorted(self.inp_dir.glob("*.npy"))
        self.augment  = augment and (split == "train")

        if len(self.files) == 0:
            raise FileNotFoundError(
                f"No patches found in {self.inp_dir}\n"
                "Run src/preprocessing/final_alignment.py first."
            )
        print(f"  [{split}] {len(self.files)} patches")

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx].name
        inp  = np.load(self.inp_dir / name).astype(np.float32)
        tgt  = np.load(self.tgt_dir / name).astype(np.float32)

        if self.augment:
            if np.random.rand() > 0.5:
                inp = np.flip(inp, axis=2).copy()
                tgt = np.flip(tgt, axis=2).copy()
            if np.random.rand() > 0.5:
                inp = np.flip(inp, axis=1).copy()
                tgt = np.flip(tgt, axis=1).copy()
            k = np.random.randint(0, 4)
            if k > 0:
                inp = np.rot90(inp, k, axes=(1,2)).copy()
                tgt = np.rot90(tgt, k, axes=(1,2)).copy()

        return torch.from_numpy(inp), torch.from_numpy(tgt)


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_rmse(pred: torch.Tensor, gt: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((pred - gt) ** 2)).item()

def compute_mae(pred: torch.Tensor, gt: torch.Tensor) -> float:
    return torch.mean(torch.abs(pred - gt)).item()


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train(args):
    # ── Setup ─────────────────────────────────────────────────────────────────
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"\n{'='*55}")
    print(f"  IIDM Training")
    print(f"{'='*55}")
    print(f"  Device     : {device}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  LR         : {args.lr}")

    # ── Directories ───────────────────────────────────────────────────────────
    ckpt_dir = Path("checkpoints"); ckpt_dir.mkdir(exist_ok=True)
    log_path = Path("results") / "train_log.json"
    Path("results").mkdir(exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\n  Loading datasets ...")
    train_ds = IIDMDataset("train", augment=True,  patch_dir=args.patch_dir)
    val_ds   = IIDMDataset("val",   augment=False, patch_dir=args.patch_dir)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers,
        pin_memory=(device.type == "cuda"), drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers,
        pin_memory=(device.type == "cuda")
    )
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val   batches : {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\n  Initializing IIDM model ...")
    model = IIDM(
        in_channels  = 6,
        T            = args.T,
        lambda_kd    = args.lambda_kd,
        lambda_recon = args.lambda_recon,
        device       = device,
    ).to(device)

    n_train = sum(p.numel() for p in model.trainable_parameters()) / 1e6
    print(f"  Trainable params : {n_train:.2f}M")

    # ── Optimizer + Scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=args.lr, weight_decay=1e-5
    )
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # AMP scaler (CUDA only)
    use_amp = (device.type == "cuda")
    scaler  = GradScaler(enabled=use_amp)

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_epoch = 1
    best_val    = float("inf")
    log         = []

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.student_vgg.load_state_dict(ckpt["student_vgg"])
        model.unet.load_state_dict(ckpt["unet"])
        model.inr.load_state_dict(ckpt["inr"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val    = ckpt.get("best_val", float("inf"))
        print(f"  Resumed from epoch {ckpt['epoch']}")

    # ══════════════════════════════════════════════════════════════════════════
    # EPOCH LOOP
    # ══════════════════════════════════════════════════════════════════════════

    print(f"\n  Starting training from epoch {start_epoch} ...\n")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        # ── TRAIN ─────────────────────────────────────────────────────────────
        model.train()
        tr = {"L_total":0, "L_diff":0, "L_kd":0, "L_recon":0}
        tr_rmse, tr_mae = 0.0, 0.0

        for step, (inp, tgt) in enumerate(train_loader):
            inp, tgt = inp.to(device), tgt.to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp):
                loss, comps = model(inp, tgt)

            scaler.scale(loss).backward()

            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.trainable_parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            for k in tr: tr[k] += comps[k]

            # Quick RMSE on training batch (no INR inference — use loss proxy)
            if step % 50 == 0:
                elapsed = time.time() - t0
                print(f"  Ep {epoch:03d} | step {step:04d}/{len(train_loader)} | "
                      f"loss={comps['L_total']:.4f} | "
                      f"diff={comps['L_diff']:.4f} | "
                      f"kd={comps['L_kd']:.4f} | "
                      f"recon={comps['L_recon']:.4f} | "
                      f"t={elapsed:.0f}s")

        for k in tr: tr[k] /= len(train_loader)

        # ── VALIDATE ──────────────────────────────────────────────────────────
        model.eval()
        val = {"L_total":0, "L_diff":0, "L_kd":0, "L_recon":0}
        val_rmse, val_mae, n_val = 0.0, 0.0, 0

        with torch.no_grad():
            for inp, tgt in val_loader:
                inp, tgt = inp.to(device), tgt.to(device)

                with autocast(enabled=use_amp):
                    loss, comps = model(inp, tgt)

                for k in val: val[k] += comps[k]

                # Predict carbon for metrics (5 DDIM steps — fast)
                pred = model.predict(inp, steps=5)
                val_rmse += compute_rmse(pred, tgt)
                val_mae  += compute_mae(pred, tgt)
                n_val    += 1

        for k in val: val[k] /= len(val_loader)
        val_rmse /= n_val
        val_mae  /= n_val

        lr_sched.step()
        epoch_time = time.time() - t0

        # ── Log ───────────────────────────────────────────────────────────────
        entry = {
            "epoch"     : epoch,
            "train"     : tr,
            "val"       : val,
            "val_rmse"  : val_rmse,
            "val_mae"   : val_mae,
            "lr"        : optimizer.param_groups[0]["lr"],
            "time_s"    : epoch_time,
        }
        log.append(entry)

        print(f"\n  ── Epoch {epoch:03d}/{args.epochs} "
              f"({epoch_time:.0f}s) ──────────────────────")
        print(f"  Train  loss={tr['L_total']:.4f}  "
              f"diff={tr['L_diff']:.4f}  "
              f"kd={tr['L_kd']:.4f}  "
              f"recon={tr['L_recon']:.4f}")
        print(f"  Val    loss={val['L_total']:.4f}  "
              f"RMSE={val_rmse:.4f}  MAE={val_mae:.4f}  "
              f"lr={entry['lr']:.2e}")

        # ── Save checkpoints ──────────────────────────────────────────────────
        def save_ckpt(path, tag=""):
            torch.save({
                "epoch"       : epoch,
                "student_vgg" : model.student_vgg.state_dict(),
                "unet"        : model.unet.state_dict(),
                "inr"         : model.inr.state_dict(),
                "teacher_cond": model.teacher_cond.state_dict(),
                "optimizer"   : optimizer.state_dict(),
                "val_loss"    : val["L_total"],
                "best_val"    : best_val,
            }, path)
            if tag: print(f"  [{tag}] Checkpoint saved → {path}")

        # Best model
        if val["L_total"] < best_val:
            best_val = val["L_total"]
            save_ckpt(ckpt_dir / "best_model.pth", "BEST")

        # Periodic checkpoint
        if epoch % args.save_every == 0:
            save_ckpt(ckpt_dir / f"epoch_{epoch:03d}.pth", f"EP{epoch}")

        # Save log
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)

        print()

    print(f"{'='*55}")
    print(f"  Training complete!")
    print(f"  Best val loss : {best_val:.4f}")
    print(f"  Checkpoint    : {ckpt_dir / 'best_model.pth'}")
    print(f"  Log           : {log_path}")
    print(f"{'='*55}")


# ══════════════════════════════════════════════════════════════════════════════
# ARGS
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Train IIDM")
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch_size",   type=int,   default=4)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--T",            type=int,   default=1000)
    p.add_argument("--lambda_kd",    type=float, default=0.1)
    p.add_argument("--lambda_recon", type=float, default=1.0)
    p.add_argument("--save_every",   type=int,   default=10)
    p.add_argument("--workers",      type=int,   default=0)
    p.add_argument("--resume",       type=str,   default="")
    p.add_argument("--patch_dir",    type=str,
                   default="data/processed/patches_final")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
