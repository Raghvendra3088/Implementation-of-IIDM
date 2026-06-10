"""
IIDM Dataset — PyTorch DataLoader
===================================
Loads preprocessed .npy patches for model training.

Usage:
    from src.preprocessing.dataset import IIDMDataset
    from torch.utils.data import DataLoader

    train_ds = IIDMDataset(split="train")
    loader   = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)

    for inputs, targets in loader:
        # inputs  : (B, 6, 256, 256) float32
        # targets : (B, 1, 256, 256) float32
        ...
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

ROOT      = Path(__file__).resolve().parents[2]
PATCH_DIR = ROOT / "data" / "processed" / "patches"
STATS_PATH = ROOT / "data" / "processed" / "norm_stats.json"


class IIDMDataset(Dataset):
    """
    PyTorch Dataset for IIDM carbon stock estimation.

    Args:
        split      : "train", "val", or "test"
        augment    : apply random horizontal/vertical flip (train only)
        patch_dir  : override default patch directory
    """

    def __init__(self, split: str = "train",
                 augment: bool = False,
                 patch_dir: Path = None):

        assert split in ("train", "val", "test"), \
            f"split must be 'train', 'val', or 'test'. Got: {split}"

        self.split    = split
        self.augment  = augment and (split == "train")
        base_dir      = patch_dir or PATCH_DIR

        self.inp_dir  = base_dir / split / "input"
        self.tgt_dir  = base_dir / split / "target"

        if not self.inp_dir.exists():
            raise FileNotFoundError(
                f"Patch directory not found: {self.inp_dir}\n"
                "Run run_preprocessing.py first."
            )

        self.files = sorted(self.inp_dir.glob("*.npy"))
        if len(self.files) == 0:
            raise RuntimeError(f"No .npy patches found in {self.inp_dir}")

        # Load normalization stats
        self.norm_stats = None
        if STATS_PATH.exists():
            with open(STATS_PATH) as f:
                self.norm_stats = json.load(f)

        print(f"[IIDMDataset] {split}: {len(self.files)} patches loaded")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        fname    = self.files[idx].name
        inp      = np.load(self.inp_dir / fname).astype(np.float32)  # (6, H, W)
        tgt      = np.load(self.tgt_dir / fname).astype(np.float32)  # (1, H, W)

        # ── Data augmentation (train only) ────────────────────────────────────
        if self.augment:
            if np.random.rand() > 0.5:          # horizontal flip
                inp = np.flip(inp, axis=2).copy()
                tgt = np.flip(tgt, axis=2).copy()
            if np.random.rand() > 0.5:          # vertical flip
                inp = np.flip(inp, axis=1).copy()
                tgt = np.flip(tgt, axis=1).copy()
            if np.random.rand() > 0.5:          # 90-degree rotation
                k   = np.random.randint(1, 4)
                inp = np.rot90(inp, k, axes=(1, 2)).copy()
                tgt = np.rot90(tgt, k, axes=(1, 2)).copy()

        return torch.from_numpy(inp), torch.from_numpy(tgt)

    def denormalize_carbon(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Reverse normalization on carbon predictions.
        Returns values in original Mg C/ha units.
        """
        if self.norm_stats is None:
            return tensor
        vmin = self.norm_stats["carbon"]["min"]
        vmax = self.norm_stats["carbon"]["max"]
        return tensor * (vmax - vmin) + vmin


if __name__ == "__main__":
    # Quick test
    for split in ("train", "val", "test"):
        ds = IIDMDataset(split=split, augment=(split == "train"))
        inp, tgt = ds[0]
        print(f"  {split:5s} — input: {inp.shape}, target: {tgt.shape}, "
              f"dtype: {inp.dtype}")
