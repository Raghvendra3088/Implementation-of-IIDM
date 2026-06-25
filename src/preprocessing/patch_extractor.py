"""
Patch Extractor — stride=64 re-extraction
Input : sentinel2_final.tif (4ch) + dem_final.tif + canopy_final.tif
Target: carbon_final.tif
Mask  : forest_mask_final.tif (sirf forest pixels wale patches)
"""
import numpy as np
import rasterio
import os
from pathlib import Path
import argparse

def extract_patches(
    sentinel_path, dem_path, canopy_path,
    carbon_path, mask_path,
    out_dir, patch_size=256, stride=64,
    min_forest_ratio=0.3,
    val_ratio=0.15, test_ratio=0.15,
    seed=42
):
    np.random.seed(seed)
    print(f"Stride={stride}, Patch={patch_size}x{patch_size}")

    # Load all rasters
    with rasterio.open(sentinel_path) as src:
        sentinel = src.read().astype(np.float32)   # (4, H, W)
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)[None] # (1, H, W)
    with rasterio.open(canopy_path) as src:
        canopy = src.read(1).astype(np.float32)[None]
    with rasterio.open(carbon_path) as src:
        carbon = src.read(1).astype(np.float32)[None]
    with rasterio.open(mask_path) as src:
        mask = src.read(1).astype(np.uint8)        # (H, W)

    H, W = mask.shape
    print(f"Raster shape: {H}x{W}")

    # Stack input: sentinel(4) + dem(1) + canopy(1) = 6ch
    inp_stack = np.concatenate([sentinel, dem, canopy], axis=0)  # (6, H, W)

    # Normalize per-channel (min-max)
    for c in range(inp_stack.shape[0]):
        mn, mx = inp_stack[c].min(), inp_stack[c].max()
        inp_stack[c] = (inp_stack[c] - mn) / (mx - mn + 1e-8)

    # Carbon normalize to [-1, 1]
    c_min, c_max = np.percentile(carbon[carbon > 0], [1, 99])
    carbon_norm  = (carbon - c_min) / (c_max - c_min + 1e-8)
    carbon_norm  = carbon_norm * 2 - 1
    carbon_norm  = np.clip(carbon_norm, -1, 1)

    print(f"Carbon range (Mg C/ha): {c_min:.2f} – {c_max:.2f}")

    # Extract patches
    patches_inp, patches_tgt = [], []
    total = 0

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            mask_patch = mask[y:y+patch_size, x:x+patch_size]
            forest_ratio = mask_patch.mean()

            if forest_ratio < min_forest_ratio:
                continue

            inp_patch = inp_stack[:, y:y+patch_size, x:x+patch_size]
            tgt_patch = carbon_norm[:, y:y+patch_size, x:x+patch_size]

            # Skip if too many NaN/zero in carbon
            if np.isnan(tgt_patch).mean() > 0.2:
                continue

            patches_inp.append(inp_patch)
            patches_tgt.append(tgt_patch)
            total += 1

    print(f"Total valid patches: {total}")

    # Shuffle and split
    idx = np.random.permutation(total)
    n_val  = int(total * val_ratio)
    n_test = int(total * test_ratio)
    n_train = total - n_val - n_test

    splits = {
        "train": idx[:n_train],
        "val":   idx[n_train:n_train+n_val],
        "test":  idx[n_train+n_val:]
    }

    for split, idxs in splits.items():
        inp_dir = Path(out_dir) / split / "input"
        tgt_dir = Path(out_dir) / split / "target"
        inp_dir.mkdir(parents=True, exist_ok=True)
        tgt_dir.mkdir(parents=True, exist_ok=True)

        for i, j in enumerate(idxs):
            np.save(inp_dir / f"patch_{i:05d}.npy", patches_inp[j])
            np.save(tgt_dir / f"patch_{i:05d}.npy", patches_tgt[j])

        print(f"  {split:5s}: {len(idxs)} patches saved")

    # Save normalization stats for denormalization later
    np.save(Path(out_dir) / "carbon_stats.npy",
            np.array([c_min, c_max]))
    print(f"\ncarbon_stats.npy saved: vmin={c_min:.2f}, vmax={c_max:.2f}")
    print("Done ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride",   type=int, default=64)
    parser.add_argument("--out_dir",  type=str, default="data/processed/patches_s64")
    parser.add_argument("--min_forest", type=float, default=0.3)
    args = parser.parse_args()

    extract_patches(
        sentinel_path = "data/processed/final/sentinel2_final.tif",
        dem_path      = "data/processed/final/dem_final.tif",
        canopy_path   = "data/processed/final/canopy_final.tif",
        carbon_path   = "data/processed/final/carbon_final.tif",
        mask_path     = "data/masks/forest_mask_final.tif",
        out_dir       = args.out_dir,
        patch_size    = 256,
        stride        = args.stride,
        min_forest_ratio = args.min_forest,
    )
