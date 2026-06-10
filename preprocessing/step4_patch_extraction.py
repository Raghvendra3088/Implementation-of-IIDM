"""
IIDM Preprocessing - Step 4: Patch Extraction & Dataset Preparation
====================================================================
Paper reference: Section 4 — model input is 256×256 patches
- Extracts overlapping patches from all aligned rasters
- Creates train/val/test splits (70/15/15)
- Normalizes each channel to [0, 1]
- Saves patches as .npy files ready for PyTorch DataLoader

Input  : data/processed/aligned_gf1.tif
         data/processed/aligned_dem.tif
         data/processed/aligned_canopy.tif
         data/processed/masked_carbon_density.tif
         data/masks/forest_mask.tif
Output : data/processed/patches/
            train/  input/  *.npy   (C×H×W float32)
                    target/ *.npy   (1×H×W float32)
            val/    ...
            test/   ...
         data/processed/norm_stats.json
"""

import os
import json
import numpy as np
import rasterio
from pathlib     import Path
from sklearn.model_selection import train_test_split

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MASKS_DIR = ROOT / "data" / "masks"
PATCH_DIR = PROCESSED / "patches"

# ── Hyperparameters ────────────────────────────────────────────────────────────
PATCH_SIZE   = 256     # paper uses 256×256
STRIDE       = 128     # 50% overlap between patches
MIN_FOREST   = 0.3     # min fraction of forest pixels in a patch (skip sparse patches)
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
# TEST_RATIO  = 0.15 (remainder)
RANDOM_SEED  = 42


def load_raster(path: Path, nodata_fill: float = 0.0) -> np.ndarray:
    """Load raster as float32 array, fill nodata with 0."""
    with rasterio.open(path) as src:
        data   = src.read().astype(np.float32)
        nodata = src.nodata

    if nodata is not None:
        data[data == nodata] = nodata_fill
    return data


def normalize_channel(arr: np.ndarray, p_low: float = 2.0,
                       p_high: float = 98.0) -> tuple:
    """
    Percentile-based normalization to [0, 1].
    Returns (normalized_array, vmin, vmax) for saving stats.
    """
    vmin = np.percentile(arr[arr != 0], p_low)
    vmax = np.percentile(arr[arr != 0], p_high)
    norm = np.clip((arr - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
    return norm, float(vmin), float(vmax)


def extract_patches(input_stack: np.ndarray,   # (C, H, W)
                     target: np.ndarray,         # (1, H, W)
                     mask: np.ndarray,           # (H, W) binary
                     patch_size: int,
                     stride: int,
                     min_forest: float) -> list:
    """
    Slide a window over the image stack and extract valid patches.
    A patch is valid if it has ≥ min_forest fraction of forest pixels.
    """
    _, H, W  = input_stack.shape
    patches  = []
    count_total, count_valid = 0, 0

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            count_total += 1

            # Check forest coverage in this patch
            mask_patch   = mask[y:y+patch_size, x:x+patch_size]
            forest_frac  = mask_patch.mean()
            if forest_frac < min_forest:
                continue

            inp_patch = input_stack[:, y:y+patch_size, x:x+patch_size]
            tgt_patch = target[:, y:y+patch_size, x:x+patch_size]

            # Skip patches with too many zeros (likely edge artifacts)
            if (inp_patch == 0).mean() > 0.5:
                continue

            patches.append((inp_patch, tgt_patch))
            count_valid += 1

    print(f"  Patches total={count_total}, valid={count_valid} "
          f"({100*count_valid/max(count_total,1):.1f}%)")
    return patches


def save_patches(patches: list, split_dir: Path) -> None:
    """Save (input, target) patch pairs as .npy files."""
    inp_dir = split_dir / "input"
    tgt_dir = split_dir / "target"
    inp_dir.mkdir(parents=True, exist_ok=True)
    tgt_dir.mkdir(parents=True, exist_ok=True)

    for i, (inp, tgt) in enumerate(patches):
        np.save(inp_dir / f"patch_{i:05d}.npy", inp)
        np.save(tgt_dir / f"patch_{i:05d}.npy", tgt)


def main():
    print("=" * 55)
    print("  STEP 4 — Patch Extraction & Dataset Splits")
    print("=" * 55)

    # ── 1. Load all aligned rasters ───────────────────────────────────────────
    print("\n[1/5] Loading aligned rasters ...")

    paths = {
        "gf1"    : PROCESSED / "aligned_gf1.tif",
        "dem"    : PROCESSED / "aligned_dem.tif",
        "canopy" : PROCESSED / "aligned_canopy.tif",
        "carbon" : PROCESSED / "masked_carbon_density.tif",
        "mask"   : MASKS_DIR  / "forest_mask.tif",
    }

    for name, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(
                f"Missing: {p}\n"
                f"Run previous preprocessing steps first."
            )

    gf1_arr    = load_raster(paths["gf1"])      # (4, H, W)
    dem_arr    = load_raster(paths["dem"])       # (1, H, W)
    canopy_arr = load_raster(paths["canopy"])    # (1, H, W)
    carbon_arr = load_raster(paths["carbon"])    # (1, H, W)

    with rasterio.open(paths["mask"]) as src:
        mask_arr = src.read(1).astype(np.float32)
        mask_arr = (mask_arr == 1).astype(np.float32)   # binary: 1=forest, 0=rest

    print(f"  GF-1   : {gf1_arr.shape}")
    print(f"  DEM    : {dem_arr.shape}")
    print(f"  Canopy : {canopy_arr.shape}")
    print(f"  Carbon : {carbon_arr.shape}")

    # ── 2. Normalize each channel ─────────────────────────────────────────────
    print("\n[2/5] Normalizing channels ...")
    norm_stats = {}

    norm_gf1   = np.zeros_like(gf1_arr)
    for b in range(gf1_arr.shape[0]):
        norm_gf1[b], vmin, vmax = normalize_channel(gf1_arr[b])
        norm_stats[f"gf1_b{b+1}"] = {"min": vmin, "max": vmax}
        print(f"  GF-1 Band {b+1}: [{vmin:.2f}, {vmax:.2f}] → [0, 1]")

    norm_dem, vmin, vmax    = normalize_channel(dem_arr[0])
    norm_dem                = norm_dem[np.newaxis]
    norm_stats["dem"]       = {"min": vmin, "max": vmax}
    print(f"  DEM         : [{vmin:.2f}, {vmax:.2f}] → [0, 1]")

    norm_canopy, vmin, vmax = normalize_channel(canopy_arr[0])
    norm_canopy             = norm_canopy[np.newaxis]
    norm_stats["canopy"]    = {"min": vmin, "max": vmax}
    print(f"  Canopy      : [{vmin:.2f}, {vmax:.2f}] → [0, 1]")

    norm_carbon, vmin, vmax = normalize_channel(carbon_arr[0])
    norm_carbon             = norm_carbon[np.newaxis]
    norm_stats["carbon"]    = {"min": vmin, "max": vmax}
    print(f"  Carbon      : [{vmin:.2f}, {vmax:.2f}] → [0, 1]")

    # Save normalization stats for inference-time denormalization
    stats_path = PROCESSED / "norm_stats.json"
    with open(stats_path, "w") as f:
        json.dump(norm_stats, f, indent=2)
    print(f"\n  [SAVED] {stats_path}")

    # ── 3. Stack input channels: GF-1(4) + DEM(1) + Canopy(1) = 6 channels ───
    print("\n[3/5] Stacking input channels (6 total) ...")
    input_stack = np.concatenate([norm_gf1, norm_dem, norm_canopy], axis=0)
    target      = norm_carbon
    print(f"  Input  shape : {input_stack.shape}  (C=6, H, W)")
    print(f"  Target shape : {target.shape}       (C=1, H, W)")

    # ── 4. Extract patches ────────────────────────────────────────────────────
    print(f"\n[4/5] Extracting {PATCH_SIZE}×{PATCH_SIZE} patches "
          f"(stride={STRIDE}, min_forest={MIN_FOREST}) ...")
    patches = extract_patches(input_stack, target, mask_arr,
                               PATCH_SIZE, STRIDE, MIN_FOREST)

    if len(patches) == 0:
        raise RuntimeError(
            "No valid patches extracted!\n"
            f"Try lowering MIN_FOREST (currently {MIN_FOREST}) "
            f"or STRIDE (currently {STRIDE})."
        )
    print(f"  Total valid patches : {len(patches)}")

    # ── 5. Train / Val / Test split ───────────────────────────────────────────
    print("\n[5/5] Splitting into train / val / test ...")
    indices = list(range(len(patches)))
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(indices)

    n_train = int(TRAIN_RATIO * len(indices))
    n_val   = int(VAL_RATIO   * len(indices))

    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    splits = {
        "train" : [patches[i] for i in train_idx],
        "val"   : [patches[i] for i in val_idx],
        "test"  : [patches[i] for i in test_idx],
    }

    for split_name, split_patches in splits.items():
        split_dir = PATCH_DIR / split_name
        save_patches(split_patches, split_dir)
        print(f"  {split_name:5s} : {len(split_patches):4d} patches → {split_dir}")

    print(f"\n[DONE] All patches saved to: {PATCH_DIR}")
    print("\n  Next step: src/models/ — model implementation")


if __name__ == "__main__":
    main()
