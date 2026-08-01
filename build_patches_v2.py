"""
IIDM patch builder v2 — NO interpolation, spatial split, GEDI footprint-only labels
"""
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
import h5py
from pathlib import Path
import glob, json
from pyproj import Transformer

BASE    = Path("/Users/raghvendra/iidm_project")
OUT_DIR = BASE / "patches_v2"
OUT_DIR.mkdir(exist_ok=True)
(OUT_DIR / "train").mkdir(exist_ok=True)
(OUT_DIR / "val").mkdir(exist_ok=True)
(OUT_DIR / "test").mkdir(exist_ok=True)

TARGET_CRS = "EPSG:32648"
TARGET_RES = 16
PATCH_SIZE = 64
STRIDE     = 32
HUIZE_BBOX = (103.0, 25.8, 103.8, 26.6)
MIN_GEDI_PIXELS = 5   # patch must have at least this many actual GEDI pixels

# ── Reuse tmp_processed from v1 (already reprojected S2 bands) ──
tmp_dir = BASE / "tmp_processed"

def load_band(path):
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32), src.transform

b02, ref_transform = load_band(tmp_dir / "s2_B02.tif")
b03, _ = load_band(tmp_dir / "s2_B03.tif")
b04, _ = load_band(tmp_dir / "s2_B04.tif")
b08, _ = load_band(tmp_dir / "s2_B08.tif")
H, W = b02.shape
print(f"Grid: {H} x {W}")

print("=" * 60)
print("STEP 1: GEDI points → pixel-level labels (NO interpolation)")
print("=" * 60)

h5_files = glob.glob(str(BASE / "GEDI_L4A_AGB_Density_V2/*.h5"))
all_lons, all_lats, all_agb = [], [], []
west, south, east, north = HUIZE_BBOX

for h5f in h5_files:
    try:
        with h5py.File(h5f, 'r') as f:
            for beam in [k for k in f.keys() if k.startswith('BEAM')]:
                try:
                    lat  = f[beam]['lat_lowestmode'][:]
                    lon  = f[beam]['lon_lowestmode'][:]
                    agb  = f[beam]['agbd'][:]
                    qual = f[beam]['l4_quality_flag'][:]
                    mask = ((lat >= south) & (lat <= north) &
                            (lon >= west)  & (lon <= east)  &
                            (qual == 1)    & (agb > 0))
                    if mask.sum() > 0:
                        all_lons.extend(lon[mask])
                        all_lats.extend(lat[mask])
                        all_agb.extend(agb[mask])
                except: pass
    except: pass

print(f"GEDI points: {len(all_lons)}")
all_carbon = np.array(all_agb) * 0.47

transformer = Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True)
xs, ys = transformer.transform(np.array(all_lons), np.array(all_lats))

# Convert to pixel row/col — NO griddata, direct assignment only
inv_transform = ~ref_transform
cols, rows = inv_transform * (xs, ys)
cols = cols.astype(int)
rows = rows.astype(int)

carbon_grid = np.full((H, W), np.nan, dtype=np.float32)
valid_mask  = np.zeros((H, W), dtype=bool)

for r, c, cval in zip(rows, cols, all_carbon):
    if 0 <= r < H and 0 <= c < W:
        # If multiple GEDI shots land on same pixel, average them
        if valid_mask[r, c]:
            carbon_grid[r, c] = (carbon_grid[r, c] + cval) / 2
        else:
            carbon_grid[r, c] = cval
            valid_mask[r, c] = True

print(f"Pixels with actual GEDI label: {valid_mask.sum()} / {H*W} ({100*valid_mask.sum()/(H*W):.2f}%)")

print("=" * 60)
print("STEP 2: Normalize inputs")
print("=" * 60)

inputs = np.stack([b02, b03, b04, b08], axis=0)
stats = {}
norm_inputs = np.zeros_like(inputs)
for i, name in enumerate(['B02','B03','B04','B08']):
    ch = inputs[i]
    valid = ch[ch > 0]
    mean, std = float(valid.mean()), float(valid.std())
    norm_inputs[i] = (ch - mean) / (std + 1e-6)
    stats[name] = {'mean': mean, 'std': std}

car_valid = carbon_grid[valid_mask]
car_mean, car_std = float(car_valid.mean()), float(car_valid.std())
stats['carbon'] = {'mean': car_mean, 'std': car_std}
norm_carbon = (carbon_grid - car_mean) / (car_std + 1e-6)
norm_carbon = np.clip(norm_carbon, -3, 3)

with open(OUT_DIR / "norm_stats.json", 'w') as f:
    json.dump(stats, f, indent=2)
print(f"Carbon: mean={car_mean:.2f} std={car_std:.2f}")

print("=" * 60)
print("STEP 3: Spatial split — by row zones (prevents leakage)")
print("=" * 60)

# Split by ROW bands: top 80% rows = train, next 10% = val, bottom 10% = test
# Ensures NO overlapping patches across splits (patches don't cross zone boundary)
train_end = int(H * 0.8)
val_end   = int(H * 0.9)

print(f"Train rows: 0-{train_end}")
print(f"Val rows:   {train_end}-{val_end}")
print(f"Test rows:  {val_end}-{H}")

print("=" * 60)
print("STEP 4: Extract patches — only where enough real GEDI labels")
print("=" * 60)

counts = {'train': 0, 'val': 0, 'test': 0}
P = PATCH_SIZE

for r in range(0, H - P + 1, STRIDE):
    for c in range(0, W - P + 1, STRIDE):
        x_patch = norm_inputs[:, r:r+P, c:c+P]
        y_patch = norm_carbon[r:r+P, c:c+P]
        mask_patch = valid_mask[r:r+P, c:c+P]

        n_valid = mask_patch.sum()
        if n_valid < MIN_GEDI_PIXELS:
            continue
        if (x_patch == 0).mean() > 0.3:
            continue

        # Determine split by patch's row position (no overlap across zones)
        if r + P <= train_end:
            split = 'train'
        elif r >= train_end and r + P <= val_end:
            split = 'val'
        elif r >= val_end:
            split = 'test'
        else:
            continue  # patch straddles zone boundary — skip to avoid leakage

        y_masked = np.where(mask_patch, y_patch, 0)  # zero out non-GEDI pixels

        np.savez_compressed(
            OUT_DIR / split / f"patch_{counts[split]:05d}.npz",
            image=x_patch.astype(np.float32),
            carbon=y_masked.astype(np.float32),
            mask=mask_patch.astype(np.float32),   # 1 where real GEDI label exists
        )
        counts[split] += 1

print(f"\nTrain: {counts['train']} | Val: {counts['val']} | Test: {counts['test']}")
print("Done. No interpolation used — only real GEDI footprint pixels as labels.")
print("Loss function must now use `mask` to only compute error on labeled pixels.")
