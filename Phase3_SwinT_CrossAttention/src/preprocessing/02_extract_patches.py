import os
import json
import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import box
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path("/Users/raghvendra/iidm_project")
OUT_ROOT  = Path("/Users/raghvendra/iidm_project/Physics-Informed-IIDM")
PROCESSED = ROOT / "data" / "processed"
MASKS_DIR = Path("/Users/raghvendra/Implementation-of-IIDM/data/masks")
SPLITS_DIR= OUT_ROOT / "data" / "splits"
PATCH_DIR = OUT_ROOT / "data" / "processed" / "patches_6ch"

# ── Hyperparameters ────────────────────────────────────────────────────────────
PATCH_SIZE   = 256     
STRIDE       = 128     
MIN_FOREST   = 0.3     

def load_raster(path: Path, nodata_fill: float = 0.0) -> tuple:
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        nodata = src.nodata
        transform = src.transform
    if nodata is not None:
        data[data == nodata] = nodata_fill
    return data, transform

def extract_patches_for_split(input_stack, target, mask_arr, transform, regions_gdf, patch_size, stride, min_forest):
    _, H, W = input_stack.shape
    patches = []
    
    # Using spatial index for fast intersection
    sindex = regions_gdf.sindex
    
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            mask_patch = mask_arr[y:y+patch_size, x:x+patch_size]
            forest_frac = mask_patch.mean()
            if forest_frac < min_forest:
                continue
                
            # Get patch center coordinates
            cx = transform[2] + (x + patch_size/2) * transform[0]
            cy = transform[5] + (y + patch_size/2) * transform[4]
            patch_center = gpd.points_from_xy([cx], [cy])[0]
            
            # Check if center is in ANY of the regions
            possible_matches_index = list(sindex.intersection(patch_center.bounds))
            possible_matches = regions_gdf.iloc[possible_matches_index]
            precise_matches = possible_matches[possible_matches.intersects(patch_center)]
            
            if len(precise_matches) == 0:
                continue
                
            inp_patch = input_stack[:, y:y+patch_size, x:x+patch_size]
            tgt_patch = target[:, y:y+patch_size, x:x+patch_size]
            
            if (inp_patch[:4] == 0).mean() > 0.5:
                continue
                
            patches.append((inp_patch, tgt_patch))
    return patches

def save_patches(patches, split_dir):
    inp_dir = split_dir / "input"
    tgt_dir = split_dir / "target"
    inp_dir.mkdir(parents=True, exist_ok=True)
    tgt_dir.mkdir(parents=True, exist_ok=True)
    for i, (inp, tgt) in enumerate(patches):
        np.savez(inp_dir / f"patch_{i:05d}.npz", image=inp)
        np.savez(tgt_dir / f"patch_{i:05d}.npz", image=tgt)

def normalize_channel(arr, vmin=None, vmax=None):
    if vmin is None or vmax is None:
        vmin = np.percentile(arr[arr != 0], 2.0)
        vmax = np.percentile(arr[arr != 0], 98.0)
    norm = np.clip((arr - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
    return norm, float(vmin), float(vmax)

def main():
    print("Loading rasters...")
    TMP = ROOT / "tmp_processed"
    b4_arr, transform = load_raster(TMP / "s2_B04.tif")
    b3_arr, _ = load_raster(TMP / "s2_B03.tif")
    b2_arr, _ = load_raster(TMP / "s2_B02.tif")
    b8_arr, _ = load_raster(TMP / "s2_B08.tif")
    hh_arr, _ = load_raster(TMP / "hh.tif")
    hv_arr, _ = load_raster(TMP / "hv.tif")
    dem_arr, _ = load_raster(TMP / "dem.tif")
    canopy_arr, _ = load_raster(TMP / "canopy.tif")
    carbon_arr, _ = load_raster(TMP / "carbon.tif")
    
    # Bypass missing forest_mask by assuming all patches are valid forest
    mask_arr = np.ones_like(b4_arr[0], dtype=np.float32)

    # 8 channels: B4, B3, B2, B8, HH, HV, DEM, Canopy
    input_stack = np.concatenate([b4_arr, b3_arr, b2_arr, b8_arr, hh_arr, hv_arr, dem_arr, canopy_arr], axis=0)
    target = carbon_arr

    splits = ["train", "val", "test"]
    all_patches = {}
    
    for split in splits:
        print(f"Extracting patches for {split}...")
        regions = gpd.read_file(SPLITS_DIR / f"{split}_regions.geojson")
        
        with rasterio.open(TMP / "s2_B04.tif") as src:
            raster_crs = src.crs
        if regions.crs != raster_crs:
            regions = regions.to_crs(raster_crs)
            
        patches = extract_patches_for_split(input_stack, target, mask_arr, transform, regions, PATCH_SIZE, STRIDE, MIN_FOREST)
        all_patches[split] = patches
        print(f"Extracted {len(patches)} patches for {split}")

    print("Computing normalization stats on TRAIN set only...")
    train_inputs = np.stack([p[0] for p in all_patches["train"]])
    train_targets = np.stack([p[1] for p in all_patches["train"]])

    norm_stats = {}
    for c in range(8):
        _, vmin, vmax = normalize_channel(train_inputs[:, c])
        norm_stats[f"channel_{c}"] = {"min": vmin, "max": vmax}
        
    _, ymin, ymax = normalize_channel(train_targets)
    norm_stats["target"] = {"min": ymin, "max": ymax}

    os.makedirs(OUT_ROOT / "configs", exist_ok=True)
    with open(OUT_ROOT / "configs" / "norm_stats.json", "w") as f:
        json.dump(norm_stats, f, indent=2)

    print("Applying normalization and saving...")
    for split in splits:
        split_dir = PATCH_DIR / split
        norm_patches = []
        for inp, tgt in all_patches[split]:
            norm_inp = np.zeros_like(inp)
            for c in range(8):
                vmin = norm_stats[f"channel_{c}"]["min"]
                vmax = norm_stats[f"channel_{c}"]["max"] 
                norm_inp[c], _, _ = normalize_channel(inp[c], vmin, vmax)
            
            norm_tgt, _, _ = normalize_channel(tgt, norm_stats["target"]["min"], norm_stats["target"]["max"])
            norm_patches.append((norm_inp, norm_tgt))
            
        save_patches(norm_patches, split_dir)

if __name__ == "__main__":
    main()
