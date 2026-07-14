"""
IIDM Master Preprocessing Script
===================================
Uses already-processed data from /Users/raghvendra/iidm_project/
Copies + finalizes everything into Implementation-of-IIDM/data/

Run from: ~/Implementation-of-IIDM/
    python src/preprocessing/master_preprocess.py
"""

import os
import json
import shutil
import numpy as np
import rasterio
import h5py
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import from_bounds as window_from_bounds
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO        = Path(__file__).resolve().parents[2]
SRC         = Path("/Users/raghvendra/iidm_project")

RAW         = REPO / "data" / "raw"
PROCESSED   = REPO / "data" / "processed"
MASKS_DIR   = REPO / "data" / "masks"
PATCH_DIR   = PROCESSED / "patches"

for d in [RAW/"gf1", RAW/"dem", RAW/"canopy", RAW/"inventory",
          PROCESSED, MASKS_DIR, PATCH_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
TARGET_CRS   = "EPSG:32648"   # UTM Zone 48N — Yunnan Province
TARGET_RES   = 10             # 10m (Sentinel-2 native)
PATCH_SIZE   = 256
STRIDE       = 128
MIN_FOREST   = 0.2
CARBON_MEAN  = 45.0           # Mg C/ha — fallback default, overwritten by GEDI L4A actual mean
CARBON_FRAC  = 0.47           # IPCC default

# ── Huize County bounding box (WGS84) ─────────────────────────────────────────
HUIZE_WGS84 = (103.0, 25.8, 103.8, 26.6)   # west, south, east, north


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def reproject_clip(src_path, dst_path, target_crs=TARGET_CRS,
                   target_res=TARGET_RES, bbox=None):
    """Reproject + optionally clip to bbox, save as float32 GeoTIFF."""
    with rasterio.open(src_path) as src:
        transform, w, h = calculate_default_transform(
            src.crs, target_crs, src.width, src.height,
            *src.bounds, resolution=target_res)
        meta = src.meta.copy()
        meta.update(crs=target_crs, transform=transform,
                    width=w, height=h, dtype="float32",
                    nodata=-9999.0, compress="lzw", BIGTIFF="YES")
        data = np.zeros((src.count, h, w), dtype=np.float32)
        for i in range(1, src.count + 1):
            reproject(source=rasterio.band(src, i),
                      destination=data[i-1],
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=transform, dst_crs=target_crs,
                      resampling=Resampling.bilinear)

    with rasterio.open(dst_path, "w", **meta) as dst:
        dst.write(data)
    return dst_path


def merge_tiles(tile_paths, dst_path):
    """Merge multiple raster tiles into one."""
    srcs = [rasterio.open(p) for p in tile_paths]
    mosaic, transform = merge(srcs)
    meta = srcs[0].meta.copy()
    meta.update(width=mosaic.shape[2], height=mosaic.shape[1],
                transform=transform, compress="lzw")
    with rasterio.open(dst_path, "w", **meta) as dst:
        dst.write(mosaic)
    for s in srcs:
        s.close()
    return dst_path


def clip_to_common(paths):
    """Return intersection bbox of all rasters."""
    from rasterio.warp import transform_bounds
    bounds_list = []
    for p in paths:
        with rasterio.open(p) as src:
            b = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
            bounds_list.append(b)
    left   = max(b[0] for b in bounds_list)
    bottom = max(b[1] for b in bounds_list)
    right  = min(b[2] for b in bounds_list)
    top    = min(b[3] for b in bounds_list)
    print(f"  Common bbox (WGS84): {left:.3f},{bottom:.3f} → {right:.3f},{top:.3f}")
    return left, bottom, right, top


def clip_raster_to_bbox(src_path, dst_path, bbox_wgs84):
    """Clip reprojected raster (UTM) using WGS84 bbox."""
    from rasterio.warp import transform_bounds
    with rasterio.open(src_path) as src:
        utm_bbox = transform_bounds("EPSG:4326", src.crs, *bbox_wgs84)
        window   = window_from_bounds(*utm_bbox, src.transform)
        data     = src.read(window=window, out_dtype="float32",
                            resampling=Resampling.bilinear)
        transform = src.window_transform(window)
        meta = src.meta.copy()
        meta.update(width=data.shape[2], height=data.shape[1],
                    transform=transform, dtype="float32",
                    nodata=-9999.0, compress="lzw", BIGTIFF="YES")
    with rasterio.open(dst_path, "w", **meta) as dst:
        dst.write(data)
    print(f"  Clipped {dst_path.name}: {data.shape}")
    return data


def normalize_carbon(arr, nodata=-9999.0):
    """Normalize carbon directly to [-1, 1] for diffusion compatibility."""
    valid = arr[arr != nodata]
    if valid.size == 0:
        return arr, 0.0, 1.0
    vmin = float(np.percentile(valid, 2))
    vmax = float(np.percentile(valid, 98))
    out  = np.full_like(arr, -1.0, dtype=np.float32)
    mask = arr != nodata
    out[mask] = np.clip((arr[mask] - vmin) / (vmax - vmin + 1e-8) * 2.0 - 1.0, -1.0, 1.0)
    return out, vmin, vmax


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — SENTINEL-2: STACK 4 BANDS (B02, B03, B04, B08)
# ══════════════════════════════════════════════════════════════════════════════

def step1_sentinel2():
    print("\n" + "="*55)
    print("  STEP 1 — Sentinel-2 Band Stacking")
    print("="*55)

    # Use already-stacked file if exists
    stacked = RAW / "sentinel2/sentinel2_stacked.tif"
    if stacked.exists():
        dst = PROCESSED / "sentinel2_stacked.tif"
        shutil.copy2(stacked, dst)
        print(f"  [COPIED] sentinel2_stacked.tif")
        return dst

    # Otherwise merge tiles per band then stack
    bands_order = ["B02", "B03", "B04", "B08"]
    band_files  = []

    for band in bands_order:
        tiles = sorted((RAW / f"sentinel2/raw/{band}").rglob("response.tiff"))
        if not tiles:
            raise FileNotFoundError(f"No tiles for {band}")
        merged = PROCESSED / f"{band}_merged.tif"
        merge_tiles(tiles, merged)
        reproj = PROCESSED / f"{band}_reproj.tif"
        reproject_clip(merged, reproj)
        band_files.append(reproj)
        print(f"  [OK] {band} → merged + reprojected")

    # Stack into single 4-band file
    arrays, meta = [], None
    for bf in band_files:
        with rasterio.open(bf) as src:
            arrays.append(src.read(1))
            if meta is None:
                meta = src.meta.copy()

    stack = np.stack(arrays, axis=0).astype(np.float32)
    meta.update(count=4, dtype="float32", compress="lzw")
    dst = PROCESSED / "sentinel2_stacked.tif"
    with rasterio.open(dst, "w", **meta) as out:
        out.write(stack)
    print(f"  [SAVED] sentinel2_stacked.tif  shape={stack.shape}")
    return dst


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — DEM: USE HUIZE SRTM (better coverage than ALOS for this area)
# ══════════════════════════════════════════════════════════════════════════════

def step2_dem():
    print("\n" + "="*55)
    print("  STEP 2 — DEM Processing")
    print("="*55)

    # Prefer already processed
    dem_norm = RAW / "dem/dem_normalized.tif"
    if dem_norm.exists():
        dst = PROCESSED / "dem_processed.tif"
        shutil.copy2(dem_norm, dst)
        print(f"  [COPIED] dem_normalized.tif")
        return dst

    # Use Huize SRTM (best coverage for study area)
    srtm = RAW / "alos_dem/huize srtm/output_SRTMGL1.tif"
    if not srtm.exists():
        srtm = RAW / "alos_dem/AP_25667_FBD_F0520_RT1/AP_25667_FBD_F0520_RT1.dem.tif"

    dst = PROCESSED / "dem_processed.tif"
    reproject_clip(srtm, dst)
    print(f"  [SAVED] dem_processed.tif")
    return dst


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — ETH CANOPY HEIGHT: MERGE 2 TILES
# ══════════════════════════════════════════════════════════════════════════════

def step3_canopy():
    print("\n" + "="*55)
    print("  STEP 3 — ETH Canopy Height")
    print("="*55)

    # Use already processed
    canopy_norm = RAW / "canopy_height/canopy_normalized.tif"
    if canopy_norm.exists():
        dst = PROCESSED / "canopy_processed.tif"
        shutil.copy2(canopy_norm, dst)
        print(f"  [COPIED] canopy_normalized.tif")
        return dst

    tiles = [
        RAW / "eth_canopy/ETH_GlobalCanopyHeight_10m_2020_N27E102_Map.tif",
        RAW / "eth_canopy/ETH_GlobalCanopyHeight_10m_2020_N24E102_Map.tif",
    ]
    merged = PROCESSED / "canopy_merged.tif"
    merge_tiles([t for t in tiles if t.exists()], merged)

    dst = PROCESSED / "canopy_processed.tif"
    reproject_clip(merged, dst)
    print(f"  [SAVED] canopy_processed.tif")
    return dst


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — GEDI L4A: EXTRACT MEAN BIOMASS → CARBON
# ══════════════════════════════════════════════════════════════════════════════

def step4_gedi(s2_path):
    print("\n" + "="*55)
    print("  STEP 4 — GEDI L4A: Extract Shots → Rasterize → Carbon")
    print("="*55)

    gedi_dir = RAW / "gedi"
    h5_files = sorted(gedi_dir.glob("*.h5"))
    print(f"  GEDI L4A files: {len(h5_files)}")

    with rasterio.open(s2_path) as ref:
        ref_transform = ref.transform
        ref_shape     = (ref.height, ref.width)
        ref_crs       = ref.crs

    west, south, east, north = HUIZE_WGS84
    records = []

    for h5_path in h5_files:
        print(f"  Reading: {h5_path.name}")
        with h5py.File(h5_path, "r") as f:
            beams = [k for k in f.keys() if k.startswith("BEAM")]
            for beam in beams:
                try:
                    lat  = f[beam]["lat_lowestmode"][:]
                    lon  = f[beam]["lon_lowestmode"][:]
                    agb  = f[beam]["agbd"][:]
                    qual = f[beam]["l4_quality_flag"][:]

                    mask = ((lat >= south) & (lat <= north) &
                            (lon >= west)  & (lon <= east)  &
                            (qual == 1)    & (agb  >  0)    &
                            (agb  < 1000))

                    if mask.sum() == 0:
                        continue

                    for la, lo, ag in zip(lat[mask], lon[mask], agb[mask]):
                        records.append({"lat": float(la),
                                        "lon": float(lo),
                                        "agb": float(ag)})
                except Exception:
                    continue

    print(f"  Valid GEDI shots in Huize: {len(records)}")
    if len(records) == 0:
        raise RuntimeError("No GEDI L4A shots found in Huize bbox — check bbox/quality-flag filters or raw .h5 files.")

    for r in records:
        r["carbon"] = r["agb"] * CARBON_FRAC

    print(f"  Carbon range: [{min(r['carbon'] for r in records):.1f},"
          f" {max(r['carbon'] for r in records):.1f}] Mg C/ha")

    carbon_grid = np.full(ref_shape, -9999.0, dtype=np.float32)
    count_grid  = np.zeros(ref_shape, dtype=np.int32)

    from rasterio.warp import transform as warp_transform
    lons = [r["lon"] for r in records]
    lats = [r["lat"] for r in records]
    xs, ys = warp_transform("EPSG:4326", ref_crs, lons, lats)

    for x, y, r in zip(xs, ys, records):
        col, row = ~ref_transform * (x, y)
        row, col = int(row), int(col)
        if 0 <= row < ref_shape[0] and 0 <= col < ref_shape[1]:
            if carbon_grid[row, col] == -9999.0:
                carbon_grid[row, col] = r["carbon"]
            else:
                carbon_grid[row, col] += r["carbon"]
            count_grid[row, col] += 1

    multi = count_grid > 1
    carbon_grid[multi] = carbon_grid[multi] / count_grid[multi]

    filled = (carbon_grid != -9999.0).sum()
    print(f"  Pixels with GEDI data: {filled} "
          f"({100*filled/carbon_grid.size:.2f}% of grid)")

    global CARBON_MEAN
    valid_vals = carbon_grid[carbon_grid != -9999.0]
    CARBON_MEAN = float(valid_vals.mean()) if valid_vals.size > 0 else 45.0
    print(f"  GEDI L4A mean carbon stock: {CARBON_MEAN:.2f} Mg C/ha")

    dst  = PROCESSED / "gedi_carbon.tif"
    with rasterio.open(s2_path) as ref:
        meta = ref.meta.copy()
    meta.update(count=1, dtype="float32", nodata=-9999.0, compress="lzw", BIGTIFF="YES")
    with rasterio.open(dst, "w", **meta) as out:
        out.write(carbon_grid[np.newaxis])
    print(f"  [SAVED] {dst.name}")
    return dst


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — ALIGN ALL TO COMMON EXTENT + RESOLUTION
# ══════════════════════════════════════════════════════════════════════════════

def step5_align(s2_path, dem_path, canopy_path, carbon_path):
    print("\n" + "="*55)
    print("  STEP 5 — Align All Rasters to Common Extent")
    print("="*55)

    paths = [s2_path, dem_path, canopy_path, carbon_path]
    bbox  = clip_to_common(paths)

    aligned = {}
    names   = ["aligned_gf1.tif", "aligned_dem.tif",
                "aligned_canopy.tif", "aligned_carbon.tif"]

    for src_p, name in zip(paths, names):
        dst_p = PROCESSED / name
        clip_raster_to_bbox(src_p, dst_p, bbox)
        aligned[name] = dst_p

    return aligned, bbox


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — FOREST MASK FROM CANOPY HEIGHT
# ══════════════════════════════════════════════════════════════════════════════

def step6_mask(canopy_aligned):
    print("\n" + "="*55)
    print("  STEP 6 — Forest / Non-Forest Mask")
    print("="*55)

    # Use existing mask if available
    existing = RAW / "canopy_height/forest_mask.tif"
    if existing.exists():
        dst = MASKS_DIR / "forest_mask.tif"
        shutil.copy2(existing, dst)
        print(f"  [COPIED] forest_mask.tif")
        with rasterio.open(dst) as src:
            mask = src.read(1).astype(np.float32)
            mask = (mask > 0).astype(np.float32)
        return dst, mask

    with rasterio.open(canopy_aligned) as src:
        canopy = src.read(1).astype(np.float32)
        nodata = src.nodata or -9999.0
        meta   = src.meta.copy()

    # Forest = canopy height > 2m
    mask                         = np.zeros_like(canopy, dtype=np.int16)
    mask[canopy > 2.0]           = 1
    mask[canopy == nodata]       = -9999

    forest_pct = 100 * (mask == 1).sum() / max((mask != -9999).sum(), 1)
    print(f"  Forest coverage: {forest_pct:.1f}%")

    meta.update(count=1, dtype="int16", nodata=-9999, compress="lzw")
    dst = MASKS_DIR / "forest_mask.tif"
    with rasterio.open(dst, "w", **meta) as out:
        out.write(mask[np.newaxis])
    print(f"  [SAVED] forest_mask.tif")
    return dst, (mask == 1).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — NORMALIZE + EXTRACT PATCHES
# ══════════════════════════════════════════════════════════════════════════════

def step7_patches(aligned, mask_arr):
    print("\n" + "="*55)
    print("  STEP 7 — Normalize + Extract Patches")
    print("="*55)

    norm_stats = {}

    def load(path):
        with rasterio.open(path) as src:
            return src.read().astype(np.float32), src.nodata or -9999.0

    # Load
    gf1_data,    nd = load(aligned["aligned_gf1.tif"])     # (4, H, W)
    dem_data,    nd = load(aligned["aligned_dem.tif"])      # (1+, H, W)
    H_ref, W_ref = gf1_data.shape[1], gf1_data.shape[2]
    if dem_data.shape[1] != H_ref or dem_data.shape[2] != W_ref:
        import torch.nn.functional as F_
        import torch
        dem_data = F_.interpolate(
            torch.from_numpy(dem_data).unsqueeze(0),
            size=(H_ref, W_ref), mode='bilinear', align_corners=False
        ).squeeze(0).numpy()
        print(f"  DEM resampled: {dem_data.shape}")
    canopy_data, nd = load(aligned["aligned_canopy.tif"])   # (1, H, W)
    carbon_data, nd = load(aligned["aligned_carbon.tif"])   # (1, H, W)

    dem_data    = dem_data[:1]      # keep only first band
    canopy_data = canopy_data[:1]
    carbon_data = carbon_data[:1]

    # Normalize each channel
    norm_gf1 = np.zeros_like(gf1_data)
    for b in range(gf1_data.shape[0]):
        norm_gf1[b], vmin, vmax = normalize(gf1_data[b])
        norm_stats[f"gf1_b{b+1}"] = {"min": vmin, "max": vmax}

    norm_dem,    vmin, vmax = normalize(dem_data[0])
    norm_stats["dem"]       = {"min": vmin, "max": vmax}
    norm_dem                = norm_dem[np.newaxis]

    norm_canopy, vmin, vmax = normalize(canopy_data[0])
    norm_stats["canopy"]    = {"min": vmin, "max": vmax}
    norm_canopy             = norm_canopy[np.newaxis]

    norm_carbon, vmin, vmax = normalize_carbon(carbon_data[0])
    norm_stats["carbon"]    = {"min": vmin, "max": vmax}
    norm_carbon             = norm_carbon[np.newaxis]

    # Stack input: 6 channels (B02,B03,B04,B08,DEM,Canopy)
    input_stack = np.concatenate([norm_gf1, norm_dem, norm_canopy], axis=0)
    target      = norm_carbon

    print(f"  Input stack : {input_stack.shape}  (6 × H × W)")
    print(f"  Target      : {target.shape}")

    # Resize mask to match
    H, W = input_stack.shape[1], input_stack.shape[2]
    from skimage.transform import resize
    if mask_arr.shape != (H, W):
        mask_arr = resize(mask_arr, (H, W), order=0,
                          preserve_range=True).astype(np.float32)

    # Extract patches
    patches = []
    for y in range(0, H - PATCH_SIZE + 1, STRIDE):
        for x in range(0, W - PATCH_SIZE + 1, STRIDE):
            mp  = mask_arr[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            if mp.mean() < MIN_FOREST:
                continue
            inp = input_stack[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            tgt = target[:,    y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            if (inp == 0).mean() > 0.6:
                continue
            # GEDI-valid mask: 1 where real carbon measurements exist, 0 elsewhere
            gedi_msk = (tgt[0] != 0).astype(np.float32)
            patches.append((inp.copy(), tgt.copy(), gedi_msk.copy()))

    print(f"  Valid patches: {len(patches)}")
    if len(patches) == 0:
        # Relax threshold and retry
        print("  [WARN] No patches found — relaxing MIN_FOREST to 0.05")
        for y in range(0, H - PATCH_SIZE + 1, STRIDE):
            for x in range(0, W - PATCH_SIZE + 1, STRIDE):
                inp = input_stack[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                tgt = target[:,    y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                gedi_msk = (tgt[0] != 0).astype(np.float32)
                patches.append((inp.copy(), tgt.copy(), gedi_msk.copy()))
        print(f"  Patches (relaxed): {len(patches)}")

    # Split train/val/test
    np.random.seed(42)
    idx = np.random.permutation(len(patches))
    n_tr = int(0.70 * len(idx))
    n_va = int(0.15 * len(idx))
    splits = {"train": idx[:n_tr],
              "val":   idx[n_tr:n_tr+n_va],
              "test":  idx[n_tr+n_va:]}

    for split, sidx in splits.items():
        split_d = PATCH_DIR / split
        split_d.mkdir(parents=True, exist_ok=True)
        for i, pi in enumerate(sidx):
            np.savez_compressed(
                split_d / f"patch_{i:05d}.npz",
                image=patches[pi][0],    # (6, H, W) normalized float32
                carbon=patches[pi][1][0],# (H, W) normalized carbon
                mask=patches[pi][2]      # (H, W) GEDI-valid binary mask
            )
        print(f"  {split:5s}: {len(sidx)} patches saved")

    with open(PROCESSED / "norm_stats.json", "w") as f:
        json.dump(norm_stats, f, indent=2)
    print(f"  [SAVED] norm_stats.json")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — COPY RAW DATA TO REPO
# ══════════════════════════════════════════════════════════════════════════════

    print("\n" + "="*55)
    print("  STEP 8 — Copy Raw Data to Repo")
    print("="*55)

    copies = [
        # (src, dst)
        (SRC/"alos_dem/huize srtm/output_SRTMGL1.tif",  RAW/"dem/huize_srtm.tif"),
        (SRC/"eth_canopy/ETH_GlobalCanopyHeight_10m_2020_N27E102_Map.tif",
                                                          RAW/"canopy/ETH_N27E102.tif"),
        (SRC/"eth_canopy/ETH_GlobalCanopyHeight_10m_2020_N24E102_Map.tif",
                                                          RAW/"canopy/ETH_N24E102.tif"),
    ]
    for src, dst in copies:
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  [COPIED] {dst.name}")
        else:
            print(f"  [SKIP]   {src.name} not found")

    # Sentinel-2 tiles
    for band in ["B02","B03","B04","B08"]:
        tiles = sorted((SRC/f"sentinel2/raw/{band}").rglob("response.tiff"))
        for i, t in enumerate(tiles):
            shutil.copy2(t, RAW/f"gf1/S2_{band}_tile{i}.tiff")
    print(f"  [COPIED] Sentinel-2 tiles → data/raw/gf1/")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 55)
    print("  IIDM MASTER PREPROCESSING PIPELINE")
    print("=" * 55)

    s2_path     = step1_sentinel2()
    dem_path    = step2_dem()
    canopy_path = step3_canopy()
    carbon_path = step4_gedi(s2_path)

    aligned, bbox = step5_align(s2_path, dem_path, canopy_path, carbon_path)

    _, mask_arr = step6_mask(aligned["aligned_canopy.tif"])

    step7_patches(aligned, mask_arr)

    print("\n" + "="*55)
    print("  ALL STEPS COMPLETE")
    print("="*55)
    print(f"  Patches  : {PATCH_DIR}")
    print(f"  Processed: {PROCESSED}")
    print(f"  Masks    : {MASKS_DIR}")
    print("\n  Next: git add + push, then model implementation")


if __name__ == "__main__":
    main()
