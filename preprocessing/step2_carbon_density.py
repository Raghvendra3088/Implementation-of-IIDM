"""
IIDM Preprocessing - Step 2: Carbon Stock Distribution Density
===============================================================
Paper reference: Appendix A, Section 2
- Co-registers all raster layers to same CRS, resolution, extent
- Normalizes ETH canopy height as spatial weight
- Distributes plot-level carbon stock spatially using canopy height weights
- Outputs: aligned rasters + carbon density map (.tif)

Input  : data/raw/gf1/       (Sentinel-2 or GF-1, .tif)
         data/raw/dem/        (ALOS PALSAR DEM, .tif)
         data/raw/canopy/     (ETH canopy height, .tif)
         data/processed/carbon_stock.csv  (from Step 1)
Output : data/processed/aligned_gf1.tif
         data/processed/aligned_dem.tif
         data/processed/aligned_canopy.tif
         data/processed/carbon_density.tif
"""

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums    import Resampling
from rasterio.warp     import (calculate_default_transform,
                                reproject, Resampling as WarpResampling)
from rasterio.transform import from_bounds
from pathlib            import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
RAW        = ROOT / "data" / "raw"
PROCESSED  = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

# Target CRS and resolution (matching paper: 16m GF-1 as reference)
TARGET_CRS = "EPSG:32648"   # UTM Zone 48N — covers Yunnan Province
TARGET_RES = 16             # metres (GF-1 native resolution)


# ── Utility functions ──────────────────────────────────────────────────────────

def find_file(directory: Path, extensions=(".tif", ".tiff")) -> Path:
    """Find first raster file in a directory."""
    for ext in extensions:
        files = sorted(directory.glob(f"*{ext}"))
        if files:
            return files[0]
    raise FileNotFoundError(
        f"No raster file ({extensions}) found in {directory}\n"
        f"Please check your data/raw/ folder structure."
    )


def reproject_raster(src_path: Path, dst_path: Path,
                     target_crs: str, target_res: float,
                     resampling=WarpResampling.bilinear) -> None:
    """
    Reproject and resample a raster to target CRS and resolution.
    Saves result as Cloud-Optimised GeoTIFF.
    """
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height,
            *src.bounds, resolution=target_res
        )
        kwargs = src.meta.copy()
        kwargs.update({
            "crs"       : target_crs,
            "transform" : transform,
            "width"     : width,
            "height"    : height,
            "dtype"     : "float32",
            "nodata"    : -9999.0,
            "compress"  : "lzw",
        })

        with rasterio.open(dst_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source      = rasterio.band(src, i),
                    destination = rasterio.band(dst, i),
                    src_transform = src.transform,
                    src_crs     = src.crs,
                    dst_transform = transform,
                    dst_crs     = target_crs,
                    resampling  = resampling,
                )
    print(f"  [OK] {dst_path.name}  ({width}×{height} px)")


def clip_to_common_extent(*raster_paths: Path) -> tuple:
    """Compute the intersection bounding box of all rasters."""
    bounds_list = []
    for p in raster_paths:
        with rasterio.open(p) as src:
            bounds_list.append(src.bounds)

    left   = max(b.left   for b in bounds_list)
    bottom = max(b.bottom for b in bounds_list)
    right  = min(b.right  for b in bounds_list)
    top    = min(b.top    for b in bounds_list)

    if left >= right or bottom >= top:
        raise ValueError(
            "Rasters do not overlap. Check that all datasets cover the same area."
        )
    return left, bottom, right, top


def clip_raster(src_path: Path, dst_path: Path,
                bounds: tuple, target_res: float) -> np.ndarray:
    """Clip raster to bounds and return array."""
    left, bottom, right, top = bounds

    with rasterio.open(src_path) as src:
        from rasterio.windows import from_bounds as window_from_bounds
        window    = window_from_bounds(left, bottom, right, top, src.transform)
        data      = src.read(window=window,
                             out_dtype="float32",
                             resampling=Resampling.bilinear)
        transform = src.window_transform(window)
        meta      = src.meta.copy()

    meta.update({
        "width"     : data.shape[2],
        "height"    : data.shape[1],
        "transform" : transform,
        "dtype"     : "float32",
        "nodata"    : -9999.0,
        "compress"  : "lzw",
    })
    with rasterio.open(dst_path, "w", **meta) as dst:
        dst.write(data)

    print(f"  [CLIPPED] {dst_path.name}  shape={data.shape}")
    return data


def normalize_array(arr: np.ndarray, nodata: float = -9999.0) -> np.ndarray:
    """Min-max normalize array, ignoring nodata pixels."""
    valid = arr[arr != nodata]
    if valid.size == 0:
        return arr
    arr_norm        = arr.astype(np.float32).copy()
    vmin, vmax      = valid.min(), valid.max()
    mask            = arr != nodata
    arr_norm[mask]  = (arr[mask] - vmin) / (vmax - vmin + 1e-8)
    arr_norm[~mask] = -9999.0
    return arr_norm


def distribute_carbon_spatially(canopy_arr: np.ndarray,
                                 carbon_mean: float,
                                 nodata: float = -9999.0) -> np.ndarray:
    """
    Paper Section A.2:
    Carbon density map = plot-level carbon stock × normalised canopy height weight.

    Each pixel carbon = carbon_mean × (canopy_height_norm / sum(canopy_height_norm))
    × total_pixels   (to preserve total carbon budget)
    """
    valid_mask  = canopy_arr != nodata
    norm_canopy = normalize_array(canopy_arr, nodata)

    # Weight = normalised canopy height (proxy for biomass distribution)
    total_weight = norm_canopy[valid_mask].sum()
    if total_weight == 0:
        raise ValueError("Canopy height sum is zero — check your canopy raster.")

    carbon_density              = np.full_like(canopy_arr, nodata, dtype=np.float32)
    n_valid                     = valid_mask.sum()
    carbon_density[valid_mask]  = (
        carbon_mean
        * (norm_canopy[valid_mask] / total_weight)
        * n_valid
    )
    return carbon_density


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  STEP 2 — Carbon Density Mapping")
    print("=" * 55)

    # ── 1. Locate raw rasters ──────────────────────────────────────────────────
    print("\n[1/5] Locating raw rasters ...")
    gf1_raw    = find_file(RAW / "gf1")
    dem_raw    = find_file(RAW / "dem")
    canopy_raw = find_file(RAW / "canopy")
    print(f"  GF-1   : {gf1_raw.name}")
    print(f"  DEM    : {dem_raw.name}")
    print(f"  Canopy : {canopy_raw.name}")

    # ── 2. Reproject all to common CRS + resolution ────────────────────────────
    print(f"\n[2/5] Reprojecting to {TARGET_CRS} @ {TARGET_RES}m ...")
    gf1_repr    = PROCESSED / "reproj_gf1.tif"
    dem_repr    = PROCESSED / "reproj_dem.tif"
    canopy_repr = PROCESSED / "reproj_canopy.tif"

    reproject_raster(gf1_raw,    gf1_repr,    TARGET_CRS, TARGET_RES,
                     WarpResampling.bilinear)
    reproject_raster(dem_raw,    dem_repr,    TARGET_CRS, TARGET_RES,
                     WarpResampling.bilinear)
    reproject_raster(canopy_raw, canopy_repr, TARGET_CRS, TARGET_RES,
                     WarpResampling.bilinear)

    # ── 3. Clip to common spatial extent ──────────────────────────────────────
    print("\n[3/5] Clipping to common extent ...")
    bounds = clip_to_common_extent(gf1_repr, dem_repr, canopy_repr)
    print(f"  Intersection bbox: {[round(b,2) for b in bounds]}")

    gf1_clip    = PROCESSED / "aligned_gf1.tif"
    dem_clip    = PROCESSED / "aligned_dem.tif"
    canopy_clip = PROCESSED / "aligned_canopy.tif"

    clip_raster(gf1_repr,    gf1_clip,    bounds, TARGET_RES)
    clip_raster(dem_repr,    dem_clip,    bounds, TARGET_RES)
    canopy_data = clip_raster(canopy_repr, canopy_clip, bounds, TARGET_RES)

    # ── 4. Load mean carbon stock from Step 1 ─────────────────────────────────
    print("\n[4/5] Loading carbon stock from Step 1 ...")
    cs_path = PROCESSED / "carbon_stock.csv"
    if not cs_path.exists():
        raise FileNotFoundError(
            f"{cs_path} not found. Run step1_carbon_stock.py first."
        )
    df           = pd.read_csv(cs_path)
    carbon_mean  = df["carbon_stock_MgCha"].mean()
    print(f"  Mean carbon stock : {carbon_mean:.2f} Mg C/ha")

    # ── 5. Create spatial carbon density map ──────────────────────────────────
    print("\n[5/5] Creating carbon density raster ...")
    canopy_band   = canopy_data[0]                      # single-band canopy height

    with rasterio.open(canopy_clip) as src:
        nodata    = src.nodata if src.nodata is not None else -9999.0
        meta      = src.meta.copy()

    carbon_map = distribute_carbon_spatially(canopy_band, carbon_mean, nodata)

    meta.update({"count": 1, "dtype": "float32",
                 "nodata": -9999.0, "compress": "lzw"})
    out_path = PROCESSED / "carbon_density.tif"
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(carbon_map[np.newaxis, :, :])

    # Stats
    valid = carbon_map[carbon_map != -9999.0]
    print(f"\n[RESULT] Carbon Density Map Stats (Mg C/ha):")
    print(f"  Mean  : {valid.mean():.4f}")
    print(f"  Std   : {valid.std():.4f}")
    print(f"  Min   : {valid.min():.4f}")
    print(f"  Max   : {valid.max():.4f}")
    print(f"\n[SAVED] {out_path}")
    print("\n[DONE] Aligned rasters:")
    print(f"  {gf1_clip}")
    print(f"  {dem_clip}")
    print(f"  {canopy_clip}")
    print(f"  {out_path}")


if __name__ == "__main__":
    main()
