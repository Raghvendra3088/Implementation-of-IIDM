"""
Sentinel-2 Preprocessing — IIDM Project
Author: RTripathi
"""
import os
import glob
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask as rio_mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geopandas as gpd
from shapely.geometry import mapping, shape

PROJECT    = os.path.expanduser("~/iidm_project")
RAW_DIR    = os.path.join(PROJECT, "sentinel2/raw")
OUTPUT_DIR = os.path.join(PROJECT, "data/processed/sentinel2")
AOI_FILE   = os.path.join(PROJECT, "huize_boundary/huize_boundary.geojson")
TARGET_CRS = "EPSG:32648"
TARGET_RES = 16.0
BANDS      = ["B02", "B03", "B04", "B08"]

def step(msg):
    print(f"\n{'─'*50}\n  {msg}\n{'─'*50}")

def load_aoi():
    gdf = gpd.read_file(AOI_FILE)
    gdf = gdf.set_crs("EPSG:4326") if gdf.crs is None else gdf.to_crs("EPSG:4326")
    print(f"  ✅ AOI loaded")
    return [mapping(g) for g in gdf.geometry]

def merge_tiles(band):
    # response.tiff files dhundho — har tile folder mein
    pattern = os.path.join(RAW_DIR, band, "*", "*", "response.tiff")
    tile_paths = sorted(glob.glob(pattern))
    print(f"  ℹ️  Pattern: {pattern}")
    print(f"  ℹ️  Found: {len(tile_paths)} tiles")
    if not tile_paths:
        print(f"  ❌ No tiles found for {band}")
        return None
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    datasets = [rasterio.open(p) for p in tile_paths]
    mosaic, transform = merge(datasets)
    meta = datasets[0].meta.copy()
    meta.update({"height": mosaic.shape[1], "width": mosaic.shape[2], "transform": transform})
    for d in datasets: d.close()
    out = os.path.join(OUTPUT_DIR, f"{band}_merged.tif")
    with rasterio.open(out, "w", **meta) as dst: dst.write(mosaic)
    print(f"  ✅ Merged → {out}")
    return out

def clip_raster(input_path, shapes, band):
    with rasterio.open(input_path) as src:
        gdf_s = gpd.GeoDataFrame(geometry=[shape(s) for s in shapes], crs="EPSG:4326")
        if src.crs and str(src.crs) != "EPSG:4326":
            gdf_s = gdf_s.to_crs(src.crs)
        out_img, out_tf = rio_mask(src, [mapping(g) for g in gdf_s.geometry], crop=True)
        meta = src.meta.copy()
        meta.update({"height": out_img.shape[1], "width": out_img.shape[2], "transform": out_tf})
    out = os.path.join(OUTPUT_DIR, f"{band}_clipped.tif")
    with rasterio.open(out, "w", **meta) as dst: dst.write(out_img)
    print(f"  ✅ Clipped → {out}")
    return out

def reproject_raster(input_path, band):
    with rasterio.open(input_path) as src:
        tf, w, h = calculate_default_transform(
            src.crs, TARGET_CRS, src.width, src.height,
            *src.bounds, resolution=TARGET_RES
        )
        meta = src.meta.copy()
        meta.update({"crs": TARGET_CRS, "transform": tf, "width": w, "height": h})
        out = os.path.join(OUTPUT_DIR, f"{band}_reprojected.tif")
        with rasterio.open(out, "w", **meta) as dst:
            for i in range(1, src.count+1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=tf, dst_crs=TARGET_CRS,
                    resampling=Resampling.bilinear
                )
    print(f"  ✅ Reprojected → {out}")
    return out

def normalize(input_path, band):
    with rasterio.open(input_path) as src:
        data = src.read(1).astype(np.float32)
        meta = src.meta.copy()
        nd = src.nodata
    if nd is not None: data = np.where(data == nd, np.nan, data)
    valid = data[~np.isnan(data)]
    mn, mx = valid.min(), valid.max()
    norm = (data - mn) / (mx - mn + 1e-8)
    norm = np.where(np.isnan(data), -9999.0, norm)
    meta.update({"dtype": "float32", "nodata": -9999.0})
    out = os.path.join(OUTPUT_DIR, f"{band}_normalized.tif")
    with rasterio.open(out, "w", **meta) as dst:
        dst.write(norm.astype(np.float32), 1)
    print(f"  ✅ Normalised [min={mn:.4f}, max={mx:.4f}] → {out}")
    return out

def stack_bands(band_paths):
    datasets = [rasterio.open(p) for p in band_paths]
    meta = datasets[0].meta.copy()
    meta.update({"count": 4, "dtype": "float32"})
    out = os.path.join(OUTPUT_DIR, "sentinel2_stacked.tif")
    with rasterio.open(out, "w", **meta) as dst:
        for i, ds in enumerate(datasets, 1):
            dst.write(ds.read(1), i)
    for d in datasets: d.close()
    print(f"  ✅ Stacked (4 bands) → {out}")
    return out

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  SENTINEL-2 PREPROCESSING | RTripathi")
    print("="*50)
    shapes = load_aoi()
    norm_paths = []
    for band in BANDS:
        step(f"Band {band}")
        merged  = merge_tiles(band)
        if not merged: continue
        clipped = clip_raster(merged, shapes, band)
        reproj  = reproject_raster(clipped, band)
        norm    = normalize(reproj, band)
        norm_paths.append(norm)
    if len(norm_paths) == 4:
        step("Stacking all 4 bands")
        stack_bands(norm_paths)
    print("\n" + "="*50)
    print("  ✅ SENTINEL-2 PREPROCESSING COMPLETE!")
    print(f"  📁 {OUTPUT_DIR}/")
    print("  Model input → sentinel2_stacked.tif")
    print("="*50)
