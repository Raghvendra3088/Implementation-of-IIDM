"""
preprocess.py - IIDM Project
Author: RTripathi
Study Area: Huize County, Yunnan, China
"""

import os
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.merge import merge
import geopandas as gpd
from shapely.geometry import mapping
from scipy.ndimage import generic_filter

HOME = os.path.expanduser("~")
PROJECT = os.path.join(HOME, "iidm_project")

DEM_FILE     = os.path.join(PROJECT, "alos_dem/huize srtm/output_SRTMGL1.tif")
CANOPY_TILE1 = os.path.join(PROJECT, "eth_canopy/ETH_GlobalCanopyHeight_10m_2020_N24E102_Map.tif")
CANOPY_TILE2 = os.path.join(PROJECT, "eth_canopy/ETH_GlobalCanopyHeight_10m_2020_N27E102_Map.tif")
AOI_FILE     = os.path.join(PROJECT, "huize_boundary/huize_boundary.geojson")
OUTPUT_DIR   = os.path.join(PROJECT, "data/processed")
TARGET_CRS   = "EPSG:32648"
TARGET_RES   = 16.0
FOREST_THR   = 2.0

def step(msg):
    print(f"\n{'─'*55}\n  {msg}\n{'─'*55}")

def print_info(path):
    with rasterio.open(path) as src:
        print(f"     Size: {src.width}x{src.height}px | CRS: {src.crs} | Res: {src.res}")

def load_aoi(path):
    gdf = gpd.read_file(path)
    gdf = gdf.set_crs("EPSG:4326") if gdf.crs is None else gdf.to_crs("EPSG:4326")
    shapes = [mapping(g) for g in gdf.geometry]
    print(f"  ✅ AOI loaded | Bounds: {gdf.total_bounds.round(3)}")
    return shapes

def merge_tiles(tile_paths, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    datasets = [rasterio.open(p) for p in tile_paths]
    mosaic, out_transform = merge(datasets)
    meta = datasets[0].meta.copy()
    meta.update({"height": mosaic.shape[1], "width": mosaic.shape[2], "transform": out_transform})
    for d in datasets: d.close()
    with rasterio.open(output_path, "w", **meta) as dst: dst.write(mosaic)
    print(f"  ✅ Tiles merged → {output_path}")
    return output_path

def clip_raster(input_path, shapes, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(input_path) as src:
        gdf_s = gpd.GeoDataFrame(geometry=[__import__('shapely').geometry.shape(s) for s in shapes], crs="EPSG:4326")
        if src.crs and str(src.crs) != "EPSG:4326":
            gdf_s = gdf_s.to_crs(src.crs)
        out_img, out_tf = rio_mask(src, [mapping(g) for g in gdf_s.geometry], crop=True)
        meta = src.meta.copy()
        meta.update({"height": out_img.shape[1], "width": out_img.shape[2], "transform": out_tf})
    with rasterio.open(output_path, "w", **meta) as dst: dst.write(out_img)
    print(f"  ✅ Clipped → {output_path}")
    return output_path

def reproject_raster(input_path, output_path, crs, res):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(input_path) as src:
        tf, w, h = calculate_default_transform(src.crs, crs, src.width, src.height, *src.bounds, resolution=res)
        meta = src.meta.copy()
        meta.update({"crs": crs, "transform": tf, "width": w, "height": h})
        with rasterio.open(output_path, "w", **meta) as dst:
            for i in range(1, src.count+1):
                reproject(source=rasterio.band(src,i), destination=rasterio.band(dst,i),
                          src_transform=src.transform, src_crs=src.crs,
                          dst_transform=tf, dst_crs=crs, resampling=Resampling.bilinear)
    print(f"  ✅ Reprojected ({crs} @ {res}m) → {output_path}")
    return output_path

def fill_holes(input_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(input_path) as src:
        data = src.read(1).astype(np.float32)
        meta = src.meta.copy()
        nd = src.nodata
    if nd is not None: data = np.where(data == nd, np.nan, data)
    def nan_fill(v):
        c = v[len(v)//2]
        if np.isnan(c):
            ok = v[~np.isnan(v)]
            return float(np.mean(ok)) if len(ok) > 0 else np.nan
        return c
    filled = generic_filter(data, nan_fill, size=3)
    print(f"  ℹ️  Remaining NaN: {int(np.isnan(filled).sum())}")
    meta.update({"dtype": "float32", "nodata": -9999.0, "count": 1})
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(np.where(np.isnan(filled), -9999.0, filled).astype(np.float32), 1)
    print(f"  ✅ Holes filled → {output_path}")
    return output_path

def compute_slope_aspect(dem_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with rasterio.open(dem_path) as src:
        elev = src.read(1).astype(np.float32)
        elev = np.where(elev == -9999.0, np.nan, elev)
        meta = src.meta.copy()
        rx, ry = abs(src.transform.a), abs(src.transform.e)
    dy, dx = np.gradient(elev, ry, rx)
    slope  = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    aspect = np.degrees(np.arctan2(-dy, dx)) % 360
    meta.update({"dtype": "float32", "count": 1, "nodata": -9999.0})
    sp = os.path.join(out_dir, "slope.tif")
    ap = os.path.join(out_dir, "aspect.tif")
    with rasterio.open(sp, "w", **meta) as d: d.write(np.where(np.isnan(slope),-9999.,slope).astype(np.float32),1)
    with rasterio.open(ap, "w", **meta) as d: d.write(np.where(np.isnan(aspect),-9999.,aspect).astype(np.float32),1)
    print(f"  ✅ Slope ({slope.min():.1f}°–{slope.max():.1f}°) → {sp}")
    print(f"  ✅ Aspect → {ap}")
    return sp, ap

def normalize(input_path, output_path, nodata_val=-9999.0):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(input_path) as src:
        data = src.read(1).astype(np.float32)
        meta = src.meta.copy()
    data = np.where(data == nodata_val, np.nan, data)
    valid = data[~np.isnan(data)]
    mn, mx = valid.min(), valid.max()
    norm = (data - mn) / (mx - mn + 1e-8)
    meta.update({"dtype": "float32", "nodata": -9999.0})
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(np.where(np.isnan(norm),-9999.,norm).astype(np.float32), 1)
    print(f"  ✅ Normalised [min={mn:.2f}, max={mx:.2f}] → {output_path}")
    return output_path

def create_forest_mask(canopy_path, output_path, threshold):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(canopy_path) as src:
        data = src.read(1).astype(np.float32)
        meta = src.meta.copy()
        nd = src.nodata
    if nd is not None: data = np.where(data == nd, 0.0, data)
    data = np.nan_to_num(data, nan=0.0)
    fmask = (data >= threshold).astype(np.uint8)
    pct = 100 * fmask.sum() / fmask.size
    print(f"  ℹ️  Forest pixels: {fmask.sum():,} / {fmask.size:,} ({pct:.1f}%)")
    meta.update({"dtype": "uint8", "nodata": 0, "count": 1})
    with rasterio.open(output_path, "w", **meta) as dst: dst.write(fmask, 1)
    print(f"  ✅ Forest mask → {output_path}")
    return output_path

def process_dem():
    step("DEM PIPELINE — SRTM (output_SRTMGL1.tif)")
    dem_dir  = os.path.join(OUTPUT_DIR, "dem")
    topo_dir = os.path.join(dem_dir, "topographic")
    if not os.path.exists(DEM_FILE):
        print(f"  ❌ File nahi mili: {DEM_FILE}"); return
    print("\n  Input file info:"); print_info(DEM_FILE)
    shapes = load_aoi(AOI_FILE)
    step("DEM 1/5: Clip to Huize AOI")
    clipped = clip_raster(DEM_FILE, shapes, os.path.join(dem_dir, "dem_clipped.tif"))
    step("DEM 2/5: Reproject UTM48N @ 16m")
    reproj  = reproject_raster(clipped, os.path.join(dem_dir, "dem_reprojected.tif"), TARGET_CRS, TARGET_RES)
    step("DEM 3/5: Fill nodata holes")
    filled  = fill_holes(reproj, os.path.join(dem_dir, "dem_filled.tif"))
    step("DEM 4/5: Slope + Aspect")
    compute_slope_aspect(filled, topo_dir)
    step("DEM 5/5: Normalize [0,1]")
    normalize(filled, os.path.join(dem_dir, "dem_normalized.tif"))
    print(f"\n  🎉 DEM COMPLETE! → {dem_dir}/")

def process_canopy():
    step("CANOPY PIPELINE — ETH (2 tiles)")
    canopy_dir = os.path.join(OUTPUT_DIR, "canopy_height")
    for f in [CANOPY_TILE1, CANOPY_TILE2]:
        if not os.path.exists(f):
            print(f"  ❌ File nahi mili: {f}"); return
    print("\n  Tile info:"); print_info(CANOPY_TILE1); print_info(CANOPY_TILE2)
    shapes = load_aoi(AOI_FILE)
    step("CANOPY 1/5: Merge 2 tiles")
    merged  = merge_tiles([CANOPY_TILE1, CANOPY_TILE2], os.path.join(canopy_dir, "canopy_merged.tif"))
    step("CANOPY 2/5: Clip to Huize AOI")
    clipped = clip_raster(merged, shapes, os.path.join(canopy_dir, "canopy_clipped.tif"))
    step("CANOPY 3/5: Reproject UTM48N @ 16m")
    reproj  = reproject_raster(clipped, os.path.join(canopy_dir, "canopy_reprojected.tif"), TARGET_CRS, TARGET_RES)
    step(f"CANOPY 4/5: Forest mask (>={FOREST_THR}m)")
    create_forest_mask(reproj, os.path.join(canopy_dir, "forest_mask.tif"), FOREST_THR)
    step("CANOPY 5/5: Normalize [0,1]")
    normalize(reproj, os.path.join(canopy_dir, "canopy_normalized.tif"), nodata_val=255.0)
    print(f"\n  🎉 CANOPY COMPLETE! → {canopy_dir}/")

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  IIDM PREPROCESSING | Author: RTripathi")
    print("  Huize County, Yunnan, China")
    print("="*55)
    if not os.path.exists(AOI_FILE):
        print(f"❌ AOI file nahi mili: {AOI_FILE}"); exit(1)
    process_dem()
    process_canopy()
    print("\n" + "="*55)
    print("  ✅ SAARI PREPROCESSING COMPLETE!")
    print(f"  📁 {OUTPUT_DIR}/")
    print("="*55)
