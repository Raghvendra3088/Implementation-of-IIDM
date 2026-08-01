"""
Complete patch builder for IIDM base paper reproduction
Sentinel-2 (B02/B03/B04/B08) @ 16m + ALOS DEM + ETH Canopy + GEDI L4A
Output: 64x64 patches → ~/iidm_project/patches_final/
"""
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import from_bounds
import h5py
from pathlib import Path
import glob, json
from scipy.interpolate import griddata

BASE    = Path("/Users/raghvendra/iidm_project")
OUT_DIR = BASE / "patches_final"
OUT_DIR.mkdir(exist_ok=True)

TARGET_CRS = "EPSG:32648"
TARGET_RES = 16          # GF-1 WFV proxy
PATCH_SIZE = 64
STRIDE     = 32
HUIZE_BBOX = (103.0, 25.8, 103.8, 26.6)  # west,south,east,north WGS84

print("=" * 60)
print("STEP 1: Merging Sentinel-2 bands")
print("=" * 60)

def get_s2_band_tiles(band):
    pattern = str(BASE / f"sentinel2/raw/{band}/tile_*/*/response.tiff")
    tiles = glob.glob(pattern)
    if not tiles:
        # try .SAFE folder
        pattern2 = str(BASE / f"sentinel2/**/*_{band}_*.jp2")
        tiles = glob.glob(pattern2, recursive=True)
        if not tiles:
            pattern3 = str(BASE / f"sentinel2/**/*_{band}*.tif*")
            tiles = glob.glob(pattern3, recursive=True)
    return tiles

def merge_and_reproject(tile_paths, dst_path, band_idx=1):
    """Merge tiles and reproject to TARGET_CRS @ TARGET_RES"""
    srcs = [rasterio.open(p) for p in tile_paths]
    mosaic, transform = merge(srcs)
    for s in srcs: s.close()

    # write temp mosaic
    tmp = dst_path.parent / f"_tmp_{dst_path.name}"
    meta = srcs[0].meta.copy()
    meta.update(width=mosaic.shape[2], height=mosaic.shape[1],
                transform=transform, count=1, dtype='float32')
    with rasterio.open(tmp, 'w', **meta) as f:
        f.write(mosaic[band_idx-1:band_idx].astype(np.float32))

    # reproject
    with rasterio.open(tmp) as src:
        t, w, h = calculate_default_transform(
            src.crs, TARGET_CRS, src.width, src.height,
            *src.bounds, resolution=TARGET_RES)
        meta2 = src.meta.copy()
        meta2.update(crs=TARGET_CRS, transform=t, width=w, height=h,
                     dtype='float32', nodata=0)
        data = np.zeros((1, h, w), dtype=np.float32)
        reproject(rasterio.band(src, 1), data[0],
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=t, dst_crs=TARGET_CRS,
                  resampling=Resampling.bilinear)
    tmp.unlink()
    with rasterio.open(dst_path, 'w', **meta2) as f:
        f.write(data)
    print(f"  Saved: {dst_path.name}  shape={data.shape}  range=[{data.min():.1f},{data.max():.1f}]")
    return dst_path, t, w, h

# Process S2 bands
tmp_dir = BASE / "tmp_processed"
tmp_dir.mkdir(exist_ok=True)

s2_bands = {}
ref_transform = ref_w = ref_h = None

for band in ['B02','B03','B04','B08']:
    tiles = get_s2_band_tiles(band)
    print(f"  {band}: {len(tiles)} tiles found")
    if not tiles:
        raise FileNotFoundError(f"No tiles for {band}!")
    dst = tmp_dir / f"s2_{band}.tif"
    p, t, w, h = merge_and_reproject(tiles, dst)
    s2_bands[band] = dst
    if ref_transform is None:
        ref_transform, ref_w, ref_h = t, w, h

print("\n" + "=" * 60)
print("STEP 2: Reprojecting ALOS DEM")
print("=" * 60)

def reproject_to_ref(src_path, dst_path, ref_transform, ref_w, ref_h, band=1):
    with rasterio.open(src_path) as src:
        meta = src.meta.copy()
        meta.update(crs=TARGET_CRS, transform=ref_transform,
                    width=ref_w, height=ref_h, count=1, dtype='float32', nodata=-9999)
        data = np.zeros((1, ref_h, ref_w), dtype=np.float32)
        reproject(rasterio.band(src, band), data[0],
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=ref_transform, dst_crs=TARGET_CRS,
                  resampling=Resampling.bilinear)
    with rasterio.open(dst_path, 'w', **meta) as f:
        f.write(data)
    print(f"  Saved: {dst_path.name}  range=[{data.min():.1f},{data.max():.1f}]")
    return dst_path

alos_dir = BASE / "alos_dem/AP_25667_FBD_F0520_RT1"
dem_path    = alos_dir / "AP_25667_FBD_F0520_RT1.dem.tif"
hh_path     = alos_dir / "AP_25667_FBD_F0520_RT1_HH.tif"
hv_path     = alos_dir / "AP_25667_FBD_F0520_RT1_HV.tif"

dem_repr = reproject_to_ref(dem_path,    tmp_dir/"dem.tif",    ref_transform, ref_w, ref_h)
hh_repr  = reproject_to_ref(hh_path,     tmp_dir/"hh.tif",     ref_transform, ref_w, ref_h)
hv_repr  = reproject_to_ref(hv_path,     tmp_dir/"hv.tif",     ref_transform, ref_w, ref_h)

print("\n" + "=" * 60)
print("STEP 3: Reprojecting ETH Canopy Height")
print("=" * 60)

canopy_tiles = glob.glob(str(BASE / "eth_canopy/*.tif"))
print(f"  {len(canopy_tiles)} canopy tiles")
canopy_srcs = [rasterio.open(p) for p in canopy_tiles]
canopy_mosaic, canopy_transform = merge(canopy_srcs)
for s in canopy_srcs: s.close()

tmp_canopy = tmp_dir / "_tmp_canopy.tif"
meta_c = canopy_srcs[0].meta.copy()
meta_c.update(width=canopy_mosaic.shape[2], height=canopy_mosaic.shape[1],
              transform=canopy_transform, count=1, dtype='float32')
with rasterio.open(tmp_canopy, 'w', **meta_c) as f:
    f.write(canopy_mosaic[:1].astype(np.float32))

canopy_repr = reproject_to_ref(tmp_canopy, tmp_dir/"canopy.tif", ref_transform, ref_w, ref_h)
tmp_canopy.unlink()
print(f"  Canopy reprojected")

print("\n" + "=" * 60)
print("STEP 4: Rasterizing GEDI L4A AGB → Carbon labels")
print("=" * 60)

h5_files = glob.glob(str(BASE / "GEDI_L4A_AGB_Density_V2/*.h5"))
print(f"  {len(h5_files)} GEDI h5 files")

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

print(f"  GEDI points in Huize: {len(all_lons)}")
if len(all_lons) < 10:
    raise ValueError("Too few GEDI points! Check HUIZE_BBOX or h5 files.")

# Convert AGB → Carbon (IPCC factor 0.47)
all_carbon = np.array(all_agb) * 0.47

# Rasterize onto reference grid using griddata interpolation
from pyproj import Transformer
transformer = Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True)
xs, ys = transformer.transform(np.array(all_lons), np.array(all_lats))

# Build pixel coordinate grid
cols = np.arange(ref_w)
rows = np.arange(ref_h)
grid_x = ref_transform.c + cols * ref_transform.a
grid_y = ref_transform.f + rows * ref_transform.e
grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)

print("  Interpolating GEDI points onto grid...")
carbon_grid = griddata(
    np.column_stack([xs, ys]), all_carbon,
    (grid_xx, grid_yy), method='linear', fill_value=np.nan
)
# Fill remaining NaN with nearest
carbon_nn = griddata(
    np.column_stack([xs, ys]), all_carbon,
    (grid_xx, grid_yy), method='nearest'
)
carbon_grid = np.where(np.isnan(carbon_grid), carbon_nn, carbon_grid)
print(f"  Carbon grid: min={carbon_grid.min():.1f} max={carbon_grid.max():.1f} mean={carbon_grid.mean():.1f} Mg C/ha")

# Save carbon raster
carbon_meta = {
    'driver': 'GTiff', 'dtype': 'float32', 'count': 1,
    'crs': TARGET_CRS, 'transform': ref_transform,
    'width': ref_w, 'height': ref_h, 'nodata': -9999
}
carbon_path = tmp_dir / "carbon.tif"
with rasterio.open(carbon_path, 'w', **carbon_meta) as f:
    f.write(carbon_grid.astype(np.float32)[np.newaxis])
print(f"  Carbon raster saved")

print("\n" + "=" * 60)
print("STEP 5: Loading all layers")
print("=" * 60)

def load_band(path):
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)

b02 = load_band(s2_bands['B02'])
b03 = load_band(s2_bands['B03'])
b04 = load_band(s2_bands['B04'])
b08 = load_band(s2_bands['B08'])
dem = load_band(dem_repr)
hh  = load_band(hh_repr)
hv  = load_band(hv_repr)
can = load_band(canopy_repr)
car = carbon_grid.astype(np.float32)

H, W = b02.shape
print(f"  Grid size: {H} x {W}")

# Stack: 8 input channels (4 S2 + DEM + HH + HV + Canopy)
# Paper uses 4 channels only (S2) — we keep auxiliary as extra for base paper
# BASE PAPER: only 4 S2 bands as input
inputs = np.stack([b02, b03, b04, b08], axis=0)  # (4, H, W) — paper exact
print(f"  Input stack: {inputs.shape}")
print(f"  Carbon label: {car.shape}")

print("\n" + "=" * 60)
print("STEP 6: Normalizing")
print("=" * 60)

# Per-channel normalization
stats = {}
norm_inputs = np.zeros_like(inputs)
for i, name in enumerate(['B02','B03','B04','B08']):
    ch = inputs[i]
    valid = ch[ch > 0]
    if len(valid) == 0:
        raise ValueError(f"Band {name} is all zeros!")
    mean, std = float(valid.mean()), float(valid.std())
    norm_inputs[i] = (ch - mean) / (std + 1e-6)
    stats[name] = {'mean': mean, 'std': std}
    print(f"  {name}: mean={mean:.2f} std={std:.2f}")

# Carbon normalization
car_valid = car[(car > 0) & np.isfinite(car)]
car_mean, car_std = float(car_valid.mean()), float(car_valid.std())
stats['carbon'] = {'mean': car_mean, 'std': car_std}
norm_carbon = (car - car_mean) / (car_std + 1e-6)
norm_carbon = np.clip(norm_carbon, -3, 3)
print(f"  Carbon: mean={car_mean:.2f} std={car_std:.2f} Mg C/ha")

with open(OUT_DIR / "norm_stats.json", 'w') as f:
    json.dump(stats, f, indent=2)

print("\n" + "=" * 60)
print("STEP 7: Extracting 64x64 patches")
print("=" * 60)

count = 0
skipped = 0
P = PATCH_SIZE

for r in range(0, H - P + 1, STRIDE):
    for c in range(0, W - P + 1, STRIDE):
        # Input patch
        x_patch = norm_inputs[:, r:r+P, c:c+P]   # (4, 64, 64)
        y_patch  = norm_carbon[r:r+P, c:c+P]       # (64, 64)

        # Skip if carbon mostly invalid
        raw_carbon = car[r:r+P, c:c+P]
        if (raw_carbon <= 0).mean() > 0.5:
            skipped += 1
            continue

        # Skip if input mostly zero (cloud/nodata)
        if (x_patch == 0).mean() > 0.3:
            skipped += 1
            continue

        np.savez_compressed(
            OUT_DIR / f"patch_{count:05d}.npz",
            image=x_patch.astype(np.float32),    # (4, 64, 64)
            carbon=y_patch.astype(np.float32),    # (64, 64)
            carbon_raw=raw_carbon.astype(np.float32)
        )
        count += 1

print(f"\n  Patches saved: {count}")
print(f"  Patches skipped: {skipped}")
print(f"  Output: {OUT_DIR}")

if count < 100:
    print("\n  WARNING: Very few patches! Check data coverage.")
else:
    print(f"\n  SUCCESS! {count} patches ready for training.")
    print(f"  Each patch: image=(4,64,64), carbon=(64,64)")
    print(f"  Paper config: 4 bands (B02/B03/B04/B08) @ 16m, 64x64 patches")
