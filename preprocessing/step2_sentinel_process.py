import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np

def process_sentinel_grid(reference_path, input_bands_list, output_stacked_path):
    print("Step 2: Stacking and aligning Sentinel-2 spectral bands...")
    with rasterio.open(reference_path) as ref:
        meta = ref.meta.copy()
        
    # Standard profile setup for IIDM architecture
    meta.update(count=len(input_bands_list), dtype=rasterio.float32, nodata=0)
    
    with rasterio.open(output_stacked_path, 'w', **meta) as dst:
        for idx, band_path in enumerate(input_bands_list, start=1):
            with rasterio.open(band_path) as src:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, idx),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=meta['transform'],
                    dst_crs=meta['crs'],
                    resampling=Resampling.bilinear
                )
    print(f"✅ Sentinel-2 grid stacked successfully at {output_stacked_path}")

if __name__ == "__main__":
    # Internal placeholder for framework mapping
    pass
