import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
import geopandas as gpd

class IIDMPreprocessor:
    def __init__(self, base_dir='data'):
        self.base_dir = base_dir
        self.raw_dir = os.path.join(base_dir, 'raw')
        self.processed_dir = os.path.join(base_dir, 'processed_patches')
        os.makedirs(self.processed_dir, exist_ok=True)

    def align_raster(self, reference_path, source_path, output_path, is_categorical=False):
        """Aligns DEM and Canopy to Sentinel-2's 16m resolution and extent."""
        print(f"Aligning {os.path.basename(source_path)} to 16m grid...")
        with rasterio.open(reference_path) as ref:
            ref_meta = ref.meta.copy()
            
        with rasterio.open(source_path) as src:
            ref_meta.update({'nodata': 0})
            with rasterio.open(output_path, 'w', **ref_meta) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=ref_meta['transform'],
                        dst_crs=ref_meta['crs'],
                        resampling=Resampling.nearest if is_categorical else Resampling.bilinear
                    )

    def process_gedi_to_target(self, reference_path, gedi_vector_path, output_path):
        """Converts GEDI L4A points into a continuous 16m Carbon Stock target raster."""
        print("Rasterizing GEDI L4A Carbon Biomass...")
        with rasterio.open(reference_path) as ref:
            gdf = gpd.read_file(gedi_vector_path).to_crs(ref.crs)
            shapes = ((geom, val) for geom, val in zip(gdf.geometry, gdf['agbd']))
            
            carbon_raster = rasterize(
                shapes=shapes,
                out_shape=(ref.height, ref.width),
                transform=ref.transform,
                fill=0,
                dtype=rasterio.float32
            )
            
            meta = ref.meta.copy()
            meta.update(dtype=rasterio.float32, nodata=0, count=1)
            with rasterio.open(output_path, 'w', **meta) as dst:
                dst.write(carbon_raster, 1)

    def extract_pytorch_patches(self, patch_size=256, forest_threshold=0.20):
        """Stacks the 4 datasets, applies canopy mask, and extracts 256x256 valid patches."""
        print("Extracting valid 256x256 PyTorch patches...")
        
        # Load the aligned datasets
        with rasterio.open(f'{self.raw_dir}/sentinel2_16m.tif') as src: s2 = src.read()
        with rasterio.open(f'{self.raw_dir}/dem_aligned.tif') as src: dem = src.read()
        with rasterio.open(f'{self.raw_dir}/canopy_aligned.tif') as src: canopy = src.read()
        with rasterio.open(f'{self.raw_dir}/carbon_target.tif') as src: carbon = src.read(1)
        
        # Create Binary Forest Mask (Canopy > 2m)
        mask = np.where(canopy[0] > 2.0, 1, 0).astype(np.uint8)
        
        # Stack inputs (Sentinel + DEM + Canopy)
        features = np.concatenate([s2, dem, canopy], axis=0)
        _, h, w = features.shape
        
        patch_count = 0
        for y in range(0, h - patch_size, patch_size):
            for x in range(0, w - patch_size, patch_size):
                patch_mask = mask[y:y+patch_size, x:x+patch_size]
                
                # Check if patch has enough forest coverage
                if np.mean(patch_mask) >= forest_threshold:
                    np.save(f"{self.processed_dir}/patch_{patch_count}.npy", {
                        'features': features[:, y:y+patch_size, x:x+patch_size],
                        'carbon': carbon[y:y+patch_size, x:x+patch_size],
                        'mask': patch_mask
                    })
                    patch_count += 1
        print(f"Extraction complete! {patch_count} patches saved to {self.processed_dir}/")

if __name__ == "__main__":
    processor = IIDMPreprocessor(base_dir='data')
    
    # Define file paths (Assumes data is placed in data/raw/)
    s2_path = 'data/raw/sentinel2_16m.tif'
    dem_path = 'data/raw/srtm_dem.tif'
    canopy_path = 'data/raw/eth_canopy.tif'
    gedi_path = 'data/raw/gedi_l4a.geojson'
    
    # Ensure Sentinel-2 base exists before proceeding
    if os.path.exists(s2_path):
        # 1 & 2. Align DEM and Canopy to Sentinel-2
        processor.align_raster(s2_path, dem_path, 'data/raw/dem_aligned.tif')
        processor.align_raster(s2_path, canopy_path, 'data/raw/canopy_aligned.tif')
        
        # 3. Process GEDI points into Carbon Raster
        processor.process_gedi_to_target(s2_path, gedi_path, 'data/raw/carbon_target.tif')
        
        # 4. Generate AI Model Patches
        processor.extract_pytorch_patches()
    else:
        print(f"Please ensure {s2_path} exists in the repository before running.")