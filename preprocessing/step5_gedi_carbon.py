import rasterio
from rasterio.features import rasterize
import geopandas as gpd
import numpy as np

def rasterize_gedi_points(reference_grid_path, gedi_vector_path, output_raster_path):
    print("Step 5: Converting GEDI vector biomass points to target matrix...")
    with rasterio.open(reference_grid_path) as ref:
        shape = (ref.height, ref.width)
        transform = ref.transform
        meta = ref.meta.copy()
        
    gdf = gpd.read_file(gedi_vector_path).to_crs(meta['crs'])
    shapes = ((geom, val) for geom, val in zip(gdf.geometry, gdf['agbd']))
    
    carbon_matrix = rasterize(
        shapes=shapes,
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=rasterio.float32
    )
    
    meta.update(dtype=rasterio.float32, count=1, nodata=0)
    with rasterio.open(output_raster_path, 'w', **meta) as dst:
        dst.write(carbon_matrix, 1)
    print(f"✅ Target matrix written to {output_raster_path}")

if __name__ == "__main__":
    pass
