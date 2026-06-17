import os
import numpy as np
import rasterio
from rasterio.transform import from_origin

folders = [
    'data/raw/gf1', 
    'data/raw/alos_dem', 
    'data/raw/eth_canopy', 
    'data/raw/sentinel2'
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, 'dummy_raster.tif')
    
    data = np.random.rand(1, 256, 256).astype(np.float32)
    transform = from_origin(103.0, 26.0, 0.00014, 0.00014)
    
    with rasterio.open(
        file_path, 'w', driver='GTiff',
        height=256, width=256, count=1, dtype=data.dtype,
        crs='EPSG:4326', transform=transform,
    ) as dst:
        dst.write(data)
    print(f"Fixed and recreated valid dummy raster: {file_path}")
