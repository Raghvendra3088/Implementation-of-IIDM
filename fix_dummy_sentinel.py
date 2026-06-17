import rasterio
import numpy as np
from rasterio.transform import from_origin

# Yahan hum explicitly 6 channels assign kar rahe hain
data = np.random.rand(6, 256, 256).astype(np.float32) 
transform = from_origin(103.0, 26.0, 0.00014, 0.00014)

with rasterio.open(
    'data/raw/sentinel2/dummy_raster.tif', 'w', driver='GTiff',
    height=256, width=256, count=6, dtype=data.dtype, # Count = 6 channels
    crs='EPSG:4326', transform=transform,
) as dst:
    dst.write(data)
print("✅ Fixed Sentinel-2 dummy raster to 6 channels.")
