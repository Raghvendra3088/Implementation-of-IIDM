import os
import json
import numpy as np
import geopandas as gpd
from shapely.geometry import box

def create_spatial_splits(boundary_file, out_dir, grid_size=(3, 3), 
                          train_frac=0.70, val_frac=0.15, test_frac=0.15):
    os.makedirs(out_dir, exist_ok=True)
    
    # Load boundary
    gdf = gpd.read_file(boundary_file)
    minx, miny, maxx, maxy = gdf.total_bounds
    
    # Create grid tiles
    nx, ny = grid_size
    dx = (maxx - minx) / nx
    dy = (maxy - miny) / ny
    
    tiles = []
    for i in range(nx):
        for j in range(ny):
            b = box(minx + i*dx, miny + j*dy, minx + (i+1)*dx, miny + (j+1)*dy)
            tiles.append(b)
            
    # Convert tiles to a GeoDataFrame
    tiles_gdf = gpd.GeoDataFrame(geometry=tiles, crs=gdf.crs)
    
    # Intersect tiles with the actual study area boundary
    tiles_gdf = gpd.overlay(tiles_gdf, gdf, how='intersection')
    
    n_tiles = len(tiles_gdf)
    indices = np.random.permutation(n_tiles)
    
    n_train = int(n_tiles * train_frac)
    n_val = int(n_tiles * val_frac)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]
    
    def save_regions(indices, name):
        regions = tiles_gdf.iloc[indices]
        # Save as geojson for easy loading later
        out_path = os.path.join(out_dir, f"{name}_regions.geojson")
        regions.to_file(out_path, driver="GeoJSON")
        print(f"Saved {len(regions)} tiles to {out_path}")
        
    save_regions(train_idx, "train")
    save_regions(val_idx, "val")
    save_regions(test_idx, "test")
    
    print("Spatial split complete. Zero overlap guaranteed by grid approach.")

if __name__ == "__main__":
    np.random.seed(42)
    boundary_file = "/Users/raghvendra/Implementation-of-IIDM/huize_boundary/huize_boundary.geojson"
    out_dir = "/Users/raghvendra/iidm_project/Physics-Informed-IIDM/data/splits"
    
    # We use 4x4 grid to get enough tiles
    create_spatial_splits(boundary_file, out_dir, grid_size=(4, 4))
