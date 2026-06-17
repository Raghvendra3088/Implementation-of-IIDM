import os
import sys
import torch
import numpy as np
import rasterio
from rasterio.windows import Window
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.iidm import IIDM

def get_valid_path(preferred, fallback):
    return preferred if os.path.exists(preferred) else fallback

def reconstruct_full_map():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🌍 Starting Full Map Reconstruction on: {device}")

    sentinel_path = get_valid_path("data/raw/sentinel2/sentinel2_stacked.tif", "data/raw/sentinel2/dummy_raster.tif")
    dem_path = get_valid_path("data/raw/alos_dem/dem_normalized.tif", "data/raw/alos_dem/dummy_raster.tif")
    canopy_path = get_valid_path("data/raw/eth_canopy/canopy_normalized.tif", "data/raw/eth_canopy/dummy_raster.tif")
    
    checkpoint_path = "checkpoints/actual_iidm_epoch_100.pth"
    output_path = "results/final_predicted_carbon_map.tif"

    os.makedirs("results", exist_ok=True)

    model = IIDM(in_channels=6, T=1000).to(device)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("✅ Trained weights loaded successfully.")
    else:
        print("⚠️ Warning: Trained weights missing. Running reconstruction with initialized weights.")
    model.eval()

    patch_size = 256

    try:
        with rasterio.open(sentinel_path) as src_opt, \
             rasterio.open(dem_path) as src_dem, \
             rasterio.open(canopy_path) as src_can:
            
            meta = src_opt.meta.copy()
            meta.update(count=1, dtype='float32')

            width, height = src_opt.width, src_opt.height
            print(f"Map Dimensions: {width}x{height} pixels")

            with rasterio.open(output_path, 'w', **meta) as dst:
                with torch.no_grad():
                    for y in tqdm(range(0, height, patch_size), desc="Processing Rows"):
                        for x in range(0, width, patch_size):
                            w = min(patch_size, width - x)
                            h = min(patch_size, height - y)
                            window = Window(x, y, w, h)

                            opt_patch = src_opt.read(window=window)
                            dem_patch = src_dem.read(window=window)
                            can_patch = src_can.read(window=window)

                            if opt_patch.shape[0] != 6:
                                opt_patch = np.repeat(opt_patch, 6, axis=0)[:6, :, :]

                            if w < patch_size or h < patch_size:
                                opt_patch = np.pad(opt_patch, ((0,0), (0, patch_size - h), (0, patch_size - w)), mode='reflect')
                                dem_patch = np.pad(dem_patch, ((0,0), (0, patch_size - h), (0, patch_size - w)), mode='reflect')
                                can_patch = np.pad(can_patch, ((0,0), (0, patch_size - h), (0, patch_size - w)), mode='reflect')

                            opt_t = torch.from_numpy(opt_patch).float().unsqueeze(0).to(device)

                            s_feats, _ = model.student(opt_t)
                            latents = s_feats[0]
                            pred_carbon = model.inr(latents, patch_size, patch_size)
                            
                            pred_np = pred_carbon.squeeze().cpu().numpy()
                            pred_np = pred_np[:h, :w]
                            dst.write(pred_np, 1, window=window)

        print(f"🎉 Success! Final High-Res Map saved at: {output_path}")

    except Exception as e:
        print(f"❌ Error during map generation: {e}")

if __name__ == "__main__":
    reconstruct_full_map()
