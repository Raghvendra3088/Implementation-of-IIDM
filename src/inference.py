import os
import sys
import torch
import numpy as np
import rasterio
from rasterio.windows import Window
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.iidm import IIDM_Full

def reconstruct_full_map():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🌍 Starting Full Map Reconstruction on: {device}")

    # Paths
    sentinel_path = "data/raw/sentinel2/sentinel2_stacked.tif" # Replace with actual name if different
    dem_path = "data/raw/alos_dem/dem_normalized.tif"
    canopy_path = "data/raw/eth_canopy/canopy_normalized.tif"
    checkpoint_path = "checkpoints/unet_epoch_100.pth" # Load the fully trained weights
    output_path = "results/final_predicted_carbon_map.tif"

    os.makedirs("results", exist_ok=True)

    # Initialize Base Model
    model = IIDM_Full().to(device)
    try:
        model.unet.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("✅ Trained weights loaded successfully.")
    except Exception as e:
        print("⚠️ Warning: Could not load trained weights. Running with untrained model for pipeline test.")
    model.eval()

    patch_size = 256

    try:
        with rasterio.open(sentinel_path) as src_opt, \
             rasterio.open(dem_path) as src_dem, \
             rasterio.open(canopy_path) as src_can:
            
            meta = src_opt.meta.copy()
            meta.update(count=1, dtype='float32') # Output is a 1-channel carbon map

            width, height = src_opt.width, src_opt.height
            print(f"Map Dimensions: {width}x{height} pixels")

            with rasterio.open(output_path, 'w', **meta) as dst:
                with torch.no_grad():
                    # Sliding window inference
                    for y in tqdm(range(0, height, patch_size), desc="Processing Rows"):
                        for x in range(0, width, patch_size):
                            # Handle edges
                            w = min(patch_size, width - x)
                            h = min(patch_size, height - y)
                            window = Window(x, y, w, h)

                            # Read patches
                            opt_patch = src_opt.read(window=window)
                            dem_patch = src_dem.read(window=window)
                            can_patch = src_can.read(window=window)

                            # Pad if at the edge to maintain 256x256 for the model
                            if w < patch_size or h < patch_size:
                                opt_patch = np.pad(opt_patch, ((0,0), (0, patch_size - h), (0, patch_size - w)), mode='reflect')
                                dem_patch = np.pad(dem_patch, ((0,0), (0, patch_size - h), (0, patch_size - w)), mode='reflect')
                                can_patch = np.pad(can_patch, ((0,0), (0, patch_size - h), (0, patch_size - w)), mode='reflect')

                            # Prepare tensors
                            opt_t = torch.from_numpy(opt_patch).float().unsqueeze(0).to(device)
                            str_t = torch.from_numpy(np.concatenate([dem_patch, can_patch], axis=0)).float().unsqueeze(0).to(device)

                            # Base IIDM Methodology: Inference
                            cond_features = model.encoder(opt_t)[-1]
                            pred_carbon = model.diffusion.sample(cond_features, str_t, image_size=256, batch_size=1)
                            
                            pred_np = pred_carbon.squeeze().cpu().numpy()

                            # Crop padding out before writing
                            pred_np = pred_np[:h, :w]
                            dst.write(pred_np, 1, window=window)

        print(f"🎉 Success! Final High-Res Carbon Map saved at: {output_path}")

    except Exception as e:
        print(f"❌ Error during map generation: {e}")

if __name__ == "__main__":
    reconstruct_full_map()
