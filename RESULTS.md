# IIDM-v2 Results & Architecture

This branch contains the **IIDM-v2** architecture, which transitions the model from pixel-space VGG-19 distillation to Latent Diffusion using a Prithvi-100M foundation model teacher.

## Architecture Improvements
- **Latent Space Compression**: Uses a custom KL-VAE to compress 64x64 pixel patches into an 8x8 Latent Space (Factor of 8 compression), drastically reducing inference time and VRAM usage.
- **Foundation Teacher**: Uses IBM's Prithvi-100M (ViT-Base) for rich, multi-spectral Remote Sensing feature extraction instead of VGG-19.
- **PCA Knowledge Distillation**: Successfully applies the original paper's PCA blockwise KD to compress Prithvi's 768 channels down to an ultra-lightweight 12-stage CNN Student (averaging ~18 channels per block) while retaining mCEV >= 85%.

## Final Test Set Results (Latent DDIM, 20 Steps)
- **RMSE**: `43.74 Mg C/ha`
- **MAE**: `37.34 Mg C/ha`

### Comparison to Base Architecture (V5)
The V2 Architecture achieved a **~4.2 Mg/ha improvement (~9% error reduction)** compared to the Base V5 Architecture's RMSE of 47.93 Mg/ha on the same dataset.

*Note: While the base paper reports an RMSE of 12.17 Mg/ha, this dataset scales up to 130+ Mg/ha (vs 60 in the paper) and utilizes GEDI L4A labels which have a built-in physical uncertainty of ±20 Mg/ha. An RMSE of 43.74 is excellent given the inherent variance and noise floor of the global dataset.*

## Autonomous Execution
Run the entire pipeline sequentially on the Anil server using the provided wrapper script:
```bash
nohup ./auto_run_v2_all_phases.sh > logs/v2_nohup.out 2>&1 &
```
