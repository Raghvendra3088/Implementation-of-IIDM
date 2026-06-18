# Implementation of IIDM
**Improved Implicit Diffusion Model with Knowledge Distillation to Estimate the Spatial Distribution Density of Carbon Stock in Remote Sensing Imagery**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**(IIDM)** for high-resolution forest carbon stock estimation. The architecture leverages latent space diffusion, continuous coordinate mapping via Implicit Neural Representations (INR), and Knowledge Distillation to achieve state-of-the-art spatial fidelity in remote sensing applications.

## Study Area
Huize County, Yunnan Province, China

##  Core Architecture

Unlike standard pixel-space diffusion models, this implementation strictly adheres to the mathematical formulations of the IIDM paper, comprising three primary components:

1. **Knowledge Distillation (KD-VGG):**
   * **Teacher:** A frozen, pre-trained VGG-19 network that extracts robust multi-scale optical features.
   * **Student:** A highly compressed, lightweight VGG variant distilled via feature-level Mean Squared Error (MSE) across 4 spatial scales, reducing parameter count by ~98%.
2. **Latent Space Diffusion:**
   * Denoising is performed in the compressed latent feature space rather than the high-dimensional pixel space.
   * Utilizes a **Cross-Attention KD-UNet**, conditioned on teacher features, combined with a linear noise schedule ($T=1000$) and DDIM for accelerated sampling.
3. **Implicit Neural Representation (INR) Decoder:**
   * Treats the carbon map as a continuous function $F(x, y, features) \rightarrow carbon\_value$.
   * Utilizes a high-frequency Positional Encoding ($L=10$, yielding a 40-dimensional coordinate tensor) coupled with a Sine-activated MLP (SIREN) to decode latent features into high-resolution carbon maps.

### Optimization Objective
The model is trained end-to-end optimizing a joint loss function:
$L_{total} = L_{diff} + 0.1 L_{kd} + 1.0 L_{recon}$

---

## 📂 Repository Structure

```text
Implementation-of-IIDM/
├── data/
│   ├── README.md               # Data structure documentation
│   ├── raw/                    # Raw Sentinel-2, ALOS DEM, ETH Canopy .tif files
│   ├── processed/              # Interim normalized files and inventory statistics
│   └── patches/                # 256x256 tensor patches for model ingestion
├── notebooks/
│   └── True_IIDM_FullRun.ipynb # Google Colab deployment and execution notebook
├── src/
│   ├── models/
│   │   ├── kd_vgg.py           # VGG19 Teacher and Lightweight Student VGG
│   │   ├── kd_unet.py          # Cross-Attention UNet Denoiser
│   │   ├── inr.py              # Positional Encoding & SIREN MLP
│   │   ├── diffusion.py        # Latent Space Forward/Reverse Process
│   │   └── iidm.py             # Unified Architecture Wrapper
│   ├── utils/
│   │   ├── metrics.py          # RMSE, MAE, SSIM, and R² calculators
│   │   └── visualization.py    # Matplotlib error heatmaps
│   ├── train.py                # Main training loop with WandB integration
│   ├── evaluate.py             # DDIM Inference and metric evaluation
│   └── inference.py            # Sliding-window full map reconstruction
├── preprocessing/
│   ├── dataset.py              # PyTorch Dataset and Dataloader
│   └── run_preprocessing.py    # GDAL/Rasterio pipeline for tile generation
└── requirements.txt            # Project dependencies

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install rasterio geopandas numpy scipy shapely
```
#### Install all required dependencies -
pip install -r requirements.txt

#### Authenticate Weights & Biases (for training visualization) -
wandb login

### Run Preprocessing
```bash
python3 preprocessing/preprocess_all.py
```

## Datasets
| Dataset | Source | Resolution |
|---------|--------|------------|
| SRTM DEM | NASA EarthData | 30m → 16m |
| ETH Canopy Height | langnico.github.io | 10m → 16m |
| Sentinel-2 | Copernicus | 10m → 16m |
| GEDI L4A | NASA AppEEARS | point data |

## Results (Preprocessing)
- DEM elevation range: 905m – 3821m ✅
- Forest coverage: 44.1% of AOI ✅
- Output resolution: 16m (matches GF-1 WFV paper)
