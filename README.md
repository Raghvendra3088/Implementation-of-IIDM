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


## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install rasterio geopandas numpy scipy shapely
```

## Run Preprocessing
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
