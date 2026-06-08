# Implementation of IIDM
**Improved Implicit Diffusion Model for Forest Carbon Stock Estimation**

## Study Area
Huize County, Yunnan Province, China

## Project Structure

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
