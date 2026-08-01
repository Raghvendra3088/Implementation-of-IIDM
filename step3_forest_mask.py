"""
IIDM Preprocessing - Step 3: Forest / Non-Forest Mask
======================================================
Paper reference: Appendix A, Section 3
- Paper uses F-Pix2Pix model for forest/non-forest classification
- This implementation uses NDVI thresholding (standard remote sensing approach)
  as a practical alternative when F-Pix2Pix weights are unavailable
- Outputs a binary mask: 1 = forest, 0 = non-forest / nodata = -9999

Input  : data/processed/aligned_gf1.tif   (from Step 2)
Output : data/masks/forest_mask.tif
         data/processed/masked_carbon_density.tif

Band order expected (GF-1 WFV / Sentinel-2):
    Band 1 = Blue
    Band 2 = Green
    Band 3 = Red
    Band 4 = NIR
"""

import numpy as np
import rasterio
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
PROCESSED  = ROOT / "data" / "processed"
MASKS_DIR  = ROOT / "data" / "masks"
MASKS_DIR.mkdir(parents=True, exist_ok=True)

# ── Thresholds ─────────────────────────────────────────────────────────────────
NDVI_THRESHOLD = 0.3     # pixels with NDVI > 0.3 are classified as forest
                         # (standard threshold; adjust based on your study area)
MIN_CANOPY_HT  = 2.0    # optional: min canopy height (m) to confirm forest


def compute_ndvi(red: np.ndarray, nir: np.ndarray,
                 nodata: float = -9999.0) -> np.ndarray:
    """
    NDVI = (NIR - Red) / (NIR + Red)
    Range: -1 to 1.  Forest typically > 0.3
    """
    ndvi             = np.full_like(red, nodata, dtype=np.float32)
    valid            = (red != nodata) & (nir != nodata) & ((nir + red) != 0)
    ndvi[valid]      = (nir[valid] - red[valid]) / (nir[valid] + red[valid])
    return ndvi


def create_forest_mask(ndvi: np.ndarray,
                       canopy: np.ndarray = None,
                       ndvi_thresh: float = NDVI_THRESHOLD,
                       canopy_thresh: float = MIN_CANOPY_HT,
                       nodata: float = -9999.0) -> np.ndarray:
    """
    Binary forest mask.
    forest = 1  if NDVI > threshold (AND canopy height > threshold if provided)
    non-forest = 0
    nodata = -9999
    """
    mask = np.zeros_like(ndvi, dtype=np.int16)

    # NDVI-based classification
    valid_ndvi = ndvi != nodata
    forest     = valid_ndvi & (ndvi > ndvi_thresh)

    # Optionally refine with canopy height
    if canopy is not None:
        valid_canopy = canopy != nodata
        forest       = forest & valid_canopy & (canopy > canopy_thresh)

    mask[forest]             = 1
    mask[~valid_ndvi]        = -9999
    return mask


def apply_mask_to_carbon(carbon_path: Path,
                          mask: np.ndarray,
                          out_path: Path,
                          meta: dict) -> None:
    """Apply forest mask to carbon density raster — zero out non-forest pixels."""
    with rasterio.open(carbon_path) as src:
        carbon = src.read(1).astype(np.float32)
        nodata = src.nodata if src.nodata is not None else -9999.0

    masked_carbon               = carbon.copy()
    # Non-forest pixels → nodata
    masked_carbon[mask == 0]    = nodata
    masked_carbon[mask == -9999] = nodata

    out_meta = meta.copy()
    out_meta.update({"count": 1, "dtype": "float32",
                     "nodata": nodata, "compress": "lzw"})
    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(masked_carbon[np.newaxis, :, :])

    valid = masked_carbon[masked_carbon != nodata]
    print(f"\n[RESULT] Masked Carbon Density (Mg C/ha):")
    print(f"  Forest pixels : {(mask == 1).sum():,}")
    print(f"  Non-forest    : {(mask == 0).sum():,}")
    print(f"  Mean (forest) : {valid.mean():.4f}")
    print(f"  Max  (forest) : {valid.max():.4f}")


def main():
    print("=" * 55)
    print("  STEP 3 — Forest / Non-Forest Mask")
    print("=" * 55)

    # ── 1. Load aligned GF-1 / Sentinel-2 ────────────────────────────────────
    print("\n[1/4] Loading aligned GF-1 raster ...")
    gf1_path = PROCESSED / "aligned_gf1.tif"
    if not gf1_path.exists():
        raise FileNotFoundError(
            f"{gf1_path} not found. Run step2_carbon_density.py first."
        )

    with rasterio.open(gf1_path) as src:
        bands  = src.count
        nodata = src.nodata if src.nodata is not None else -9999.0
        meta   = src.meta.copy()
        data   = src.read().astype(np.float32)

    print(f"  Shape : {data.shape}  (bands × rows × cols)")
    print(f"  Bands : {bands}")

    # ── 2. Extract Red and NIR bands ──────────────────────────────────────────
    # GF-1 WFV / Sentinel-2 standard band order: B, G, R, NIR
    if bands >= 4:
        red = data[2]    # Band 3 (0-indexed: 2)
        nir = data[3]    # Band 4 (0-indexed: 3)
        print("  Using Band 3 (Red) and Band 4 (NIR)")
    elif bands == 1:
        raise ValueError(
            "Single-band image detected. Need at least 4-band (B, G, R, NIR) image."
        )
    else:
        raise ValueError(
            f"Expected ≥4 bands, got {bands}. "
            "Check your GF-1 / Sentinel-2 file."
        )

    # Replace any 0-value pixels with nodata (common in satellite imagery)
    red[red == 0] = nodata
    nir[nir == 0] = nodata

    # ── 3. Compute NDVI ───────────────────────────────────────────────────────
    print("\n[2/4] Computing NDVI ...")
    ndvi  = compute_ndvi(red, nir, nodata)
    valid = ndvi[ndvi != nodata]
    print(f"  NDVI range : [{valid.min():.3f}, {valid.max():.3f}]")
    print(f"  Mean NDVI  : {valid.mean():.3f}")

    # ── 4. Load canopy height (optional, for refined mask) ────────────────────
    canopy_path = PROCESSED / "aligned_canopy.tif"
    canopy_arr  = None
    if canopy_path.exists():
        print("\n[3/4] Loading canopy height for mask refinement ...")
        with rasterio.open(canopy_path) as src:
            canopy_arr = src.read(1).astype(np.float32)
    else:
        print("\n[3/4] Canopy raster not found — using NDVI only for mask.")

    # ── 5. Create forest mask ─────────────────────────────────────────────────
    print(f"\n[4/4] Creating forest mask (NDVI threshold = {NDVI_THRESHOLD}) ...")
    mask = create_forest_mask(ndvi, canopy_arr, NDVI_THRESHOLD,
                               MIN_CANOPY_HT, nodata)

    forest_pct = 100 * (mask == 1).sum() / (mask != -9999).sum()
    print(f"  Forest coverage : {forest_pct:.1f}%")

    # ── 6. Save mask ──────────────────────────────────────────────────────────
    mask_meta = meta.copy()
    mask_meta.update({"count": 1, "dtype": "int16",
                      "nodata": -9999, "compress": "lzw"})

    mask_path = MASKS_DIR / "forest_mask.tif"
    with rasterio.open(mask_path, "w", **mask_meta) as dst:
        dst.write(mask[np.newaxis, :, :])
    print(f"\n[SAVED] {mask_path}")

    # ── 7. Apply mask to carbon density ──────────────────────────────────────
    carbon_path = PROCESSED / "carbon_density.tif"
    if carbon_path.exists():
        out_masked = PROCESSED / "masked_carbon_density.tif"
        apply_mask_to_carbon(carbon_path, mask, out_masked, meta)
        print(f"[SAVED] {out_masked}")
    else:
        print("[WARN] carbon_density.tif not found — skipping masked carbon output.")

    print("\n[DONE] Outputs:")
    print(f"  {mask_path}")
    print(f"  {PROCESSED / 'masked_carbon_density.tif'}")


if __name__ == "__main__":
    main()
