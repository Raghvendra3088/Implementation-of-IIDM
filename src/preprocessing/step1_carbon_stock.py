"""
IIDM Preprocessing - Step 1: Carbon Stock Calculation
======================================================
Paper reference: Appendix A, Section 1
- Calculates Above-Ground Biomass (AGB) from forest inventory data
- Converts AGB to carbon stock using IPCC default carbon fraction (0.47)
- Outputs per-plot carbon stock values as CSV

Input  : data/raw/inventory/  (CSV or shapefile with forest plot data)
Output : data/processed/carbon_stock.csv
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[1]
INVENTORY_DIR = ROOT / "data" / "raw" / "inventory"
OUT_DIR      = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── IPCC default coefficients (2006 IPCC Guidelines for National GHG Inventories)
CARBON_FRACTION  = 0.47   # C fraction of dry biomass (dimensionless)
ROOT_SHOOT_RATIO = 0.26   # below-ground to above-ground ratio (tropical/subtropical forest)
BEF              = 1.65   # Biomass Expansion Factor (stem → total AGB)

# ── Species-specific Wood Density (g/cm³)
# Add more species as needed for your inventory
WOOD_DENSITY = {
    "Pinus yunnanensis"  : 0.51,
    "Quercus"            : 0.60,
    "Betula"             : 0.55,
    "Mixed broadleaved"  : 0.57,
    "default"            : 0.50,
}


def load_inventory(inventory_dir: Path) -> pd.DataFrame:
    """
    Load forest inventory data.
    Supports: .csv, .shp, .xlsx
    Expected columns (flexible mapping below):
        plot_id, species, dbh_cm, height_m, stem_count, area_ha
    """
    files = list(inventory_dir.glob("*.csv")) + \
            list(inventory_dir.glob("*.shp")) + \
            list(inventory_dir.glob("*.xlsx"))

    if not files:
        raise FileNotFoundError(
            f"No inventory files found in {inventory_dir}\n"
            "Expected .csv / .shp / .xlsx with columns: "
            "plot_id, species, dbh_cm, height_m, stem_count, area_ha"
        )

    f = files[0]
    print(f"[INFO] Loading inventory from: {f.name}")

    if f.suffix == ".shp":
        df = gpd.read_file(f).drop(columns="geometry", errors="ignore")
    elif f.suffix == ".xlsx":
        df = pd.read_excel(f)
    else:
        df = pd.read_csv(f)

    # ── Flexible column renaming ─────────────────────────────────────────────
    rename_map = {
        # common alternate names → standard names used here
        "Plot_ID"   : "plot_id",   "PLOT"     : "plot_id",
        "Species"   : "species",   "SPECIES"  : "species",
        "DBH"       : "dbh_cm",    "Dbh"      : "dbh_cm",
        "Height"    : "height_m",  "HEIGHT"   : "height_m",
        "Stems"     : "stem_count","N_TREES"  : "stem_count",
        "Area"      : "area_ha",   "AREA"     : "area_ha",
    }
    df.rename(columns=rename_map, inplace=True)

    required = ["plot_id", "dbh_cm", "height_m", "stem_count", "area_ha"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )
    return df


def calculate_agb_per_tree(dbh_cm: float, height_m: float,
                            wood_density: float) -> float:
    """
    Allometric equation for AGB (kg/tree).
    Brown (1997) pantropical equation:
        AGB = 0.0509 × ρ × D² × H
    where ρ = wood density (g/cm³), D = DBH (cm), H = height (m)
    """
    return 0.0509 * wood_density * (dbh_cm ** 2) * height_m


def plot_carbon_stock(row: pd.Series) -> float:
    """
    Total carbon stock for a single plot (Mg C / ha).

    Steps:
      1. AGB per tree  (allometric equation)
      2. AGB per plot  (AGB_tree × stem_count)
      3. Total biomass (AGB × BEF)              — accounts for branches/foliage
      4. Total biomass incl. roots (× 1 + R:S ratio)
      5. Carbon stock  (biomass × carbon_fraction)
      6. Per-hectare   (÷ area_ha)
    """
    species      = row.get("species", "default")
    wd           = WOOD_DENSITY.get(species, WOOD_DENSITY["default"])

    agb_tree     = calculate_agb_per_tree(row["dbh_cm"], row["height_m"], wd)  # kg
    agb_plot_kg  = agb_tree * row["stem_count"]                                # kg
    agb_plot_Mg  = agb_plot_kg / 1000.0                                        # Mg

    total_bio    = agb_plot_Mg * BEF * (1 + ROOT_SHOOT_RATIO)                 # Mg
    carbon_Mg    = total_bio * CARBON_FRACTION                                 # Mg C
    carbon_ha    = carbon_Mg / row["area_ha"]                                  # Mg C/ha

    return round(carbon_ha, 4)


def main():
    print("=" * 55)
    print("  STEP 1 — Carbon Stock Calculation")
    print("=" * 55)

    df = load_inventory(INVENTORY_DIR)
    print(f"[INFO] Loaded {len(df)} plots")

    # Calculate carbon stock for each plot
    df["carbon_stock_MgCha"] = df.apply(plot_carbon_stock, axis=1)

    # Summary statistics
    print("\n[RESULT] Carbon Stock Summary (Mg C/ha):")
    print(f"  Mean  : {df['carbon_stock_MgCha'].mean():.2f}")
    print(f"  Std   : {df['carbon_stock_MgCha'].std():.2f}")
    print(f"  Min   : {df['carbon_stock_MgCha'].min():.2f}")
    print(f"  Max   : {df['carbon_stock_MgCha'].max():.2f}")
    print(f"  Count : {len(df)}")

    # Save output
    out_path = OUT_DIR / "carbon_stock.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[SAVED] {out_path}")
    print("  Columns:", list(df.columns))


if __name__ == "__main__":
    main()
