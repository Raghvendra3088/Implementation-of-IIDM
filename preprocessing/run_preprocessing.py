"""
IIDM Preprocessing — Run All Steps
====================================
Runs all 4 preprocessing steps in sequence.

Usage:
    python src/preprocessing/run_preprocessing.py

Steps:
    1. Carbon stock calculation      → data/processed/carbon_stock.csv
    2. Carbon density mapping        → data/processed/carbon_density.tif
    3. Forest / non-forest mask      → data/masks/forest_mask.tif
    4. Patch extraction & splits     → data/processed/patches/
"""

import sys
import time
import traceback
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "preprocessing"))

def run_step(module_path: str, step_name: str) -> bool:
    print(f"\n{'#'*60}")
    print(f"  RUNNING: {step_name}")
    print(f"{'#'*60}")
    t0 = time.time()
    try:
        import importlib.util
        spec   = importlib.util.spec_from_file_location("step", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.main()
        elapsed = time.time() - t0
        print(f"\n  ✓ {step_name} completed in {elapsed:.1f}s")
        return True
    except FileNotFoundError as e:
        print(f"\n  ✗ {step_name} FAILED — Missing file:\n    {e}")
        return False
    except Exception as e:
        print(f"\n  ✗ {step_name} FAILED:\n    {e}")
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("  IIDM PREPROCESSING PIPELINE")
    print("=" * 60)

    preproc_dir = ROOT / "src" / "preprocessing"

    steps = [
        (str(preproc_dir / "step1_carbon_stock.py"),
         "Step 1: Carbon Stock Calculation"),

        (str(preproc_dir / "step2_carbon_density.py"),
         "Step 2: Carbon Density Mapping"),

        (str(preproc_dir / "step3_forest_mask.py"),
         "Step 3: Forest/Non-Forest Mask"),

        (str(preproc_dir / "step4_patch_extraction.py"),
         "Step 4: Patch Extraction & Splits"),
    ]

    results = []
    for path, name in steps:
        ok = run_step(path, name)
        results.append((name, ok))
        if not ok:
            print(f"\n[PIPELINE STOPPED] Fix the error above before continuing.")
            break

    print(f"\n{'='*60}")
    print("  PIPELINE SUMMARY")
    print(f"{'='*60}")
    for name, ok in results:
        status = "✓ DONE" if ok else "✗ FAILED"
        print(f"  {status}  {name}")

    all_ok = all(ok for _, ok in results)
    if all_ok:
        print(f"\n  All steps complete!")
        print(f"  Patches ready at: data/processed/patches/")
        print(f"  Next: implement models in src/models/")
    else:
        print(f"\n  Fix the failed step and re-run.")
    print("=" * 60)


if __name__ == "__main__":
    main()
