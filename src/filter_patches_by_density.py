"""
Filter existing patches_v2 dataset by GEDI-label density.
Keeps only patches with >= min_gedi_frac valid-pixel fraction in mask.
Copies filtered patches into a new patches_v2_filtered dir with same split structure.
"""
import os, glob, shutil
import numpy as np

SRC_ROOT = 'data/processed/patches_v2'
DST_ROOT = 'data/processed/patches_v2_filtered'
MIN_GEDI_FRAC = 0.02   # tune this — start with 2%

for split in ['train', 'val', 'test']:
    src_dir = os.path.join(SRC_ROOT, split)
    dst_dir = os.path.join(DST_ROOT, split)
    os.makedirs(dst_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(src_dir, '*.npz')))
    kept = 0
    for f in files:
        d = np.load(f)
        frac = d['mask'].mean()
        if frac >= MIN_GEDI_FRAC:
            shutil.copy(f, os.path.join(dst_dir, os.path.basename(f)))
            kept += 1
    print(f"{split}: kept {kept}/{len(files)} ({100*kept/max(len(files),1):.1f}%)")

