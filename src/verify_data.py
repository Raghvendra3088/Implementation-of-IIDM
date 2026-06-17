import sys
import os
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.dataset import IIDMDataset as CarbonDataset

try:
    dataset = CarbonDataset(patch_dir=Path('data/patches'))
    print(f"Dataset found! Size: {len(dataset)}")
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Optical shape: {sample['optical'].shape}")
    else:
        print("Dataset is empty. You need to run preprocessing first.")
except Exception as e:
    print(f"Error: {e}")
