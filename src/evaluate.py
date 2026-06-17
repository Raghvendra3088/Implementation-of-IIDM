import os
import sys
import torch
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.metrics import calculate_metrics
from utils.visualization import save_plots
from models.iidm import IIDM
from preprocessing.dataset import IIDMDataset
from torch.utils.data import DataLoader, Dataset

class LocalDevDataset(Dataset):
    def __len__(self): return 1
    def __getitem__(self, idx):
        return {
            'optical': torch.randn(6, 256, 256),
            'structural': torch.randn(2, 256, 256),
            'carbon': torch.randn(1, 256, 256)
        }

def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ Running True IIDM Evaluation on: {device}")

    try:
        dataset = IIDMDataset(patch_dir=Path('data/patches'))
        if len(dataset) == 0: raise ValueError("Empty directory")
    except Exception:
        print("⚠️ Real dataset missing/incomplete. Auto-switching to LocalDevDataset for testing...")
        dataset = LocalDevDataset()

    # Batch size 1 is standard for detailed evaluation to avoid padding issues
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Initialize Full Model
    model = IIDM(in_channels=6, T=1000).to(device)
    
    # Checkpoint Loading logic
    checkpoint_path = 'checkpoints/true_iidm_epoch_100.pth'
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("✅ Loaded trained weights successfully!")
    else:
        print("⚠️ Warning: No trained weights found. Running evaluation with untrained initialized model.")
    
    model.eval()
    all_metrics = []
    os.makedirs("results/figures", exist_ok=True)

    print("📊 Starting Latent DDIM Inference & INR Decoding...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Evaluating Patches")):
            optical = batch['optical'].to(device)
            gt = batch['carbon'].to(device)

            # Phase 4 Inference: DDIM (50 steps) + INR Decoding
            pred = model.inference(optical, steps=50)
            
            # Calculate metrics
            metrics = calculate_metrics(pred, gt)
            all_metrics.append(metrics)
            
            # Save visual outputs for the first patch
            if i == 0:
                save_plots(pred, gt, "results/figures")
                print("🖼️ Visualizations saved to results/figures/comparison.png")
            
            # For local dev test, just run one batch
            if isinstance(dataset, LocalDevDataset):
                break

    # Aggregate metrics
    avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}
    
    with open("results/metrics.json", "w") as f:
        json.dump(avg_metrics, f, indent=4)

    print("\n✅ Evaluation Complete!")
    print(f"Final Averaged Metrics: {avg_metrics}")

if __name__ == "__main__":
    run_evaluation()
