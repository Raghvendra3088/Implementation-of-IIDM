import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import glob
from src.models.prithvi_teacher import PrithviTeacher12

class ImageSubsetDataset(Dataset):
    def __init__(self, patch_dir, max_samples=500):
        self.files = sorted(glob.glob(os.path.join(patch_dir, 'train', '*.npz')))[:max_samples]
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        d = np.load(self.files[i])
        return torch.from_numpy(d['image'])

def main():
    torch.backends.cudnn.enabled = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    teacher = PrithviTeacher12().to(device).eval()
    ds = ImageSubsetDataset('data/processed/patches_v2', max_samples=500)
    ld = DataLoader(ds, batch_size=8, shuffle=False)
    
    print(f"Processing {len(ds)} images for PCA calculation...")
    
    all_covariances = [0] * 12
    N = 0
    
    for x in ld:
        x = x.to(device)
        B = x.shape[0]
        N += B
        
        with torch.no_grad():
            features = teacher(x)  # List of 12 tensors [B, 768, 256, 256]
            
        for i, feat in enumerate(features):
            # feat: [B, C, H, W]
            B_f, C, H, W = feat.shape
            f = feat.view(B_f, C, -1) # [B, C, H*W]
            
            # Compute covariance for this batch
            f_mean = f.mean(dim=2, keepdim=True)
            f_bar = f - f_mean
            
            # Covariance matrix C x C
            # We average over spatial dimension first, then sum across batch
            batch_cov = torch.bmm(f_bar, f_bar.transpose(1, 2)).mean(dim=0) / (H * W)
            all_covariances[i] += batch_cov.cpu()
            
    print("\nCalculated Covariance Matrices. Computing Eigenvalues (mCEV >= 0.85)...")
    
    student_channels = []
    
    for i in range(12):
        cov = all_covariances[i] / N
        
        # SVD/Eigendecomposition
        eigenvalues, _ = torch.linalg.eigh(cov)
        eigenvalues = eigenvalues.flip(dims=(0,)) # Sort descending
        
        total_var = eigenvalues.sum().item()
        cum_var = torch.cumsum(eigenvalues, dim=0).numpy() / total_var
        
        k = np.argmax(cum_var >= 0.85) + 1
        student_channels.append(int(k))
        
        print(f"Block {i+1}: Original Channels = 768 -> KD Channels = {k} (Explained Var = {cum_var[k-1]*100:.2f}%)")
        
    print(f"\nFinal PRITHVI_STUDENT_CH_12 = {student_channels}")

if __name__ == '__main__':
    main()
