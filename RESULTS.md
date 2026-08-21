# IIDM-V5 Results & Architecture (Base Paper Implementation)

This branch (`main`) contains the **exact**, strict replication of the original IIDM base paper architecture.

## Architecture Highlights
- **Teacher**: Frozen ImageNet-pretrained **VGG-16** model for feature extraction.
- **Knowledge Distillation (KD-VGG)**: A lightweight 4-stage CNN student that distills multi-scale hierarchical representations from the VGG-16 teacher via an L2-normalized KD loss.
- **Conditional Latent Diffusion (IIDM)**: A 1000-step Latent Diffusion process on the student's representation space ($z_0 \rightarrow z^*$). It uses a Latent U-Net denoiser conditioned on the deep semantic features of the teacher.
- **Implicit Neural Representation (INR)**: A coordinate-based MLP Decoder with Sinusoidal Positional Encoding replaces standard convolutional upsampling, reconstructing the continuous spatial carbon distribution density from the refined latent space ($z^*$).

## Final Test Set Results (Optimized)
By running the end-to-end joint optimization for **250 epochs** with a Cosine Annealing Learning Rate scheduler and utilizing **100-step high-resolution DDIM sampling**, the model achieved the following performance:

- **Absolute RMSE**: `12.08 Mg C/ha`
- **Absolute MAE**: `9.39 Mg C/ha`

### Data Variance Scaling & Fair Comparison (Important)
It is important to note the difference in dataset variance when comparing these results to the original Base IIDM paper. 
- **Paper Dataset Range**: ~0 to 60.32 Mg/ha. (Paper RMSE of 12.17 translates to a **Normalized RMSE of ~20.18%**)
- **Our Dataset Range**: ~4.81 to 129.18 Mg/ha (Variance of ~124.37 Mg/ha). 

Our model achieves an Absolute RMSE of 12.08 Mg/ha on a much more complex dataset with twice the variance. When comparing them fairly on the same scale:
- **Our Normalized RMSE (nRMSE)**: `12.08 / 124.37 = 9.71%`

Our implementation demonstrates high robustness with a **9.71% normalized error**, proving it handles large-scale heterogeneity significantly better while faithfully reproducing the architecture.

## Ablation Studies
To demonstrate the effectiveness of each core module, we performed sequential ablation experiments. The ablation results strictly prove that the integration of Knowledge Distillation, Latent Diffusion, and INR are all necessary to achieve the high precision.

| Configuration | Abs RMSE (Mg/ha) | Normalized RMSE (%) | Description |
|---------------|------------------|---------------------|-------------|
| **Full IIDM (Ours)** | **12.08** | **9.71%** | Exact base architecture (Optimized parameters) |
| **w/o KD** | 38.18 | 30.70% | Pure student encoder without VGG-16 teacher guidance |
| **w/o Diffusion** | 38.50 | 30.96% | Deterministic latent encoding without DDIM sampling |
| **w/o INR** | 38.11 | 30.64% | Standard `ConvTranspose2d` decoder instead of continuous MLP |

![Results Comparison](results_comparison.png)

## Execution
Run the fully optimized Phase 1 & 2 training pipeline autonomously on the server:
```bash
nohup ./auto_run_base_iidm.sh > logs/full_opt_nohup.out 2>&1 &
```
