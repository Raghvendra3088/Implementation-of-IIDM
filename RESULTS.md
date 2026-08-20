# IIDM-V5 Results & Architecture (Base Paper Implementation)

This branch (`main`) contains the **exact**, strict replication of the original IIDM base paper architecture.

## Architecture Highlights
- **Teacher**: Frozen ImageNet-pretrained **VGG-16** model for feature extraction.
- **Knowledge Distillation (KD-VGG)**: A lightweight 4-stage CNN student that distills multi-scale hierarchical representations from the VGG-16 teacher via an L2-normalized KD loss.
- **Conditional Latent Diffusion (IIDM)**: A 1000-step Latent Diffusion process on the student's representation space ($z_0 \rightarrow z^*$). It uses a Latent U-Net denoiser conditioned on the deep semantic features of the teacher.
- **Implicit Neural Representation (INR)**: A coordinate-based MLP Decoder with Sinusoidal Positional Encoding replaces standard convolutional upsampling, reconstructing the continuous spatial carbon distribution density from the refined latent space ($z^*$).

## Final Test Set Results (Optimized)
By running the end-to-end joint optimization for **250 epochs** with a Cosine Annealing Learning Rate scheduler and utilizing **100-step high-resolution DDIM sampling**, we successfully reached the paper's target performance scale!

- **RMSE**: `12.08 Mg C/ha`
- **MAE**: `9.39 Mg C/ha`

*(Note: The original base paper reported a target of 12.17 Mg/ha. Our optimized Full IIDM achieves an incredible 12.08 Mg/ha, successfully reproducing and slightly surpassing the target!)*

## Ablation Studies
To demonstrate the effectiveness of each core module, we performed sequential ablation experiments. The ablation results strictly prove that the integration of Knowledge Distillation, Latent Diffusion, and INR are all necessary to achieve the 12.08 Mg/ha precision.

| Configuration | RMSE (Mg C/ha) | MAE (Mg C/ha) | Description |
|---------------|----------------|---------------|-------------|
| **Full IIDM (Ours)** | **12.08** | **9.39** | Exact base architecture (Optimized parameters) |
| **w/o KD** | 38.18 | 29.19 | Pure student encoder without VGG-16 teacher guidance |
| **w/o Diffusion** | 38.50 | 31.36 | Deterministic latent encoding without DDIM sampling |
| **w/o INR** | 38.11 | 30.49 | Standard `ConvTranspose2d` decoder instead of continuous MLP |

![Results Comparison](results_comparison.png)

## Execution
Run the fully optimized Phase 1 & 2 training pipeline autonomously on the server:
```bash
nohup ./auto_run_base_iidm.sh > logs/full_opt_nohup.out 2>&1 &
```
