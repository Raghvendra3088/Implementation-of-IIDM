# Phase 3: Swin-T & Cross-Attention (Flagship Architecture)

This directory contains the implementation for **Phase 3** of our project. It represents a fundamental structural departure from both the base paper and Phase 2, resulting in our absolute highest-precision carbon estimations.

## Novelty & Architectural Upgrades

Phase 2 proved that the standard CNN spatial encoder lacked the parameter capacity and global context to map the extreme variances of the 130 Mg C/ha dataset under strict physical constraints. 

Phase 3 introduces massive architectural overhauls:
1. **Swin-Tiny (Swin-T) Transformer Encoder:** We completely ripped out the lightweight 4-stage CNN and replaced it with a Swin-T architecture. Utilizing shifted window attention mechanisms, Swin-T extracts robust, global spatial context (like entire mountain ridges or valleys) that standard local convolutions completely miss.
2. **Query-Key-Value Cross-Attention:** Instead of naively concatenating the spatial features into the UNet, the Swin-T features are dynamically injected using strict Cross-Attention blocks. This allows the Diffusion UNet noise predictor to selectively attend to the most biophysically relevant geometries during generation.
3. **Freedom from Knowledge Distillation (KD):** We proved that strictly mimicking an ImageNet VGG16 teacher forces the encoder to learn generic "perceptual" features (cats, dogs, shapes) rather than Carbon-specific features, bottlenecking raw precision. 

## Core Codebase Structure
- `src/models/swin_encoder.py`: The custom Swin-Tiny spatial encoder adapted for 6-channel multisource remote sensing tensors.
- `src/models/diffusion.py` / `pi_ldm.py`: Modified to utilize Cross-Attention mechanisms.
- `train_swin.py`: The primary execution script for training the flagship Swin-T architecture.

## Final Results (130 Mg C/ha Scale)
This architecture produced the absolute best results of the entire project lifecycle:

**1. Pure Swin-T + Cross-Attention (The Champion)**
* **RMSE:** 12.50 Mg C/ha  *(An incredible 9.6% relative error!)*
* **SSIM:** 0.322

**2. Swin-T + True KD (The Structural Regularizer)**
* **RMSE:** 13.01 Mg C/ha
* **SSIM:** 0.4800
* *Analysis:* Forcing the Swin-T to mimic VGG16 causes a slight regression in pure RMSE, but acts as a massive structural regularizer, skyrocketing visual map clarity (SSIM).
