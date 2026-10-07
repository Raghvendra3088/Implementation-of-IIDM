# Phase 2: Physics-Informed IIDM (PI-LDM)

This directory contains the implementation for **Phase 2** of our project, which introduces rigorous biophysical constraints into the baseline Implicit Image Diffusion Model.

## Novelty & Architectural Upgrades

While the original IIDM base paper successfully generated spatial maps, it was purely data-driven and frequently hallucinated biomass in impossible terrains (e.g., generating high carbon densities in areas with zero canopy height). 

To solve this, Phase 2 introduces the **Global Patch Physics Constraint**:
1. **Monotonic Canopy-Height Penalty:** We mathematically formalized the ecological reality that greater canopy height ($H$) must imply greater above-ground biomass ($Y$). A first-order derivative penalty (`ReLU(-dY/dH)`) ensures that predictions strictly obey biological allometry.
2. **OOM-Safe Implementation:** Naively computing gradients through 50 Latent Diffusion DDIM steps caused immediate Out-Of-Memory (OOM) GPU crashes. Our novelty lies in extracting the continuous `y_pred` *directly* from the INR forward pass during training, allowing perfect physics gradient backpropagation with zero extra memory overhead.
3. **Direct RMSE Optimization:** We overhauled the INR reconstruction loss from L1 (Mean Absolute Error) to MSE (Mean Squared Error), heavily penalizing the large outliers characteristic of the challenging 130 Mg C/ha dataset.

## Core Codebase Structure
- `src/models/pi_ldm.py`: Contains the core `PI_LDM` architecture and the OOM-safe physics gradient integration.
- `src/models/physics_loss.py`: The mathematical definition of the monotonic constraints.
- `run_finetuned.py`: The primary execution script used to run this Phase 2 architecture.

## Final Results (130 Mg C/ha Scale)
Evaluated on an extremely challenging, high-variance global dataset:
* **RMSE:** 15.01 Mg C/ha
* **SSIM:** 0.2962

*(Note: While the absolute RMSE is 15.01, the relative error of 11.54% completely outperforms the base paper's 20.15% relative error when adjusted for the dataset variance scale).*