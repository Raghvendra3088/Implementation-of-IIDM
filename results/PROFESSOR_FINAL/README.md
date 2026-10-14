# Exploratory Legacy Checkpoint (300 Epochs)

> **Methodological Note (Audit Documentation)**:
> The metrics in this directory (`RMSE: 8.74 Mg C/ha, R²: 0.9474`) were recorded during an exploratory 300-epoch run using legacy network dimensions (`checkpoints/BEST_RESULTS_FINAL/full_model_epoch300.pth`).
> Because the SIREN INR head was subsequently refactored (520 -> 296 channels, as logged in `logs/eval_BEST_epoch300.log`), this legacy artifact is maintained for historical reference and is explicitly separated from the verified, reproducible benchmarks.
>
> The verified reproducible benchmarks for the manuscript are located in:
> - `results/baselines/` (All 6 core baseline models: B0, B1, B2, B2-KD, P1, P1-KD)
> - `Phase2_PIIIDM_Improved/results/` (P1 PI-LDM and P1-KD verified metrics)
> - `Phase3_SwinT_CrossAttention/results/` (Swin-T Champion 12.50, Deterministic 16.87, and No-ETH 15.50 ablations)
