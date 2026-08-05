#!/bin/bash
# Autonomous Execution Script for IIDM-v2 (Prithvi + Latent Diffusion)
# Assumes virtual environment is activated

set -e
set -o pipefail

# Setup logging
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FULL_LOG="$LOG_DIR/iidm_v2_full_run_$TIMESTAMP.log"

echo "==========================================================" | tee -a "$FULL_LOG"
echo "Starting IIDM-v2 Autonomous Execution Pipeline" | tee -a "$FULL_LOG"
echo "Start Time: $(date)" | tee -a "$FULL_LOG"
echo "==========================================================" | tee -a "$FULL_LOG"

echo "Activating Environment..." | tee -a "$FULL_LOG"
source /DATA1/anil/iidm_venv/bin/activate

# ---------------------------------------------------------
# Phase 1: KL-VAE
# ---------------------------------------------------------
echo "" | tee -a "$FULL_LOG"
echo ">>> [Phase 1] Training KL-VAE (100 Epochs)" | tee -a "$FULL_LOG"
start_time=$(date +%s)
python src/train_vae.py 2>&1 | tee -a "$FULL_LOG"
end_time=$(date +%s)
echo ">>> Phase 1 completed in $((end_time - start_time)) seconds." | tee -a "$FULL_LOG"

# ---------------------------------------------------------
# Phase 2: Prithvi Foundation Teacher Blockwise KD
# ---------------------------------------------------------
echo "" | tee -a "$FULL_LOG"
echo ">>> [Phase 2] Training Prithvi Blockwise KD (12 Blocks x 15 Epochs)" | tee -a "$FULL_LOG"
start_time=$(date +%s)
python src/train_prithvi_blockwise_kd.py 2>&1 | tee -a "$FULL_LOG"
end_time=$(date +%s)
echo ">>> Phase 2 completed in $((end_time - start_time)) seconds." | tee -a "$FULL_LOG"

# ---------------------------------------------------------
# Phase 3 & 4: Latent KD-UNet Diffusion Training
# ---------------------------------------------------------
echo "" | tee -a "$FULL_LOG"
echo ">>> [Phase 4] Training Latent Diffusion KD-UNet (100 Epochs)" | tee -a "$FULL_LOG"
start_time=$(date +%s)
python src/train_v2_ldm.py 2>&1 | tee -a "$FULL_LOG"
end_time=$(date +%s)
echo ">>> Phase 4 completed in $((end_time - start_time)) seconds." | tee -a "$FULL_LOG"

# ---------------------------------------------------------
# Phase 5: Latent DDIM Evaluation & Comparison
# ---------------------------------------------------------
echo "" | tee -a "$FULL_LOG"
echo ">>> [Phase 5] Running Latent DDIM Inference & Evaluation (Test Set)" | tee -a "$FULL_LOG"
start_time=$(date +%s)
python src/eval_v2_ldm_ddim.py 2>&1 | tee -a "$FULL_LOG"
end_time=$(date +%s)
echo ">>> Phase 5 completed in $((end_time - start_time)) seconds." | tee -a "$FULL_LOG"

echo "" | tee -a "$FULL_LOG"
echo "==========================================================" | tee -a "$FULL_LOG"
echo "IIDM-v2 Autonomous Execution Pipeline Finished Successfully!" | tee -a "$FULL_LOG"
echo "End Time: $(date)" | tee -a "$FULL_LOG"
echo "==========================================================" | tee -a "$FULL_LOG"
