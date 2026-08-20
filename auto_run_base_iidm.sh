#!/bin/bash
# Autonomous Execution Script for Base IIDM Project
# Architecture: VGG-16 Teacher -> CNN Student -> Latent Diffusion -> INR
# Fully optimized for Base IIDM run

set -e
set -o pipefail

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FULL_LOG="$LOG_DIR/iidm_full_run_$TIMESTAMP.log"

echo "==========================================================" | tee -a "$FULL_LOG"
echo "Starting Base IIDM Full Run (Optimized Parameters)" | tee -a "$FULL_LOG"
echo "Architecture: VGG-16 Teacher -> CNN Student -> Latent Diffusion -> INR" | tee -a "$FULL_LOG"
echo "Start Time: $(date)" | tee -a "$FULL_LOG"
echo "==========================================================" | tee -a "$FULL_LOG"

echo "Activating Environment..." | tee -a "$FULL_LOG"
source /DATA1/anil/iidm_venv/bin/activate

# ---------------------------------------------------------
# Phase 1: End-to-End Joint Training
# ---------------------------------------------------------
echo "" | tee -a "$FULL_LOG"
echo ">>> [Phase 1] Joint End-to-End Optimization (250 Epochs)" | tee -a "$FULL_LOG"
start_time=$(date +%s)
python src/train_iidm_base.py --epochs 250 2>&1 | tee -a "$FULL_LOG"
end_time=$(date +%s)
echo ">>> Phase 1 completed in $((end_time - start_time)) seconds." | tee -a "$FULL_LOG"

# ---------------------------------------------------------
# Phase 2: DDIM Latent Inference & INR Decoding
# ---------------------------------------------------------
echo "" | tee -a "$FULL_LOG"
echo ">>> [Phase 2] Running Latent DDIM Inference & Evaluation (100 Steps)" | tee -a "$FULL_LOG"
start_time=$(date +%s)
python src/eval_iidm_base.py --n_steps 100 2>&1 | tee -a "$FULL_LOG"
end_time=$(date +%s)
echo ">>> Phase 2 completed in $((end_time - start_time)) seconds." | tee -a "$FULL_LOG"

echo "" | tee -a "$FULL_LOG"
echo "==========================================================" | tee -a "$FULL_LOG"
echo "Optimized Full IIDM Pipeline Finished Successfully!" | tee -a "$FULL_LOG"
echo "End Time: $(date)" | tee -a "$FULL_LOG"
echo "==========================================================" | tee -a "$FULL_LOG"
