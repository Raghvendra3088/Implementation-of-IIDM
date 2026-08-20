#!/bin/bash
# Autonomous Execution Script for Base IIDM Ablation Studies
# Runs 4 configurations sequentially: Full, w/o KD, w/o Diffusion, w/o INR

set -e
set -o pipefail

LOG_DIR="logs_ablation"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FULL_LOG="$LOG_DIR/iidm_ablations_$TIMESTAMP.log"

echo "==========================================================" | tee -a "$FULL_LOG"
echo "Starting Base IIDM Ablation Studies Pipeline" | tee -a "$FULL_LOG"
echo "Configurations: Full, no_kd, no_diffusion, no_inr" | tee -a "$FULL_LOG"
echo "Start Time: $(date)" | tee -a "$FULL_LOG"
echo "==========================================================" | tee -a "$FULL_LOG"

echo "Activating Environment..." | tee -a "$FULL_LOG"
source /DATA1/anil/iidm_venv/bin/activate

ABLATIONS=("none" "no_kd" "no_diffusion" "no_inr")
declare -A RESULTS_RMSE
declare -A RESULTS_MAE
declare -A RESULTS_R2

for ABLATION in "${ABLATIONS[@]}"; do
    echo "" | tee -a "$FULL_LOG"
    echo "==========================================================" | tee -a "$FULL_LOG"
    echo ">>> Starting Ablation: $ABLATION" | tee -a "$FULL_LOG"
    echo "==========================================================" | tee -a "$FULL_LOG"
    
    CKPT_PATH="checkpoints/ablation_${ABLATION}.pth"
    
    start_time=$(date +%s)
    
    echo ">> [Phase 1] Training ($ABLATION)..." | tee -a "$FULL_LOG"
    python src/train_iidm_base.py --ablation $ABLATION --save_path $CKPT_PATH 2>&1 | tee -a "$FULL_LOG"
    
    echo ">> [Phase 2] Evaluating ($ABLATION)..." | tee -a "$FULL_LOG"
    # Capture the output of eval to parse the metrics
    EVAL_OUT=$(python src/eval_iidm_base.py --ablation $ABLATION --ckpt $CKPT_PATH)
    echo "$EVAL_OUT" | tee -a "$FULL_LOG"
    
    # Parse metrics
    RMSE=$(echo "$EVAL_OUT" | grep "Test RMSE" | awk '{print $4}')
    MAE=$(echo "$EVAL_OUT" | grep "Test MAE" | awk '{print $4}')
    R2=$(echo "$EVAL_OUT" | grep "Test R2" | awk '{print $4}')
    
    RESULTS_RMSE[$ABLATION]=$RMSE
    RESULTS_MAE[$ABLATION]=$MAE
    RESULTS_R2[$ABLATION]=$R2
    
    end_time=$(date +%s)
    echo ">>> Ablation $ABLATION completed in $((end_time - start_time)) seconds." | tee -a "$FULL_LOG"
done

echo "" | tee -a "$FULL_LOG"
echo "==========================================================" | tee -a "$FULL_LOG"
echo "FINAL ABLATION RESULTS" | tee -a "$FULL_LOG"
echo "==========================================================" | tee -a "$FULL_LOG"
printf "%-15s | %-10s | %-10s | %-10s\n" "Configuration" "RMSE" "MAE" "R2" | tee -a "$FULL_LOG"
echo "----------------------------------------------------------" | tee -a "$FULL_LOG"
for ABLATION in "${ABLATIONS[@]}"; do
    printf "%-15s | %-10s | %-10s | %-10s\n" "$ABLATION" "${RESULTS_RMSE[$ABLATION]}" "${RESULTS_MAE[$ABLATION]}" "${RESULTS_R2[$ABLATION]}" | tee -a "$FULL_LOG"
done
echo "==========================================================" | tee -a "$FULL_LOG"
echo "Ablation Studies Pipeline Finished Successfully!" | tee -a "$FULL_LOG"
echo "End Time: $(date)" | tee -a "$FULL_LOG"
echo "==========================================================" | tee -a "$FULL_LOG"
