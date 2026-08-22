#!/bin/bash
FREE_MB_REQUIRED=12000
CHECK_INTERVAL=60
LOG=logs/train_swin_physics_v1.log
mkdir -p logs checkpoints

echo "=== Waiting for ${FREE_MB_REQUIRED}MB free GPU ===" | tee $LOG
echo "Started: $(date)" | tee -a $LOG

while true; do
    BEST_LINE=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
                | awk -F',' '{print $2, $1}' | sort -rn | head -1)
    BEST_FREE=$(echo $BEST_LINE | awk '{print $1}')
    BEST_GPU=$(echo $BEST_LINE | awk '{print $2}' | tr -d ' ')
    echo "$(date '+%H:%M:%S') GPU $BEST_GPU: ${BEST_FREE}MB free" | tee -a $LOG

    if [ "$BEST_FREE" -ge "$FREE_MB_REQUIRED" ]; then
        echo "=== GPU $BEST_GPU has ${BEST_FREE}MB — LAUNCHING ===" | tee -a $LOG
        break
    fi
    sleep $CHECK_INTERVAL
done

CUDA_VISIBLE_DEVICES=$BEST_GPU \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 src/training/train_physics_iidm.py \
    --patch_dir data_base_readonly/processed/patches \
    --save_dir  checkpoints/ \
    --log_path  logs/train_swin_physics_v1.log \
    --epochs    60 \
    --use_swin \
    --use_physics_loss \
    --lambda_phys 0.05 \
    >> $LOG 2>&1

echo "Training finished: $(date)" >> $LOG
