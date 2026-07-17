#!/bin/bash
# 3 ablations sequentially — GPU capture + wait
FREE_MB_REQUIRED=8000
CHECK_INTERVAL=30
LOG_DIR=logs
mkdir -p $LOG_DIR checkpoints

wait_gpu() {
    echo "[$1] Waiting for ${FREE_MB_REQUIRED}MB free GPU..."
    while true; do
        BEST_LINE=$(nvidia-smi --query-gpu=index,memory.free \
                    --format=csv,noheader,nounits \
                    | awk -F',' '{print $2, $1}' | sort -rn | head -1)
        BEST_FREE=$(echo $BEST_LINE | awk '{print $1}')
        BEST_GPU=$(echo $BEST_LINE | awk '{print $2}' | tr -d ' ')
        echo "  $(date '+%H:%M:%S') Best GPU $BEST_GPU: ${BEST_FREE}MB free"
        if [ "$BEST_FREE" -ge "$FREE_MB_REQUIRED" ]; then
            echo "  GPU $BEST_GPU acquired for $1"
            export SELECTED_GPU=$BEST_GPU
            return
        fi
        sleep $CHECK_INTERVAL
    done
}

run_ablation() {
    MODE=$1
    LOG=$LOG_DIR/ablation_extra_${MODE}.log
    echo "========================================"
    echo "STARTING: $MODE  at $(date)"
    echo "========================================"

    wait_gpu $MODE

    CUDA_VISIBLE_DEVICES=$SELECTED_GPU \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python3 src/ablation_extra.py \
        --mode $MODE \
        --patch_dir data/processed/patches \
        --epochs 60 \
        --batch_size 4 \
        --save_dir checkpoints/ \
        --log_dir  logs/ \
        2>&1 | tee $LOG

    echo "DONE: $MODE  at $(date)"
    echo ""
}

# Sequential — ek ke baad ek
run_ablation student_only
run_ablation student_kd_inr
run_ablation student_kd_diff

echo "ALL 3 ABLATIONS COMPLETE at $(date)"
echo "Results:"
ls -la checkpoints/ablation_extra_*.pth 2>/dev/null
ls -la logs/ablation_extra_*.json 2>/dev/null
