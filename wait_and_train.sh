#!/bin/bash
echo "Waiting for free GPU (>35GB)..."
while true; do
    GPU=$(nvidia-smi --query-gpu=index,memory.free \
          --format=csv,noheader,nounits | \
          awk -F',' '$2 > 35000 {print $1; exit}')
    
    if [ ! -z "$GPU" ]; then
        echo "$(date): Free GPU $GPU found! Starting full training..."
        CUDA_VISIBLE_DEVICES=$GPU python src/train.py \
          --epochs 200 \
          --batch_size 16 \
          --lr 1e-4 \
          --patch_dir data/processed/patches_s64 \
          --save_every 10 \
          --workers 4 \
          > logs/train_v2_full.log 2>&1
        echo "Training complete!"
        break
    fi

    echo "$(date): No free GPU. Retry in 10 min..."
    sleep 600
done
