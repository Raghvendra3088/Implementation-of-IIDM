#!/bin/bash
LOG=/DATA1/anil/Physics-Informed-IIDM/logs/train_swin_physics_v1.log
mkdir -p /DATA1/anil/Physics-Informed-IIDM/logs
mkdir -p /DATA1/anil/Physics-Informed-IIDM/checkpoints

echo "=== Physics-Informed IIDM Training ===" | tee $LOG
echo "Started: $(date)" | tee -a $LOG
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader | tee -a $LOG

cd /DATA1/anil/Physics-Informed-IIDM

PYTHONPATH=/DATA1/anil/Physics-Informed-IIDM \
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 src/training/train_physics_iidm.py \
    --patch_dir data/processed/patches \
    --save_dir  checkpoints/ \
    --log_path  logs/train_swin_physics_v1.log \
    --epochs    60 \
    --batch_size 4 \
    --use_swin \
    --use_physics_loss \
    --lambda_phys 0.05 \
    2>&1 | tee -a $LOG

echo "Training finished: $(date)" | tee -a $LOG
