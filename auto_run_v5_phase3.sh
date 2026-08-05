#!/bin/bash
set -e
export PYTHONHTTPSVERIFY=0

echo "[$(date)] Phase 3: Blockwise Distillation (Teacher UNet -> KD-UNet)..."
/DATA1/anil/iidm_venv/bin/python src/train_unet_blockwise_kd.py --batch_size 16 --epochs 10

echo "[$(date)] Phase 4: End-to-End Fine-Tuning (KD-UNet)..."
/DATA1/anil/iidm_venv/bin/python src/train_base.py --model student --batch_size 16 --epochs 100

echo "[$(date)] Phase 5: Final Evaluation & Map Generation..."
/DATA1/anil/iidm_venv/bin/python src/eval_test_ddim.py
/DATA1/anil/iidm_venv/bin/python src/evaluate_base_maps.py

echo "[$(date)] PIPELINE V5 PHASE 3 ONWARDS COMPLETE!"
