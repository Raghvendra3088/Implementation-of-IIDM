#!/bin/bash
set -e

echo "[$(date)] Starting Phase 3 (End-to-End Fine Tuning)..."
rm -f checkpoints/base_paper/base_resume.pth checkpoints/base_paper/base_best.pth
env CUDA_VISIBLE_DEVICES=0 /DATA1/anil/iidm_venv/bin/python -u -c 'import torch; torch.backends.cudnn.enabled=False; import sys; sys.argv=["src/train_base.py", "--patch_dir", "patches_v2", "--epochs", "100", "--batch_size", "16"]; from src.train_base import main; main()' > logs/base_paper_train_fixed.log 2>&1

echo "[$(date)] Phase 3 Completed! Starting Phase 4A (DDIM Test Evaluation)..."
env CUDA_VISIBLE_DEVICES=0 /DATA1/anil/iidm_venv/bin/python -u -c 'import torch; torch.backends.cudnn.enabled=False; import sys; sys.argv=["src/eval_test_ddim.py", "--patch_dir", "patches_v2", "--batch_size", "16", "--n_steps", "20"]; from src.eval_test_ddim import main; main()' > logs/phase4a_ddim_eval_fixed.log 2>&1

echo "[$(date)] Phase 4A Completed! Starting Phase 4B (Maps & Figures)..."
env CUDA_VISIBLE_DEVICES=0 /DATA1/anil/iidm_venv/bin/python -u -c 'import torch; torch.backends.cudnn.enabled=False; import sys; sys.argv=["src/evaluate_base_maps.py", "--patch_dir", "patches_v2", "--batch_size", "16", "--n_steps", "20"]; from src.evaluate_base_maps import main; main()' > logs/phase4b_maps_eval_fixed.log 2>&1

echo "[$(date)] Pipeline Fully Completed!"
