#!/bin/bash
set -e
echo "[$(date)] Starting Phase 4A (DDIM Table A3 Test Evaluation)..."
env CUDA_VISIBLE_DEVICES=0 /DATA1/anil/iidm_venv/bin/python -u -c 'import torch; torch.backends.cudnn.enabled=False; import sys; sys.argv=["src/eval_test_ddim.py", "--patch_dir", "patches_v2", "--batch_size", "16", "--n_steps", "20"]; from src.eval_test_ddim import main; main()' > logs/phase4a_ddim_eval.log 2>&1

echo "[$(date)] Phase 4A completed! Starting Phase 4B (Map Generation & Visualizations)..."
env CUDA_VISIBLE_DEVICES=0 /DATA1/anil/iidm_venv/bin/python -u -c 'import torch; torch.backends.cudnn.enabled=False; import sys; sys.argv=["src/evaluate_base_maps.py", "--patch_dir", "patches_v2", "--batch_size", "16", "--n_steps", "20"]; from src.evaluate_base_maps import main; main()' > logs/phase4b_maps_eval.log 2>&1

echo "[$(date)] All phases successfully finished!"
