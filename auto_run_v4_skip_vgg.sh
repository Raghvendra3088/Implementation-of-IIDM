#!/bin/bash
set -e

export PYTHONHTTPSVERIFY=0

echo "[$(date)] Phase 2: Training Teacher UNet (31M) from Scratch..."
rm -f checkpoints/teacher_unet/teacher_best.pth checkpoints/teacher_unet/teacher_resume.pth
env CUDA_VISIBLE_DEVICES=0 /DATA1/anil/iidm_venv/bin/python -u -c 'import torch; torch.backends.cudnn.enabled=False; import sys; sys.argv=["src/train_teacher_unet.py", "--patch_dir", "patches_v2", "--epochs", "100", "--batch_size", "16"]; from src.train_teacher_unet import main; main()' > logs/teacher_unet_train_v4.log 2>&1

echo "[$(date)] Phase 3: UNet PCA Blockwise Distillation..."
env CUDA_VISIBLE_DEVICES=0 /DATA1/anil/iidm_venv/bin/python -u -c 'import torch; torch.backends.cudnn.enabled=False; import sys; sys.argv=["src/train_unet_blockwise_kd.py", "--patch_dir", "patches_v2", "--batch_size", "16"]; from src.train_unet_blockwise_kd import main; main()' > logs/train_unet_blockwise_v4.log 2>&1

echo "[$(date)] Phase 4: End-to-End Fine Tuning (Student)..."
rm -f checkpoints/base_paper/base_resume.pth checkpoints/base_paper/base_best.pth
env CUDA_VISIBLE_DEVICES=0 /DATA1/anil/iidm_venv/bin/python -u -c 'import torch; torch.backends.cudnn.enabled=False; import sys; sys.argv=["src/train_base.py", "--patch_dir", "patches_v2", "--epochs", "100", "--batch_size", "16"]; from src.train_base import main; main()' > logs/base_paper_train_v4.log 2>&1

echo "[$(date)] Phase 5A: DDIM Test Evaluation..."
env CUDA_VISIBLE_DEVICES=0 /DATA1/anil/iidm_venv/bin/python -u -c 'import torch; torch.backends.cudnn.enabled=False; import sys; sys.argv=["src/eval_test_ddim.py", "--patch_dir", "patches_v2", "--batch_size", "16", "--n_steps", "20"]; from src.eval_test_ddim import main; main()' > logs/phase5a_ddim_eval_v4.log 2>&1

echo "[$(date)] Phase 5B: Maps & Figures..."
env CUDA_VISIBLE_DEVICES=0 /DATA1/anil/iidm_venv/bin/python -u -c 'import torch; torch.backends.cudnn.enabled=False; import sys; sys.argv=["src/evaluate_base_maps.py", "--patch_dir", "patches_v2", "--batch_size", "16", "--n_steps", "20"]; from src.evaluate_base_maps import main; main()' > logs/phase5b_maps_eval_v4.log 2>&1

echo "[$(date)] Full Pipeline V4 (Skip VGG) Completed!"
