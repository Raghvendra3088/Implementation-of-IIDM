#!/bin/bash
# Master Execution Script for PI-LDM Autonomous Training
# Runs on Anil Server: 172.30.1.14
# Conda Environment: iidm_venv

source /DATA1/anil/miniconda3/bin/activate iidm_venv
cd /DATA1/anil/Physics-Informed-IIDM

mkdir -p logs

echo "Starting Autonomous Training Pipeline for PI-LDM..."

echo "[1/4] Running Preprocessing (Spatial Split & 6-Ch Patch Extraction)..."
# These might already be running, so we check or just run them.
python src/preprocessing/01_spatial_split.py > logs/01_split.log 2>&1
python src/preprocessing/02_extract_patches.py > logs/02_extract.log 2>&1
echo "Preprocessing finished."

# The following training scripts would be fully implemented in the next phase
# Currently they act as placeholders indicating the automated schedule.

echo "[2/4] Phase 2: Training VAE and KD-VGG..."
# python src/training/train_vae.py > logs/vae.log 2>&1
# python src/training/train_kd_vgg.py > logs/kdvgg.log 2>&1

echo "[3/4] Phase 3: Training CNN Baseline and LDM (No Physics)..."
# for seed in 42 123 456; do
#     python src/training/train_cnn_baseline.py --seed $seed > logs/cnn_${seed}.log 2>&1
#     python src/training/train_ldm.py --seed $seed --lambda_phys 0.0 > logs/ldm_${seed}.log 2>&1
# done

echo "[4/4] Phase 4: Training PI-LDM (Proposed Method)..."
# for seed in 42 123 456; do
#     python src/training/train_pi_ldm.py --seed $seed --lambda_phys 0.05 > logs/pildm_${seed}.log 2>&1
# done

echo "Autonomous pipeline completed. Please check logs/ for details."
