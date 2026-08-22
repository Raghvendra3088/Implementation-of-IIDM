#!/bin/bash
# Streamlined Execution Script for PI-LDM Autonomous Training
# Runs on Anil Server: 172.30.1.14
# Conda Environment: iidm_venv

# Activate conda
source /DATA1/anil/iidm_venv/bin/activate
cd /DATA1/anil/Physics-Informed-IIDM

mkdir -p logs

echo "Starting Streamlined Autonomous Training Pipeline for PI-LDM..."

# Phase 1 (Preprocessing) is completely done locally, data is synced in data/processed/patches_6ch

echo "[1/2] Training VAE..."
python src/training/train_vae.py > logs/vae.log 2>&1
echo "VAE training finished."

echo "[2/2] Training PI-LDM (Proposed Method & Sensitivity Analysis)..."
# Train the main seeds for the default lambda
for seed in 42 123 456; do
    echo "Training PI-LDM Seed $seed..."
    python src/training/train_pi_ldm.py --seed $seed --lambda_phys 0.05 > logs/pildm_seed${seed}_L0.05.log 2>&1
done

# Train sensitivity analysis for one seed (e.g., 42)
for lambda_val in 0.001 0.01 0.1 0.5 1.0; do
    echo "Training PI-LDM Seed 42 with Lambda $lambda_val..."
    python src/training/train_pi_ldm.py --seed 42 --lambda_phys $lambda_val > logs/pildm_seed42_L${lambda_val}.log 2>&1
done

echo "Autonomous pipeline completed. Please check logs/ for details."
