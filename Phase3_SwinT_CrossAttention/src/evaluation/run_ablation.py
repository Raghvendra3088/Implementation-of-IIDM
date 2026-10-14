#!/usr/bin/env python3
"""
Phase 3 Swin-T & Cross-Attention Ablation Evaluation Suite.

Evaluates Phase 3 configurations:
  1. Flagship Swin-T + Cross-Attention (Champion)
  2. Swin-T + True Knowledge Distillation (VGG16 Teacher Regularizer)
  3. Swin-T Direct Regression (Deterministic Baseline)
  4. Swin-T No-ETH Canopy Height (Modality / Circularity Ablation)
"""

import sys
from pathlib import Path

# Redirect to the dedicated Phase 3 evaluation suite
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from evaluate_swin_models import main as run_swin_evaluation

if __name__ == "__main__":
    print("=" * 70)
    print("Running Phase 3 Vision Transformer (Swin-T) Ablation Benchmark Suite")
    print("=" * 70)
    run_swin_evaluation()
