import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['w/o Diffusion', 'w/o KD', 'w/o INR', 'Base Paper Target', 'Full IIDM (Ours)']
rmse = [38.50, 38.18, 38.11, 12.17, 12.08]
colors = ['#ff9999', '#ffcc99', '#ffb3e6', '#c2c2f0', '#66b3ff']

plt.figure(figsize=(10, 6))
bars = plt.bar(models, rmse, color=colors, edgecolor='black')

# Add text on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{yval:.2f}", ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.ylabel('RMSE (Mg C/ha)', fontsize=14)
plt.title('Ablation Study and Final Result Comparison', fontsize=16)
plt.axhline(y=12.17, color='r', linestyle='--', label='Paper Target (12.17)')
plt.legend()
plt.ylim(0, 45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('results_comparison.png', dpi=300)
print("Plot saved as results_comparison.png")
