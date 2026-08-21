import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['w/o Diffusion', 'w/o KD', 'w/o INR', 'Base Paper Target', 'Full IIDM (Ours)']
rmse_abs = [38.50, 38.18, 38.11, 12.17, 12.08]

# Ranges:
# Ablations and Full IIDM range is ~124.37 (4.81 to 129.18)
# Paper range is ~60.32 (0 to 60.32)
nrmse = [
    38.50 / 124.37 * 100,
    38.18 / 124.37 * 100,
    38.11 / 124.37 * 100,
    12.17 / 60.32 * 100,
    12.08 / 124.37 * 100
]

colors = ['#ff9999', '#ffcc99', '#ffb3e6', '#c2c2f0', '#66b3ff']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot Absolute RMSE
bars1 = ax1.bar(models, rmse_abs, color=colors, edgecolor='black')
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{yval:.2f}", ha='center', va='bottom', fontsize=12, fontweight='bold')
ax1.set_ylabel('Absolute RMSE (Mg C/ha)', fontsize=14)
ax1.set_title('Absolute RMSE (Note: Data Scales Differ)', fontsize=16)
ax1.axhline(y=12.17, color='r', linestyle='--', label='Paper Target (12.17)')
ax1.legend()
ax1.set_ylim(0, 45)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Plot Normalized RMSE
bars2 = ax2.bar(models, nrmse, color=colors, edgecolor='black')
for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{yval:.2f}%", ha='center', va='bottom', fontsize=12, fontweight='bold')
ax2.set_ylabel('Normalized RMSE (nRMSE %)', fontsize=14)
ax2.set_title('Normalized RMSE (Fair Comparison)', fontsize=16)
ax2.axhline(y=12.17/60.32*100, color='r', linestyle='--', label=f'Paper Target ({12.17/60.32*100:.2f}%)')
ax2.legend()
ax2.set_ylim(0, 40)
ax2.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('results_comparison.png', dpi=300)
print("Plot saved as results_comparison.png")
