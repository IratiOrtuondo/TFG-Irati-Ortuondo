#!/usr/bin/env python3
"""Quick script to plot native resolution SMAP copol data."""
import numpy as np
import matplotlib.pyplot as plt

# Load native resolution co-pol data for 20150607
data = np.load('data/interim/aligned-smap-copol-20150607-vv-native.npz', allow_pickle=True)

print("Keys:", list(data.keys()))

# Get native co-pol data
copol_native = data['S_copol_dB_native']
print(f"Shape: {copol_native.shape}")

# Statistics
valid = copol_native[np.isfinite(copol_native)]
print(f"Valid pixels: {valid.size}/{copol_native.size}")
if valid.size > 0:
    print(f"Range: [{valid.min():.2f}, {valid.max():.2f}] dB")
    print(f"Mean: {valid.mean():.2f} dB")
    print(f"Std: {valid.std():.2f} dB")

# Plot
fig, ax = plt.subplots(figsize=(14, 10))
cmap = plt.get_cmap('viridis').copy()
cmap.set_bad('lightgray', alpha=0.3)

im = ax.imshow(copol_native, origin='upper', cmap=cmap, vmin=-25, vmax=-5)
ax.set_title('SMAP Co-Pol VV Native Resolution (~3km) - 20150607', 
             fontsize=16, fontweight='bold')
ax.set_xlabel('Column', fontsize=12)
ax.set_ylabel('Row', fontsize=12)
plt.colorbar(im, ax=ax, label='σ⁰_VV [dB]', shrink=0.8)

if valid.size > 0:
    stats = f"Valid: {valid.size}\nMin: {valid.min():.2f} dB\nMax: {valid.max():.2f} dB\nMean: {valid.mean():.2f} dB"
    ax.text(0.02, 0.98, stats, transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

plt.tight_layout()

import os
os.makedirs('data/interim/plots', exist_ok=True)
outpath = 'data/interim/plots/aligned_smap_copol_20150607_vv_native.png'
fig.savefig(outpath, dpi=200, bbox_inches='tight')
print(f"\nSaved: {outpath}")
plt.show()
