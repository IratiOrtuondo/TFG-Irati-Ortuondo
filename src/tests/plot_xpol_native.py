#!/usr/bin/env python3
"""Quick script to plot native resolution SMAP cross-pol data."""
import numpy as np
import matplotlib.pyplot as plt

# Load native resolution cross-pol data for 20150607
data = np.load('data/interim/aligned-smap-xpol-20150607-native.npz', allow_pickle=True)

print("Keys:", list(data.keys()))

# Get native cross-pol data
xpol_native = data['S_xpol_dB_native']
print(f"Shape: {xpol_native.shape}")

# Statistics
valid = xpol_native[np.isfinite(xpol_native)]
print(f"Valid pixels: {valid.size}/{xpol_native.size}")
if valid.size > 0:
    print(f"Range: [{valid.min():.2f}, {valid.max():.2f}] dB")
    print(f"Mean: {valid.mean():.2f} dB")
    print(f"Std: {valid.std():.2f} dB")

# Plot
fig, ax = plt.subplots(figsize=(14, 10))
cmap = plt.get_cmap('plasma').copy()
cmap.set_bad('lightgray', alpha=0.3)

im = ax.imshow(xpol_native, origin='upper', cmap=cmap, vmin=-35, vmax=-15)
ax.set_title('SMAP Cross-Pol Native Resolution (~3km) - 20150607', 
             fontsize=16, fontweight='bold')
ax.set_xlabel('Column', fontsize=12)
ax.set_ylabel('Row', fontsize=12)
plt.colorbar(im, ax=ax, label='σ⁰_HV [dB]', shrink=0.8)

if valid.size > 0:
    stats = f"Valid: {valid.size}\nMin: {valid.min():.2f} dB\nMax: {valid.max():.2f} dB\nMean: {valid.mean():.2f} dB"
    ax.text(0.02, 0.98, stats, transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

plt.tight_layout()

import os
os.makedirs('data/interim/plots', exist_ok=True)
outpath = 'data/interim/plots/aligned_smap_xpol_20150607_native.png'
fig.savefig(outpath, dpi=200, bbox_inches='tight')
print(f"\nSaved: {outpath}")
plt.show()
