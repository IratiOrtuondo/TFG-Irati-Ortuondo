#!/usr/bin/env python3
"""Plot both native and coarse resolution SMAP cross-pol data."""
import numpy as np
import matplotlib.pyplot as plt

# Load both files
data_coarse = np.load('data/interim/aligned-smap-xpol-20150607.npz', allow_pickle=True)
data_native = np.load('data/interim/aligned-smap-xpol-20150607-native.npz', allow_pickle=True)

print("=== COARSE RESOLUTION ===")
print("Keys:", list(data_coarse.keys()))
xpol_coarse = data_coarse['S_xpol_dB']
print(f"Shape: {xpol_coarse.shape}")
valid_coarse = xpol_coarse[np.isfinite(xpol_coarse)]
print(f"Valid pixels: {valid_coarse.size}/{xpol_coarse.size}")
if valid_coarse.size > 0:
    print(f"Range: [{valid_coarse.min():.2f}, {valid_coarse.max():.2f}] dB")
    print(f"Mean: {valid_coarse.mean():.2f} dB")

print("\n=== NATIVE RESOLUTION ===")
print("Keys:", list(data_native.keys()))
xpol_native = data_native['S_xpol_dB_native']
print(f"Shape: {xpol_native.shape}")
valid_native = xpol_native[np.isfinite(xpol_native)]
print(f"Valid pixels: {valid_native.size}/{xpol_native.size}")
if valid_native.size > 0:
    print(f"Range: [{valid_native.min():.2f}, {valid_native.max():.2f}] dB")
    print(f"Mean: {valid_native.mean():.2f} dB")

# Create figure with both plots
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

cmap = plt.get_cmap('plasma').copy()
cmap.set_bad('lightgray', alpha=0.3)

# Plot 1: Coarse resolution
im1 = axes[0].imshow(xpol_coarse, origin='upper', cmap=cmap, vmin=-35, vmax=-15)
axes[0].set_title('SMAP Cross-Pol Coarse (~36km)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Column', fontsize=11)
axes[0].set_ylabel('Row', fontsize=11)
plt.colorbar(im1, ax=axes[0], label='σ⁰_HV [dB]', shrink=0.8)

if valid_coarse.size > 0:
    stats = f"Valid: {valid_coarse.size}\nMin: {valid_coarse.min():.2f} dB\nMax: {valid_coarse.max():.2f} dB\nMean: {valid_coarse.mean():.2f} dB"
    axes[0].text(0.02, 0.98, stats, transform=axes[0].transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

# Plot 2: Native resolution
im2 = axes[1].imshow(xpol_native, origin='upper', cmap=cmap, vmin=-35, vmax=-15)
axes[1].set_title('SMAP Cross-Pol Native (~3km)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Column', fontsize=11)
axes[1].set_ylabel('Row', fontsize=11)
plt.colorbar(im2, ax=axes[1], label='σ⁰_HV [dB]', shrink=0.8)

if valid_native.size > 0:
    stats = f"Valid: {valid_native.size}\nMin: {valid_native.min():.2f} dB\nMax: {valid_native.max():.2f} dB\nMean: {valid_native.mean():.2f} dB"
    axes[1].text(0.02, 0.98, stats, transform=axes[1].transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

fig.suptitle('SMAP Cross-Pol - 20150607', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()

import os
os.makedirs('data/interim/plots', exist_ok=True)
outpath = 'data/interim/plots/aligned_smap_xpol_20150607_comparison.png'
fig.savefig(outpath, dpi=200, bbox_inches='tight')
print(f"\nSaved: {outpath}")
plt.show()
