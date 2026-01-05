#!/usr/bin/env python3
"""
Plot step2 beta/gamma results: original and recalc, differences, and stats.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
NPZ_ORIG = BASE / "data" / "interim" / "step3_beta_gamma.npz"
NPZ_RECALC = BASE / "data" / "interim" / "step3_beta_gamma_recalc.npz"
OUT_DIR = BASE / "data" / "processed" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_npz(p):
    d = np.load(p)
    beta = d['beta_K_per_dB']
    gamma = d['Gamma_dimensionless']
    n_samples = d['n_samples'] if 'n_samples' in d else None
    r2 = d['r2'] if 'r2' in d else None
    meta = {k: d[k] for k in ['height','width'] if k in d}
    return dict(beta=beta, gamma=gamma, n_samples=n_samples, r2=r2, meta=meta)

print('Loading files...')
if not NPZ_ORIG.exists() and not NPZ_RECALC.exists():
    raise SystemExit('No beta/gamma NPZ files found')

orig = load_npz(NPZ_ORIG) if NPZ_ORIG.exists() else None
recalc = load_npz(NPZ_RECALC) if NPZ_RECALC.exists() else None

# Prepare plotting
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Step2 Beta/Gamma — Original vs Recalc', fontsize=16, fontweight='bold')

# Row 0: beta maps
ax = axes[0,0]
if orig is not None:
    im = ax.imshow(orig['beta'], cmap='RdYlBu', aspect='auto')
    ax.set_title('beta (orig) K/dB')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
else:
    ax.text(0.5,0.5,'no orig', ha='center')

ax = axes[0,1]
if recalc is not None:
    im = ax.imshow(recalc['beta'], cmap='RdYlBu', aspect='auto')
    ax.set_title('beta (recalc) K/dB')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
else:
    ax.text(0.5,0.5,'no recalc', ha='center')

ax = axes[0,2]
if orig is not None and recalc is not None:
    diff = recalc['beta'] - orig['beta']
    im = ax.imshow(diff, cmap='bwr', aspect='auto')
    ax.set_title('beta (recalc - orig)')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
else:
    ax.text(0.5,0.5,'no diff', ha='center')

# Row 1: gamma maps
ax = axes[1,0]
if orig is not None:
    im = ax.imshow(orig['gamma'], cmap='coolwarm', aspect='auto')
    ax.set_title('gamma (orig)')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
else:
    ax.text(0.5,0.5,'no orig', ha='center')

ax = axes[1,1]
if recalc is not None:
    im = ax.imshow(recalc['gamma'], cmap='coolwarm', aspect='auto')
    ax.set_title('gamma (recalc)')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
else:
    ax.text(0.5,0.5,'no recalc', ha='center')

ax = axes[1,2]
if orig is not None and recalc is not None:
    diffg = recalc['gamma'] - orig['gamma']
    im = ax.imshow(diffg, cmap='bwr', aspect='auto')
    ax.set_title('gamma (recalc - orig)')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
else:
    ax.text(0.5,0.5,'no diff', ha='center')

for ax in axes.ravel():
    ax.set_xlabel('X (pixel)')
    ax.set_ylabel('Y (pixel)')

plt.tight_layout()
outfile = OUT_DIR / 'step2_beta_gamma_compare.png'
plt.savefig(outfile, dpi=200, bbox_inches='tight')
plt.close()
print('Saved:', outfile)

# Print basic stats
def stats(name, arr):
    print(f"{name}: shape={arr.shape} min={np.nanmin(arr):.4f} max={np.nanmax(arr):.4f} mean={np.nanmean(arr):.4f} std={np.nanstd(arr):.4f}")

if orig is not None:
    stats('orig beta', orig['beta'])
    stats('orig gamma', orig['gamma'])
    if orig['n_samples'] is not None:
        print('orig n_samples min/max:', np.nanmin(orig['n_samples']), np.nanmax(orig['n_samples']))

if recalc is not None:
    stats('recalc beta', recalc['beta'])
    stats('recalc gamma', recalc['gamma'])
    if recalc['n_samples'] is not None:
        print('recalc n_samples min/max:', np.nanmin(recalc['n_samples']), np.nanmax(recalc['n_samples']))

if orig is not None and recalc is not None:
    print('beta diff stats:')
    stats('beta diff', recalc['beta'] - orig['beta'])
    print('gamma diff stats:')
    stats('gamma diff', recalc['gamma'] - orig['gamma'])

print('Done')
