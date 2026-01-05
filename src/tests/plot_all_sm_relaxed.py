#!/usr/bin/env python3
"""
Plot all SM_fine relaxed maps in one figure and compute statistics.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math
import csv

DATA_DIR = Path(r"c:\Users\ortuo\tfgirati\tfg-nisar\data\processed")
PLOTS_DIR = DATA_DIR / "plots"
SUMMARY_DIR = DATA_DIR / "summaries"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

# Crop consistent with other scripts
CROP_Y = 27
CROP_X = 35

# Geographic bounds for extent
LON_MIN = -104.8884912
LAT_MIN = 39.8008444
LON_MAX = -103.7115088
LAT_MAX = 40.6991556

files = sorted(DATA_DIR.glob('SM_fine_*_relaxed.npz'))
if len(files) == 0:
    print('No relaxed SM files found in', DATA_DIR)
    raise SystemExit(1)

dates = []
maps = []
stats_rows = []

for p in files:
    d = np.load(p)
    if 'soil_moisture' not in d:
        continue
    sm = d['soil_moisture']
    # crop
    smc = sm[:CROP_Y, :CROP_X]
    date = p.name.split('_')[2]
    dates.append(date)
    maps.append(smc)
    valid = smc[np.isfinite(smc) & (smc > 0) & (smc < 1)]
    stats_rows.append({
        'date': date,
        'n_pixels': int(np.sum(np.isfinite(smc))),
        'mean': float(np.nanmean(valid)) if valid.size>0 else np.nan,
        'std': float(np.nanstd(valid)) if valid.size>0 else np.nan,
        'min': float(np.nanmin(valid)) if valid.size>0 else np.nan,
        'max': float(np.nanmax(valid)) if valid.size>0 else np.nan,
    })

# Stack maps -> (N, Y, X)
stack = np.stack(maps, axis=0)

# Per-pixel mean/std
mean_map = np.nanmean(stack, axis=0)
std_map = np.nanstd(stack, axis=0)

# Global statistics across dates (mean of per-date means)
date_means = [r['mean'] for r in stats_rows if not np.isnan(r['mean'])]
global_stats = {
    'n_dates': len(dates),
    'mean_of_means': float(np.nanmean(date_means)) if len(date_means)>0 else np.nan,
    'std_of_means': float(np.nanstd(date_means)) if len(date_means)>0 else np.nan,
}

# Save CSV summary
csv_path = SUMMARY_DIR / 'sm_relaxed_per_date_stats.csv'
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['date','n_pixels','mean','std','min','max'])
    w.writeheader()
    for r in stats_rows:
        w.writerow(r)

# Save mean/std maps
np.savez_compressed(SUMMARY_DIR / 'sm_relaxed_all_dates_maps.npz', mean_map=mean_map, std_map=std_map, dates=np.array(dates))

# Plot all maps in a grid
n = len(maps)
cols = min(5, n)
rows = math.ceil(n/cols)
fig, axs = plt.subplots(rows, cols, figsize=(4*cols, 3*rows), squeeze=False)
vmin = 0.0
vmax = 0.5
for idx, (date, smc) in enumerate(zip(dates, maps)):
    r = idx // cols
    c = idx % cols
    ax = axs[r][c]
    im = ax.imshow(smc, cmap='YlGnBu', vmin=vmin, vmax=vmax, extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], origin='upper')
    ax.set_title(date)
    ax.set_xticks([])
    ax.set_yticks([])

# turn off empty axes
for idx in range(n, rows*cols):
    r = idx // cols
    c = idx % cols
    axs[r][c].axis('off')

fig.suptitle('sm disaggregated all dates', fontsize=16, fontweight='bold')
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
fig.colorbar(im, cax=cbar_ax, label='SM (m3/m3)')

out_png = PLOTS_DIR / 'SM_disaggregated_all_dates.png'
plt.savefig(out_png, dpi=300, bbox_inches='tight')
plt.close()

print('Plot saved:', out_png)
print('Per-date CSV saved:', csv_path)
print('Mean/std maps saved:', SUMMARY_DIR / 'sm_relaxed_all_dates_maps.npz')
print('Global stats:', global_stats)
