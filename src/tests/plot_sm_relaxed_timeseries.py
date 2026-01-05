#!/usr/bin/env python3
"""
Plot time series of per-date statistics for SM_fine relaxed outputs.
Reads: data/processed/summaries/sm_relaxed_per_date_stats.csv
Saves: data/processed/plots/SM_relaxed_timeseries.png
"""
import csv
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = Path(r"c:\Users\ortuo\tfgirati\tfg-nisar\data\processed")
SUM_DIR = DATA_DIR / "summaries"
PLOT_DIR = DATA_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

csv_path = SUM_DIR / 'sm_relaxed_per_date_stats.csv'
if not csv_path.exists():
    print('CSV not found:', csv_path)
    raise SystemExit(1)

dates = []
means = []
stds = []
mins = []
maxs = []

with open(csv_path, 'r', newline='') as f:
    r = csv.DictReader(f)
    for row in r:
        # parse date like 20150607
        try:
            dt = datetime.strptime(row['date'], '%Y%m%d')
        except Exception:
            dt = datetime.strptime(row['date'], '%Y-%m-%d') if '-' in row['date'] else None
        dates.append(dt)
        means.append(float(row['mean']) if row['mean']!='' else np.nan)
        stds.append(float(row['std']) if row['std']!='' else np.nan)
        mins.append(float(row['min']) if row['min']!='' else np.nan)
        maxs.append(float(row['max']) if row['max']!='' else np.nan)

# Sort by date in case
order = np.argsort(dates)
dates = np.array(dates)[order]
means = np.array(means)[order]
stds = np.array(stds)[order]
mins = np.array(mins)[order]
maxs = np.array(maxs)[order]

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(dates, means, '-o', label='Mean SM (per date)', color='C0')
ax.fill_between(dates, means-stds, means+stds, color='C0', alpha=0.2, label='±1 std')
ax.plot(dates, mins, '--', color='gray', label='Min (valid)')
ax.plot(dates, maxs, '--', color='black', label='Max (valid)')

ax.set_title('SM disaggregated - temporal evolution (relaxed)')
ax.set_ylabel('Soil moisture (m³/m³)')
ax.set_ylim(0, 0.6)
ax.grid(True, alpha=0.3)
ax.legend()

out = PLOT_DIR / 'SM_relaxed_timeseries.png'
plt.tight_layout()
plt.savefig(out, dpi=300)
plt.close()

print('Saved timeseries plot:', out)
