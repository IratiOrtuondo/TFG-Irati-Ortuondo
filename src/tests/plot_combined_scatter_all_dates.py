#!/usr/bin/env python3
"""Create a combined scatter (panel 3 style) with points from all dates (exclude 20150615).
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import h5py
from scipy.stats import pearsonr
import os


RAW_DIR = Path(r"c:\Users\ortuo\tfgirati\tfg-nisar\data\raw")
PROCESSED_DIR = Path(r"c:\Users\ortuo\tfgirati\tfg-nisar\data\processed")
OUTPUT_DIR = PROCESSED_DIR / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATES = ['20150607', '20150610', '20150615', '20150618', '20150620']
EXCLUDE = {'20150615'}

# same region bounds as original script
LON_MIN = -104.8884912
LAT_MIN = 39.8008444
LON_MAX = -103.7115088
LAT_MAX = 40.6991556


def extract_smap_sm_3x3(date):
    smap_file_p = RAW_DIR / f"SMAP_L3_SM_P_{date}_R19240_001.h5"
    smap_file_a = RAW_DIR / f"SMAP_L3_SM_A_{date}_R13080_001.h5"
    if smap_file_p.exists():
        smap_file = smap_file_p
    elif smap_file_a.exists():
        smap_file = smap_file_a
    else:
        return None

    with h5py.File(smap_file, 'r') as f:
        if 'Soil_Moisture_Retrieval_Data_AM' in f:
            group = f['Soil_Moisture_Retrieval_Data_AM']
        elif 'Soil_Moisture_Retrieval_Data_PM' in f:
            group = f['Soil_Moisture_Retrieval_Data_PM']
        else:
            return None

        sm = group['soil_moisture'][:]
        lat = group['latitude'][:]
        lon = group['longitude'][:]
        sm = np.where(sm == -9999.0, np.nan, sm)

        mask = ((lat >= LAT_MIN) & (lat <= LAT_MAX) & (lon >= LON_MIN) & (lon <= LON_MAX))
        if not np.any(mask):
            return None
        rows, cols = np.where(mask)
        r0, r1 = rows.min(), rows.max() + 1
        c0, c1 = cols.min(), cols.max() + 1
        return sm[r0:r1, c0:c1]


def aggregate_to_coarse(fine_data, coarse_shape):
    native_shape = fine_data.shape
    block_y = native_shape[0] // coarse_shape[0]
    block_x = native_shape[1] // coarse_shape[1]
    H_trim = block_y * coarse_shape[0]
    W_trim = block_x * coarse_shape[1]
    fine_trim = fine_data[:H_trim, :W_trim]
    reshaped = fine_trim.reshape(coarse_shape[0], block_y, coarse_shape[1], block_x)
    agg = np.nanmean(np.nanmean(reshaped, axis=3), axis=1)
    return agg


def load_sm_fine(date):
    # try the common filenames used in repo
    candidates = [
        PROCESSED_DIR / f"SM_fine_{date}_TBV_tauomega_ATBD_reg_relaxed.npz",
        PROCESSED_DIR / f"SM_fine_{date}_TBV_tauomega_ATBD_reg.npz",
        PROCESSED_DIR / f"SM_fine_{date}_relaxed.npz",
    ]
    for p in candidates:
        if p.exists():
            try:
                data = np.load(p)
                # try common keys
                for k in ('soil_moisture', 'SM_fine', 'sm'):
                    if k in data:
                        return np.asarray(data[k])
                # fallback first ndarray
                for k in data.keys():
                    if isinstance(data[k], np.ndarray):
                        return np.asarray(data[k])
            except Exception:
                continue
    return None


def main():
    xs = []
    ys = []
    for date in DATES:
        if date in EXCLUDE:
            print(f"Skipping excluded date {date}")
            continue

        print(f"Loading date {date}")
        coarse = extract_smap_sm_3x3(date)
        if coarse is None:
            print(f"  coarse missing for {date}, skipping")
            continue

        fine = load_sm_fine(date)
        if fine is None:
            print(f"  fine missing for {date}, skipping")
            continue

        # crop fine to canonical size used elsewhere if needed
        fine = np.asarray(fine)
        coarse = np.asarray(coarse)
        if fine.ndim > 2:
            fine = fine.squeeze()
        if coarse.ndim > 2:
            coarse = coarse.squeeze()

        if fine.shape != coarse.shape:
            try:
                fine_agg = aggregate_to_coarse(fine, coarse.shape)
            except Exception as e:
                print(f"  aggregation failed for {date}: {e}")
                continue
        else:
            fine_agg = fine

        mask = np.isfinite(coarse) & np.isfinite(fine_agg)
        if np.sum(mask) < 1:
            print(f"  no overlapping valid pixels for {date}")
            continue

        xs.append(coarse[mask].ravel())
        ys.append(fine_agg[mask].ravel())

    if len(xs) == 0:
        print('No data to plot')
        return

    x_all = np.concatenate(xs)
    y_all = np.concatenate(ys)

    if x_all.size > 1:
        r, _ = pearsonr(x_all, y_all)
    else:
        r = np.nan
    rmse = np.sqrt(np.nanmean((y_all - x_all) ** 2))
    bias = np.nanmean(y_all - x_all)
    n = x_all.size

    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(x_all, y_all, s=80, alpha=0.6, edgecolors='k')
    mn = min(np.nanmin(x_all), np.nanmin(y_all))
    mx = max(np.nanmax(x_all), np.nanmax(y_all))
    pad = (mx - mn) * 0.05 if mx > mn else 0.01
    ax.plot([mn-pad, mx+pad], [mn-pad, mx+pad], 'r--', linewidth=2)
    ax.set_xlabel('SM Coarse (m3/m3)')
    ax.set_ylabel('SM Fine aggregated (m3/m3)')
    ax.set_title('Combined pixel-wise comparison (all dates excl. 20150615)')
    ax.grid(True, linestyle=':', linewidth=0.6)

    stats_text = f'Correlation: {r:.3f}\nRMSE: {rmse:.4f}\nBias: {bias:.4f}\nN pixels: {n}'
    ax.text(0.03, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    outp = OUTPUT_DIR / 'SM_fine_vs_SMAP_coarse_combined_excl20150615.png'
    fig.savefig(outp, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved combined scatter: {outp}')


if __name__ == '__main__':
    main()
