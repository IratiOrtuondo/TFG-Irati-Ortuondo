#!/usr/bin/env python3
"""Diagnostics for combined scatter: per-date stats, pooled correlation, colored scatter."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
import h5py

ROOT = Path(r"c:\Users\ortuo\tfgirati\tfg-nisar")
RAW = ROOT / 'data' / 'raw'
PROC = ROOT / 'data' / 'processed'
OUT = PROC / 'plots'
OUT.mkdir(parents=True, exist_ok=True)

DATES = ['20150607', '20150610', '20150615', '20150618', '20150620']
EXCLUDE = {'20150615'}

LON_MIN = -104.8884912
LAT_MIN = 39.8008444
LON_MAX = -103.7115088
LAT_MAX = 40.6991556
# match cropping used elsewhere
CROP_Y = 27
CROP_X = 35


def extract_smap_sm_3x3(date):
    # Prefer raw HDF5 extraction (gives small 3x5 coarse in many cases)
    p1 = RAW / f"SMAP_L3_SM_P_{date}_R19240_001.h5"
    p2 = RAW / f"SMAP_L3_SM_A_{date}_R13080_001.h5"
    p = p1 if p1.exists() else (p2 if p2.exists() else None)
    if p is not None:
        with h5py.File(p, 'r') as f:
            grp = None
            if 'Soil_Moisture_Retrieval_Data_AM' in f:
                grp = f['Soil_Moisture_Retrieval_Data_AM']
            elif 'Soil_Moisture_Retrieval_Data_PM' in f:
                grp = f['Soil_Moisture_Retrieval_Data_PM']
            else:
                grp = None
            if grp is not None:
                sm = grp['soil_moisture'][:]
                lat = grp['latitude'][:]
                lon = grp['longitude'][:]
                sm = np.where(sm == -9999.0, np.nan, sm)
                mask = (lat >= LAT_MIN) & (lat <= LAT_MAX) & (lon >= LON_MIN) & (lon <= LON_MAX)
                if np.any(mask):
                    rows, cols = np.where(mask)
                    r0, r1 = rows.min(), rows.max()+1
                    c0, c1 = cols.min(), cols.max()+1
                    return sm[r0:r1, c0:c1]

    # fallback to processed NPZ coarse file if available
    proc_npz = PROC / 'smap_sm_coarse_{}.npz'.format(date)
    if proc_npz.exists():
        try:
            d = np.load(proc_npz)
            for k in ('sm_coarse','sm'):
                if k in d:
                    arr = np.asarray(d[k])
                    return arr.squeeze()
        except Exception:
            pass
    return None


def load_sm_fine(date):
    # Force use of the 'relaxed' outputs when available
    candidates = [
        PROC / f"SM_fine_{date}_TBV_tauomega_ATBD_reg_relaxed.npz",
        PROC / f"SM_fine_{date}_relaxed.npz",
        PROC / f"SM_fine_{date}_TBV_tauomega_ATBD_reg.npz",
    ]
    for p in candidates:
        if p.exists():
            d = np.load(p)
            for k in ('soil_moisture','SM_fine','sm'):
                if k in d:
                    return np.asarray(d[k])
            for k in d.keys():
                if isinstance(d[k], np.ndarray):
                    return np.asarray(d[k])
    return None


def aggregate_to_coarse(fine, coarse_shape):
    Hf, Wf = fine.shape
    Hc, Wc = coarse_shape
    br = Hf // Hc
    bc = Wf // Wc
    Ht = br * Hc
    Wt = bc * Wc
    fine_t = fine[:Ht, :Wt]
    resh = fine_t.reshape(Hc, br, Wc, bc)
    agg = np.nanmean(np.nanmean(resh, axis=3), axis=1)
    return agg


def main():
    per_stats = []
    xs = []
    ys = []
    labels = []
    for date in DATES:
        if date in EXCLUDE:
            continue
        coarse = extract_smap_sm_3x3(date)
        fine = load_sm_fine(date)
        if coarse is None or fine is None:
            print('missing', date); continue
        coarse = np.asarray(coarse)
        fine = np.asarray(fine)
        if fine.ndim>2: fine = fine.squeeze()
        if coarse.ndim>2: coarse = coarse.squeeze()
        # crop fine to canonical display area used in per-date plots
        if fine.shape[0] >= CROP_Y and fine.shape[1] >= CROP_X:
            fine = fine[:CROP_Y, :CROP_X]
        # aggregate fine to coarse grid using coarse.shape
        if fine.shape != coarse.shape:
            try:
                fine_agg = aggregate_to_coarse(fine, coarse.shape)
            except Exception as e:
                print('agg fail', date, e); continue
        else:
            fine_agg = fine
        mask = np.isfinite(coarse)&np.isfinite(fine_agg)
        n = np.sum(mask)
        if n<2:
            per_stats.append((date, np.nan, np.nan, np.nan, n, np.nan, np.nan))
            continue
        x = coarse[mask].ravel(); y = fine_agg[mask].ravel()
        r, p = pearsonr(x,y)
        rmse = np.sqrt(np.mean((y-x)**2))
        bias = np.mean(y-x)
        per_stats.append((date, r, rmse, bias, n, np.nanmean(x), np.nanmean(y)))
        xs.append(x); ys.append(y); labels.append(np.repeat(date, x.size))

    # pooled
    if len(xs)==0:
        print('no data'); return
    x_all = np.concatenate(xs); y_all = np.concatenate(ys); lab_all = np.concatenate(labels)
    r_pool, _ = pearsonr(x_all, y_all)
    rmse_pool = np.sqrt(np.mean((y_all-x_all)**2))
    bias_pool = np.mean(y_all-x_all)

    # print table
    print('\nPer-date statistics (excluded 20150615):')
    print(f"{'date':<10} {'R':>6} {'RMSE':>8} {'Bias':>8} {'N':>6} {'coarse_mean':>12} {'fine_mean':>12}")
    for t in per_stats:
        print(f"{t[0]:<10} {t[1]:6.3f} {t[2]:8.4f} {t[3]:8.4f} {t[4]:6d} {t[5]:12.4f} {t[6]:12.4f}")
    print('\nPooled: R={:.3f}, RMSE={:.4f}, Bias={:.4f}, N={}'.format(r_pool, rmse_pool, bias_pool, x_all.size))

    # colored scatter plot
    cmap = plt.get_cmap('tab10')
    fig, ax = plt.subplots(figsize=(7,7))
    unique_dates = sorted(list(set(lab_all)))
    for i, d in enumerate(unique_dates):
        m = (lab_all==d)
        ax.scatter(x_all[m], y_all[m], s=60, alpha=0.7, label=d, color=cmap(i%10), edgecolors='k')
    mn = min(np.nanmin(x_all), np.nanmin(y_all)); mx = max(np.nanmax(x_all), np.nanmax(y_all))
    pad = (mx-mn)*0.05 if mx>mn else 0.01
    ax.plot([mn-pad, mx+pad], [mn-pad, mx+pad], 'r--', linewidth=2)
    ax.set_xlabel('SM Coarse (m3/m3)')
    ax.set_ylabel('SM Fine aggregated (m3/m3)')
    ax.set_title('Combined scatter colored by date')
    ax.legend(title='date')
    stats_text = f'Pooled R={r_pool:.3f}\nPooled RMSE={rmse_pool:.4f}\nPooled Bias={bias_pool:.4f}\nN={x_all.size}'
    ax.text(0.03, 0.97, stats_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    out = OUT / 'combined_scatter_colored_by_date_excl20150615.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    np.savez_compressed(PROC / 'summaries' / 'diagnose_combined_scatter_excl20150615.npz', x=x_all, y=y_all, lab=lab_all)
    print('Saved colored scatter:', out)


if __name__ == '__main__':
    main()
