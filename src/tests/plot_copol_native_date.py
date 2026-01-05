#!/usr/bin/env python3
"""plot_copol_native_date.py

Plot native-resolution co-pol (VV and HH) for a specific date.

Usage:
  python plot_copol_native_date.py --date 20150502 --data-dir data/interim
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Plot native co-pol VV and HH for a date")
    parser.add_argument("--date", required=True, help="Date in YYYYMMDD format")
    parser.add_argument("--data-dir", default="data/interim", help="Directory with NPZ files")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    
    vv_path = data_dir / f"aligned-smap-copol-{args.date}-vv-native.npz"
    hh_path = data_dir / f"aligned-smap-copol-{args.date}-hh-native.npz"
    
    # Load both if available
    vv_data = None
    hh_data = None
    
    if vv_path.exists():
        vv_npz = np.load(vv_path)
        vv_data = vv_npz["S_copol_dB_native"]
    
    if hh_path.exists():
        hh_npz = np.load(hh_path)
        hh_data = hh_npz["S_copol_dB_native"]
    
    # Create figure
    n_plots = (1 if vv_data is not None else 0) + (1 if hh_data is not None else 0)
    
    if n_plots == 0:
        print(f"No valid data found for {args.date}")
        return
    
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    
    idx = 0
    
    if vv_data is not None:
        ax = axes[idx]
        finite_count = np.isfinite(vv_data).sum()
        im = ax.imshow(vv_data, cmap="viridis", interpolation="nearest")
        ax.set_title(f"SMAP L3 Co-pol VV Native\n{args.date}\nShape: {vv_data.shape} | Finite: {finite_count:,}")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        plt.colorbar(im, ax=ax, label="σ⁰ VV (dB)")
        idx += 1
    
    if hh_data is not None:
        ax = axes[idx]
        finite_count = np.isfinite(hh_data).sum()
        im = ax.imshow(hh_data, cmap="viridis", interpolation="nearest")
        ax.set_title(f"SMAP L3 Co-pol HH Native\n{args.date}\nShape: {hh_data.shape} | Finite: {finite_count:,}")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        plt.colorbar(im, ax=ax, label="σ⁰ HH (dB)")
    
    plt.tight_layout()
    
    out_path = data_dir / f"copol_native_{args.date}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {out_path}")
    plt.show()

if __name__ == "__main__":
    main()
