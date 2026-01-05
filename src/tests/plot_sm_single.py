#!/usr/bin/env python3
"""Plot only soil moisture with statistics.

Usage:
  python plot_sm_single.py --npz ../data/processed/SM_fine_20150607_VV_tauomega.npz --out-dir ../data/interim/plots
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description="Plot soil moisture with statistics")
    p.add_argument("--npz", required=True, help="Path to SM NPZ")
    p.add_argument("--out-dir", default="../data/interim/plots", help="Output directory")
    p.add_argument("--crop-y", type=int, default=28, help="Max row index")
    p.add_argument("--crop-x", type=int, default=35, help="Max column index")
    args = p.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    SM = d.get("soil_moisture")
    
    if SM is None:
        for k in ("sm", "SM"):
            if k in d:
                SM = d[k]
                break
    
    if SM is None:
        raise KeyError("soil_moisture not found in NPZ")
    
    # Crop
    SM = SM[:args.crop_y, :args.crop_x]
    
    # Statistics
    valid = SM[np.isfinite(SM)]
    if valid.size > 0:
        sm_min = float(valid.min())
        sm_max = float(valid.max())
        sm_mean = float(valid.mean())
    else:
        sm_min = sm_max = sm_mean = np.nan
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap("plasma").copy()
    cmap.set_bad("white")
    
    im = ax.imshow(SM, origin="upper", cmap=cmap, vmin=0.0, vmax=0.6)
    
    date = str(d.get("date", "unknown"))
    pol = str(d.get("pol", "unknown"))
    
    ax.set_title(f"Soil Moisture - {date} - {pol}\n"
                 f"Min: {sm_min:.3f} | Mean: {sm_mean:.3f} | Max: {sm_max:.3f}",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    
    cbar = plt.colorbar(im, ax=ax, label="Soil Moisture [m³/m³]")
    
    plt.tight_layout()
    
    os.makedirs(args.out_dir, exist_ok=True)
    base = Path(args.npz).stem
    outpng = Path(args.out_dir) / f"{base}_SM_only.png"
    fig.savefig(outpng, dpi=200, bbox_inches="tight")
    print(f"[OK] Saved: {outpng}")
    print(f"     Min: {sm_min:.3f} | Mean: {sm_mean:.3f} | Max: {sm_max:.3f}")


if __name__ == "__main__":
    main()
