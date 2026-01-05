#!/usr/bin/env python3
"""plot_soil_moisture_disagg.py

Plot soil moisture from radar-based disaggregation.

Usage:
  python plot_soil_moisture_disagg.py --date 20150607 --pol VV --data-dir data/processed
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from affine import Affine
from rasterio.crs import CRS


def main():
    parser = argparse.ArgumentParser(description="Plot soil moisture disaggregation")
    parser.add_argument("--date", type=str, required=True, help="Date YYYYMMDD")
    parser.add_argument("--pol", type=str, required=True, choices=["VV", "HH"], help="Radar polarization")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory with soil moisture NPZ files")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for PNG (optional)")
    
    args = parser.parse_args()
    
    # Load soil moisture NPZ
    npz_path = args.data_dir / f"soil_moisture_{args.date}_{args.pol}_disagg.npz"
    
    if not npz_path.exists():
        raise FileNotFoundError(f"Soil moisture NPZ not found: {npz_path}")
    
    print(f"Loading {npz_path.name}")
    data = np.load(npz_path)
    
    sm = data['soil_moisture']
    sm_36km = data['SM_36km']
    delta_sigma = data['delta_sigma_copol']
    gamma = data['gamma']
    
    tf_arr = data['transform']
    tf = Affine(tf_arr[0], tf_arr[1], tf_arr[2], tf_arr[3], tf_arr[4], tf_arr[5])
    crs = CRS.from_wkt(str(data['crs_wkt']))
    
    print(f"  Shape: {sm.shape}")
    print(f"  Finite pixels: {np.isfinite(sm).sum()}")
    print(f"  SM range: [{np.nanmin(sm):.3f}, {np.nanmax(sm):.3f}] m³/m³")
    print(f"  SM mean: {np.nanmean(sm):.3f} m³/m³")
    print(f"  SM std: {np.nanstd(sm):.3f} m³/m³")
    print(f"  CRS: {crs.to_string()}")
    
    # Create 3-panel plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Panel 1: Soil Moisture Fine
    ax1 = axes[0]
    im1 = ax1.imshow(sm, cmap='YlGnBu', vmin=0.0, vmax=0.5, interpolation='nearest')
    ax1.set_title(f'Soil Moisture (Disaggregated) - {args.date} - {args.pol}\n'
                  f'Mean: {np.nanmean(sm):.3f} m³/m³ | Std: {np.nanstd(sm):.3f}')
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')
    cbar1 = plt.colorbar(im1, ax=ax1, label='SM [m³/m³]')
    
    # Panel 2: Backscatter Anomaly
    ax2 = axes[1]
    vmax_sigma = max(abs(np.nanmin(delta_sigma)), abs(np.nanmax(delta_sigma)))
    im2 = ax2.imshow(delta_sigma, cmap='RdBu_r', vmin=-vmax_sigma, vmax=vmax_sigma, interpolation='nearest')
    ax2.set_title(f'Backscatter Anomaly (σ_fine - σ_36km)\n'
                  f'Range: [{np.nanmin(delta_sigma):.2f}, {np.nanmax(delta_sigma):.2f}] dB')
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Row')
    cbar2 = plt.colorbar(im2, ax=ax2, label='Δσ [dB]')
    
    # Panel 3: SM Coarse (36 km)
    ax3 = axes[2]
    im3 = ax3.imshow(sm_36km, cmap='YlGnBu', vmin=0.0, vmax=0.5, interpolation='nearest')
    ax3.set_title(f'Soil Moisture Coarse (36 km)\n'
                  f'Mean: {np.nanmean(sm_36km):.3f} m³/m³')
    ax3.set_xlabel('Column')
    ax3.set_ylabel('Row')
    cbar3 = plt.colorbar(im3, ax=ax3, label='SM [m³/m³]')
    
    plt.tight_layout()
    
    # Save or show
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.out_dir / f"soil-moisture-disagg-{args.date}-{args.pol.lower()}.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"\n[OK] Saved: {out_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    main()
