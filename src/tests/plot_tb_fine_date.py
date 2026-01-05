#!/usr/bin/env python3
"""plot_tb_fine_date.py

Plot TB fine (disaggregated) for a specific date.

Usage:
  python plot_tb_fine_date.py --date 20150607 --pol VV --data-dir data/processed
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from affine import Affine
from rasterio.crs import CRS


def main():
    parser = argparse.ArgumentParser(description="Plot TB fine (disaggregated)")
    parser.add_argument("--date", type=str, required=True, help="Date YYYYMMDD")
    parser.add_argument("--pol", type=str, required=True, choices=["VV", "HH"], help="Radar polarization")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory with TB fine NPZ files")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for PNG (optional)")
    
    args = parser.parse_args()
    
    # Load TB fine NPZ
    npz_path = args.data_dir / f"TB_fine_{args.date}_{args.pol}_native.npz"
    
    if not npz_path.exists():
        raise FileNotFoundError(f"TB fine NPZ not found: {npz_path}")
    
    print(f"Loading {npz_path.name}")
    data = np.load(npz_path)
    
    tb_fine = data['TB_fine']
    tf_arr = data['transform']
    tf = Affine(tf_arr[0], tf_arr[1], tf_arr[2], tf_arr[3], tf_arr[4], tf_arr[5])
    crs = CRS.from_wkt(str(data['crs_wkt']))
    
    print(f"  Shape: {tb_fine.shape}")
    print(f"  Finite pixels: {np.isfinite(tb_fine).sum()}")
    print(f"  TB range: [{np.nanmin(tb_fine):.2f}, {np.nanmax(tb_fine):.2f}] K")
    print(f"  CRS: {crs.to_string()}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(tb_fine, cmap='RdYlBu_r', vmin=240, vmax=280, interpolation='nearest')
    ax.set_title(f'TB Fine (Disaggregated) - {args.date} - {args.pol}\n'
                 f'Shape: {tb_fine.shape} | Finite: {np.isfinite(tb_fine).sum()} | '
                 f'Range: [{np.nanmin(tb_fine):.1f}, {np.nanmax(tb_fine):.1f}] K')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    
    cbar = plt.colorbar(im, ax=ax, label='TB [K]')
    
    plt.tight_layout()
    
    # Save or show
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.out_dir / f"tb-fine-{args.date}-{args.pol.lower()}.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"\n[OK] Saved: {out_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    main()
