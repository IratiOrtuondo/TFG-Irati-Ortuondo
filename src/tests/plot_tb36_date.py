#!/usr/bin/env python3
"""plot_tb36_date.py

Plot SMAP TB 36km for a specific date.

Usage:
  python plot_tb36_date.py --date 20150607 --data-dir data/interim --pol V
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from affine import Affine
from rasterio.crs import CRS


def main():
    parser = argparse.ArgumentParser(description="Plot SMAP TB 36km")
    parser.add_argument("--date", type=str, required=True, help="Date YYYYMMDD")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory with NPZ files")
    parser.add_argument("--pol", type=str, required=True, choices=["V", "H"], help="Polarization")
    parser.add_argument("--disagg", action="store_true", help="Plot disaggregated TB (coarse and fine)")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for PNG (optional)")
    
    args = parser.parse_args()
    
    if args.disagg:
        pol_map = {'V': 'VV', 'H': 'HH'}

        # coarse from data/interim smap-tb36-... file
        coarse_path = args.data_dir / f"smap-tb36-{args.date}-{args.pol.lower()}.npz"
        if not coarse_path.exists():
            raise FileNotFoundError(f"Coarse TB NPZ not found: {coarse_path}")
        print(f"Loading coarse: {coarse_path.name}")
        coarse_data = np.load(coarse_path)
        tb_coarse = coarse_data['TB_36km']

        # fine from data/processed TB_fine_... file
        processed_dir = args.data_dir.parent / 'processed'
        fine_name = f"TB_fine_{args.date}_{pol_map.get(args.pol, args.pol)}_native.npz"
        fine_path = processed_dir / fine_name
        if not fine_path.exists():
            # try alternative naming without underscores
            alt = processed_dir / f"TB_fine_{args.date}_{pol_map.get(args.pol, args.pol)}native.npz"
            if alt.exists():
                fine_path = alt
            else:
                raise FileNotFoundError(f"Fine TB NPZ not found: {fine_path}")

        print(f"Loading fine: {fine_path.name}")
        fine_data = np.load(fine_path)
        tb_fine = fine_data['TB_fine']

        print(f"  Coarse shape: {tb_coarse.shape}")
        print(f"  Fine shape: {tb_fine.shape}")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # shared vmin/vmax from combined data for comparable colors
        all_min = min(np.nanmin(tb_coarse), np.nanmin(tb_fine))
        all_max = max(np.nanmax(tb_coarse), np.nanmax(tb_fine))
        rng = all_max - all_min
        pad = max(1.0, 0.02 * rng)
        vmin = all_min - pad
        vmax = all_max + pad

        for ax, arr, title in ((axes[0], tb_coarse, 'Brightness temperature coarse'), (axes[1], tb_fine, 'Brightness temperature fine')):
            im = ax.imshow(arr, cmap='RdYlBu_r', vmin=vmin, vmax=vmax, interpolation='nearest')
            ax.set_title(title, fontsize=20)
            ax.set_xticks([])
            ax.set_ylabel('Row')

            # If this is the fine plot, crop to x<=35 and y<=27
            if 'fine' in title.lower():
                ax.set_xlim(0, 35)
                # imshow uses origin='upper' by default; set_ylim(27, 0) to show rows 0..27
                ax.set_ylim(27, 0)

            cbar = plt.colorbar(im, ax=ax, label='TB [K]')
            cbar.ax.tick_params(labelsize=16)
            cbar.set_label('TB [K]', fontsize=16)

        plt.tight_layout()
    else:
        # Load TB NPZ (coarse 36 km)
        npz_path = args.data_dir / f"smap-tb36-{args.date}-{args.pol.lower()}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"TB NPZ not found: {npz_path}")
        print(f"Loading {npz_path.name}")
        data = np.load(npz_path)

        tb = data['TB_36km']
        tf_arr = data['transform']
        tf = Affine(tf_arr[0], tf_arr[1], tf_arr[2], tf_arr[3], tf_arr[4], tf_arr[5])
        crs = CRS.from_wkt(str(data['crs_wkt']))

        print(f"  Shape: {tb.shape}")
        print(f"  Finite pixels: {np.isfinite(tb).sum()}")
        print(f"  TB range: [{np.nanmin(tb):.1f}, {np.nanmax(tb):.1f}] K")
        print(f"  CRS: {crs.to_string()}")

        fig, ax = plt.subplots(figsize=(8, 6))

        tb_min = np.nanmin(tb)
        tb_max = np.nanmax(tb)
        rng = tb_max - tb_min
        pad = max(1.0, 0.02 * rng)
        vmin = tb_min - pad
        vmax = tb_max + pad

        im = ax.imshow(tb, cmap='RdYlBu_r', vmin=vmin, vmax=vmax, interpolation='nearest')
        ax.set_title('Brightness temperature', fontsize=24)
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.set_xticks([])

        cbar = plt.colorbar(im, ax=ax, label='TB [K]')
        cbar.ax.tick_params(labelsize=16)
        cbar.set_label('TB [K]', fontsize=16)

        plt.tight_layout()
    
    # Save or show
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        if args.disagg:
            out_path = args.out_dir / f"tb36-{args.date}-{args.pol.lower()}-disagg.png"
        else:
            out_path = args.out_dir / f"tb36-{args.date}-{args.pol.lower()}.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"\n[OK] Saved: {out_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    main()
