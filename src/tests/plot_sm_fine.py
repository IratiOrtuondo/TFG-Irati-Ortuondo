#!/usr/bin/env python3
"""
Plot soil moisture from step6 output.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def plot_sm_fine(sm_file, out_dir=None):
    """
    Plot SM_fine from step6 tau-omega retrieval.
    
    Parameters
    ----------
    sm_file : str
        Path to SM_fine_*.npz file
    out_dir : str, optional
        Output directory for PNG
    """
    data = np.load(sm_file)
    
    sm = data['soil_moisture']
    date = str(data['date'])
    pol = str(data['pol'])
    
    # Load other retrieved variables
    tb = data['TB']
    emissivity = data['emissivity']
    reflectivity = data['reflectivity']
    epsilon = data['epsilon']
    
    print(f"[INFO] Loaded SM for {date} {pol}")
    print(f"[STAT] SM shape: {sm.shape}")
    print(f"       SM valid pixels: {np.sum(np.isfinite(sm))}")
    
    if np.sum(np.isfinite(sm)) > 0:
        valid = sm[np.isfinite(sm)]
        print(f"       SM range: [{np.min(valid):.4f}, {np.max(valid):.4f}] m³/m³")
        print(f"       SM mean: {np.mean(valid):.4f} m³/m³")
        print(f"       SM std: {np.std(valid):.4f} m³/m³")
    
    # Create figure with 6 subplots
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # 1. Soil Moisture
    im0 = axes[0, 0].imshow(sm, cmap='YlGnBu', interpolation='nearest', 
                             vmin=0, vmax=0.5)
    axes[0, 0].set_title(f'Soil Moisture\n{date} {pol}')
    axes[0, 0].set_xlabel('Column')
    axes[0, 0].set_ylabel('Row')
    cbar0 = plt.colorbar(im0, ax=axes[0, 0], label='m³/m³')
    
    # 2. TB
    im1 = axes[0, 1].imshow(tb, cmap='RdYlBu_r', interpolation='nearest',
                             vmin=200, vmax=300)
    axes[0, 1].set_title(f'TB (Input)\n{date} {pol}')
    axes[0, 1].set_xlabel('Column')
    axes[0, 1].set_ylabel('Row')
    plt.colorbar(im1, ax=axes[0, 1], label='K')
    
    # 3. Emissivity
    im2 = axes[0, 2].imshow(emissivity, cmap='viridis', interpolation='nearest',
                             vmin=0.5, vmax=1.0)
    axes[0, 2].set_title(f'Emissivity\nRange: [{np.nanmin(emissivity):.3f}, {np.nanmax(emissivity):.3f}]')
    axes[0, 2].set_xlabel('Column')
    axes[0, 2].set_ylabel('Row')
    plt.colorbar(im2, ax=axes[0, 2], label='Dimensionless')
    
    # 4. Reflectivity
    im3 = axes[1, 0].imshow(reflectivity, cmap='plasma', interpolation='nearest',
                             vmin=0, vmax=0.5)
    axes[1, 0].set_title(f'Reflectivity\nRange: [{np.nanmin(reflectivity):.3f}, {np.nanmax(reflectivity):.3f}]')
    axes[1, 0].set_xlabel('Column')
    axes[1, 0].set_ylabel('Row')
    plt.colorbar(im3, ax=axes[1, 0], label='Dimensionless')
    
    # 5. Dielectric constant
    im4 = axes[1, 1].imshow(epsilon, cmap='hot_r', interpolation='nearest',
                             vmin=1, vmax=30)
    axes[1, 1].set_title(f'Dielectric Constant ε\nRange: [{np.nanmin(epsilon):.1f}, {np.nanmax(epsilon):.1f}]')
    axes[1, 1].set_xlabel('Column')
    axes[1, 1].set_ylabel('Row')
    plt.colorbar(im4, ax=axes[1, 1], label='Dimensionless')
    
    # 6. Histogram of SM
    axes[1, 2].hist(sm[np.isfinite(sm)], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[1, 2].set_title(f'SM Distribution\nn = {np.sum(np.isfinite(sm))} pixels')
    axes[1, 2].set_xlabel('Soil Moisture (m³/m³)')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].axvline(np.nanmean(sm), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.nanmean(sm):.3f}')
    axes[1, 2].legend()
    
    plt.suptitle(f'Step 6: Soil Moisture Retrieval (Tau-Omega) - {date} {pol}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if out_dir:
        out_path = Path(out_dir) / f'SM_fine_{date}_{pol}.png'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Saved plot to {out_path}")
    
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot soil moisture retrieval results')
    parser.add_argument('--data-file', type=str,
                        help='Path to SM_fine NPZ file')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                        help='Directory with SM_fine files')
    parser.add_argument('--dates', nargs='+',
                        help='Dates to plot (YYYYMMDD)')
    parser.add_argument('--pol', type=str, default='VV',
                        help='Polarization')
    parser.add_argument('--out-dir', type=str, default='data/processed/plots',
                        help='Output directory for plots')
    
    args = parser.parse_args()
    
    if args.data_file:
        # Plot single file
        plot_sm_fine(args.data_file, args.out_dir)
    elif args.dates:
        # Plot multiple dates
        for date in args.dates:
            sm_file = Path(args.data_dir) / f'SM_fine_{date}_{args.pol}_tauomega.npz'
            if sm_file.exists():
                print(f"\n{'='*60}")
                print(f"Processing {date}")
                print(f"{'='*60}")
                plot_sm_fine(str(sm_file), args.out_dir)
            else:
                print(f"[SKIP] {date}: File not found: {sm_file}")
    else:
        print("Error: Must specify --data-file or --dates")
