#!/usr/bin/env python3
"""
Plot disaggregated TB from step5 output.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def plot_tb_fine(tb_file, out_dir=None):
    """
    Plot TB_fine from step5 disaggregation output.
    
    Parameters
    ----------
    tb_file : str
        Path to TB_fine_*.npz file
    out_dir : str, optional
        Output directory for PNG
    """
    data = np.load(tb_file)
    
    tb_fine = data['TB_fine']
    date = str(data['date'])
    pol = str(data['pol'])
    
    # Load components for analysis
    tb36_native = data['TB36_native']
    beta = data['beta_native']
    gamma = data['gamma_native']
    delta_copol = data['delta_copol']
    delta_xpol = data['delta_xpol']
    
    print(f"[INFO] Loaded TB_fine for {date} {pol}")
    print(f"[STAT] TB_fine shape: {tb_fine.shape}")
    print(f"       TB_fine valid pixels: {np.sum(np.isfinite(tb_fine))}")
    
    if np.sum(np.isfinite(tb_fine)) > 0:
        valid = tb_fine[np.isfinite(tb_fine)]
        print(f"       TB_fine range: [{np.min(valid):.2f}, {np.max(valid):.2f}] K")
        print(f"       TB_fine mean: {np.mean(valid):.2f} K")
        print(f"       TB_fine std: {np.std(valid):.2f} K")
    
    # Create figure with 6 subplots
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # 1. TB_fine (disaggregated)
    im0 = axes[0, 0].imshow(tb_fine, cmap='RdYlBu_r', interpolation='nearest', 
                             vmin=200, vmax=300)
    axes[0, 0].set_title(f'TB Fine (Disaggregated)\n{date} {pol}')
    axes[0, 0].set_xlabel('Column')
    axes[0, 0].set_ylabel('Row')
    plt.colorbar(im0, ax=axes[0, 0], label='K')
    
    # 2. TB36 (coarse, upsampled)
    im1 = axes[0, 1].imshow(tb36_native, cmap='RdYlBu_r', interpolation='nearest',
                             vmin=200, vmax=300)
    axes[0, 1].set_title(f'TB 36km (Upsampled)\n{date} {pol}')
    axes[0, 1].set_xlabel('Column')
    axes[0, 1].set_ylabel('Row')
    plt.colorbar(im1, ax=axes[0, 1], label='K')
    
    # 3. TB difference (fine - coarse)
    tb_diff = tb_fine - tb36_native
    vmax_diff = max(abs(np.nanpercentile(tb_diff, 5)), abs(np.nanpercentile(tb_diff, 95)))
    im2 = axes[0, 2].imshow(tb_diff, cmap='RdBu_r', interpolation='nearest',
                             vmin=-vmax_diff, vmax=vmax_diff)
    axes[0, 2].set_title(f'TB Difference (Fine - Coarse)\nRange: [{np.nanmin(tb_diff):.1f}, {np.nanmax(tb_diff):.1f}] K')
    axes[0, 2].set_xlabel('Column')
    axes[0, 2].set_ylabel('Row')
    plt.colorbar(im2, ax=axes[0, 2], label='K')
    
    # 4. Beta (K/dB)
    im3 = axes[1, 0].imshow(beta, cmap='RdBu_r', interpolation='nearest')
    axes[1, 0].set_title(f'β (K/dB)\nRange: [{np.nanmin(beta):.2f}, {np.nanmax(beta):.2f}]')
    axes[1, 0].set_xlabel('Column')
    axes[1, 0].set_ylabel('Row')
    plt.colorbar(im3, ax=axes[1, 0], label='K/dB')
    
    # 5. Delta co-pol
    vmax_dcopol = max(abs(np.nanpercentile(delta_copol, 5)), abs(np.nanpercentile(delta_copol, 95)))
    im4 = axes[1, 1].imshow(delta_copol, cmap='RdBu_r', interpolation='nearest',
                             vmin=-vmax_dcopol, vmax=vmax_dcopol)
    axes[1, 1].set_title(f'Δσ_pp (dB)\nRange: [{np.nanmin(delta_copol):.2f}, {np.nanmax(delta_copol):.2f}]')
    axes[1, 1].set_xlabel('Column')
    axes[1, 1].set_ylabel('Row')
    plt.colorbar(im4, ax=axes[1, 1], label='dB')
    
    # 6. Delta cross-pol
    vmax_dxpol = max(abs(np.nanpercentile(delta_xpol, 5)), abs(np.nanpercentile(delta_xpol, 95)))
    im5 = axes[1, 2].imshow(delta_xpol, cmap='RdBu_r', interpolation='nearest',
                             vmin=-vmax_dxpol, vmax=vmax_dxpol)
    axes[1, 2].set_title(f'Δσ_xpol (dB)\nRange: [{np.nanmin(delta_xpol):.2f}, {np.nanmax(delta_xpol):.2f}]')
    axes[1, 2].set_xlabel('Column')
    axes[1, 2].set_ylabel('Row')
    plt.colorbar(im5, ax=axes[1, 2], label='dB')
    
    plt.suptitle(f'Step 5: TB Disaggregation Results - {date} {pol}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if out_dir:
        out_path = Path(out_dir) / f'TB_fine_{date}_{pol}.png'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Saved plot to {out_path}")
    
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot TB disaggregation results')
    parser.add_argument('--data-file', type=str,
                        help='Path to TB_fine NPZ file')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                        help='Directory with TB_fine files')
    parser.add_argument('--dates', nargs='+',
                        help='Dates to plot (YYYYMMDD)')
    parser.add_argument('--pol', type=str, default='VV',
                        help='Polarization')
    parser.add_argument('--out-dir', type=str, default='data/processed/plots',
                        help='Output directory for plots')
    
    args = parser.parse_args()
    
    if args.data_file:
        # Plot single file
        plot_tb_fine(args.data_file, args.out_dir)
    elif args.dates:
        # Plot multiple dates
        for date in args.dates:
            tb_file = Path(args.data_dir) / f'TB_fine_{date}_{args.pol}_native.npz'
            if tb_file.exists():
                print(f"\n{'='*60}")
                print(f"Processing {date}")
                print(f"{'='*60}")
                plot_tb_fine(str(tb_file), args.out_dir)
            else:
                print(f"[SKIP] {date}: File not found: {tb_file}")
    else:
        print("Error: Must specify --data-file or --dates")
