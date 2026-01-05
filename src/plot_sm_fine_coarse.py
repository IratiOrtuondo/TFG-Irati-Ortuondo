#!/usr/bin/env python3
"""
Compare SM_fine (native resolution) vs SM_coarse (aggregated to 36km)
for multiple dates.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
DATES = ['20150607', '20150610', '20150615', '20150618', '20150620']
DATA_DIR = Path(r"c:\Users\ortuo\tfgirati\tfg-nisar\data\processed")
OUTPUT_DIR = DATA_DIR / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Crop to remove edge artifacts
CROP_Y = 27
CROP_X = 35

# Geographic bounds for the region (used to set extent for plots)
LON_MIN = -104.8884912
LAT_MIN = 39.8008444
LON_MAX = -103.7115088
LAT_MAX = 40.6991556

def aggregate_to_coarse(fine_data, native_shape=(30, 39), coarse_shape=(3, 5)):
    """
    Aggregate fine resolution data to coarse resolution by averaging blocks.
    """
    # Calculate block sizes
    block_y = native_shape[0] // coarse_shape[0]
    block_x = native_shape[1] // coarse_shape[1]
    
    # Initialize coarse array
    coarse = np.zeros(coarse_shape)
    
    # Aggregate by blocks (mean of each block)
    for i in range(coarse_shape[0]):
        for j in range(coarse_shape[1]):
            y_start = i * block_y
            y_end = (i + 1) * block_y
            x_start = j * block_x
            x_end = (j + 1) * block_x
            
            block = fine_data[y_start:y_end, x_start:x_end]
            # Use nanmean to handle potential NaN values
            coarse[i, j] = np.nanmean(block)
    
    return coarse

print("="*70)
print("SM FINE vs COARSE COMPARISON - ALL DATES")
print("="*70)

for date in DATES:
    print(f"\n{'='*70}")
    print(f"Processing: {date}")
    print(f"{'='*70}")
    
    # Load SM_fine (prefer relaxed variant)
    sm_file_relaxed = DATA_DIR / f"SM_fine_{date}_TBV_tauomega_ATBD_reg_relaxed.npz"
    sm_file_orig = DATA_DIR / f"SM_fine_{date}_TBV_tauomega_ATBD_reg.npz"
    if sm_file_relaxed.exists():
        sm_file = sm_file_relaxed
    elif sm_file_orig.exists():
        sm_file = sm_file_orig
    else:
        print(f"[WARN] SM file not found: {sm_file_relaxed.name} or {sm_file_orig.name}")
        continue

    data = np.load(sm_file)
    sm_fine_full = data['soil_moisture']

    # Crop to remove edges
    sm_fine = sm_fine_full[:CROP_Y, :CROP_X]

    # Load SMAP coarse (pre-extracted NPZ if available)
    sm_coarse_file = DATA_DIR / f"smap_sm_coarse_{date}.npz"
    if sm_coarse_file.exists():
        d = np.load(sm_coarse_file)
        # expected key: 'sm_coarse' (reprojected to native grid or same native shape)
        sm_coarse_full = d['sm_coarse']
        # If the coarse array has the same native grid as fine, just crop to the same window
        if sm_coarse_full.shape == sm_fine_full.shape:
            sm_coarse = sm_coarse_full[:CROP_Y, :CROP_X]
            sm_fine = sm_fine_full[:CROP_Y, :CROP_X]
        else:
            # If shapes differ, try to regrid by cropping the coarse to the lon/lat box
            # Many pre-extracted coarse files are already reprojected to the native grid (30x39).
            # If coarse is small (e.g., 3x3), expand it by nearest-neighbor to native shape where possible.
            try:
                # If coarse has lat/lon arrays, try to locate the window corresponding to region.
                lat = d.get('latitude')
                lon = d.get('longitude')
                if lat is not None and lon is not None:
                    # find mask inside requested lon/lat box
                    mask = ((lat >= LAT_MIN) & (lat <= LAT_MAX) & (lon >= LON_MIN) & (lon <= LON_MAX))
                    if mask.shape == sm_coarse_full.shape and np.any(mask):
                        # extract bounding box
                        rows, cols = np.where(mask)
                        r0, r1 = rows.min(), rows.max() + 1
                        c0, c1 = cols.min(), cols.max() + 1
                        sm_coarse = sm_coarse_full[r0:r1, c0:c1]
                        # if necessary, pad or resize to match CROP_Y/CROP_X using simple cropping/padding
                        sm_coarse = sm_coarse[:CROP_Y, :CROP_X]
                    else:
                        # fallback to simple nearest expansion: tile the small coarse to match native crop size
                        small = sm_coarse_full
                        reps_y = max(1, CROP_Y // small.shape[0])
                        reps_x = max(1, CROP_X // small.shape[1])
                        tiled = np.tile(small, (reps_y + 1, reps_x + 1))
                        sm_coarse = tiled[:CROP_Y, :CROP_X]
                else:
                    # no lat/lon provided: tile small coarse
                    small = sm_coarse_full
                    reps_y = max(1, CROP_Y // small.shape[0])
                    reps_x = max(1, CROP_X // small.shape[1])
                    tiled = np.tile(small, (reps_y + 1, reps_x + 1))
                    sm_coarse = tiled[:CROP_Y, :CROP_X]
            except Exception:
                print(f"[WARN] Problem processing SMAP coarse for {date}, falling back to aggregated fine")
                sm_coarse = aggregate_to_coarse(sm_fine_full, native_shape=sm_fine_full.shape, coarse_shape=(3, 5))
    else:
        # fallback: aggregate fine to a coarse-like field (less preferred)
        print(f"[WARN] SMAP coarse NPZ not found for {date}, falling back to aggregated fine")
        sm_coarse = aggregate_to_coarse(sm_fine_full, native_shape=sm_fine_full.shape, coarse_shape=(3, 5))
    
    # Build 36km aggregated coarse grid from available data
    try:
        # If we loaded a reprojected coarse on native grid use it for aggregation
        if 'sm_coarse_full' in locals():
            sm_coarse_agg = aggregate_to_coarse(sm_coarse_full, native_shape=sm_coarse_full.shape, coarse_shape=(3, 5))
        else:
            sm_coarse_agg = aggregate_to_coarse(sm_fine_full, native_shape=sm_fine_full.shape, coarse_shape=(3, 5))
    except Exception:
        sm_coarse_agg = aggregate_to_coarse(sm_fine_full, native_shape=sm_fine_full.shape, coarse_shape=(3, 5))

    # Statistics (use cropped fine and aggregated coarse)
    sm_fine_valid = sm_fine[(sm_fine > 0) & (sm_fine < 1) & np.isfinite(sm_fine)]
    sm_coarse_valid = sm_coarse_agg[np.isfinite(sm_coarse_agg)]
    
    print(f"\nSM_fine (native ~3km):")
    print(f"  Shape: {sm_fine.shape}")
    print(f"  Mean: {np.nanmean(sm_fine_valid):.4f} m3/m3")
    print(f"  Std: {np.nanstd(sm_fine_valid):.4f} m3/m3")
    print(f"  Min: {np.nanmin(sm_fine_valid):.4f} m3/m3")
    print(f"  Max: {np.nanmax(sm_fine_valid):.4f} m3/m3")
    
    print(f"\nSM_coarse (aggregated ~36km):")
    print(f"  Shape: {sm_coarse_agg.shape}")
    print(f"  Mean: {np.nanmean(sm_coarse_valid):.4f} m3/m3")
    print(f"  Std: {np.nanstd(sm_coarse_valid):.4f} m3/m3")
    print(f"  Min: {np.nanmin(sm_coarse_valid):.4f} m3/m3")
    print(f"  Max: {np.nanmax(sm_coarse_valid):.4f} m3/m3")
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Soil Moisture Comparison - {date}', fontsize=16, fontweight='bold')
    
    # 1. SM_fine (native resolution) plotted with geographic extent
    ax = axes[0]
    im1 = ax.imshow(sm_fine, cmap='YlGnBu', vmin=0, vmax=0.5, extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], origin='upper')
    ax.set_title(f'SM Fine (~3km)\n{sm_fine.shape[0]}x{sm_fine.shape[1]} pixels', 
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    cbar1 = plt.colorbar(im1, ax=ax, label='SM (m3/m3)')
    
    stats_text = f'Mean: {np.nanmean(sm_fine_valid):.3f}\n'
    stats_text += f'Std: {np.nanstd(sm_fine_valid):.3f}\n'
    stats_text += f'Range: [{np.nanmin(sm_fine_valid):.3f}, {np.nanmax(sm_fine_valid):.3f}]'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    
    # 2. SM_coarse (36km aggregated) displayed with same geographic extent (blocky)
    ax = axes[1]
    im2 = ax.imshow(sm_coarse_agg, cmap='YlGnBu', vmin=0, vmax=0.5, extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], origin='upper', interpolation='nearest')
    ax.set_title(f'SM Coarse (~36km)\n{sm_coarse_agg.shape[0]}x{sm_coarse_agg.shape[1]} pixels', 
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    cbar2 = plt.colorbar(im2, ax=ax, label='SM (m3/m3)')
    
    stats_text = f'Mean: {np.nanmean(sm_coarse_valid):.3f}\n'
    stats_text += f'Std: {np.nanstd(sm_coarse_valid):.3f}\n'
    stats_text += f'Range: [{np.nanmin(sm_coarse_valid):.3f}, {np.nanmax(sm_coarse_valid):.3f}]'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    
    # 3. Scatter plot (pixel-wise comparison after aggregation)
    # Aggregate fine to coarse pixels for comparison (use same coarse shape)
    sm_fine_agg = aggregate_to_coarse(sm_fine_full[:CROP_Y, :CROP_X], native_shape=(CROP_Y, CROP_X), coarse_shape=sm_coarse_agg.shape)
    
    ax = axes[2]
    mask = np.isfinite(sm_fine_agg.ravel()) & np.isfinite(sm_coarse_agg.ravel())
    ax.scatter(sm_coarse_agg.ravel()[mask], sm_fine_agg.ravel()[mask], 
               alpha=0.6, s=100, edgecolors='black', linewidths=1)
    
    # 1:1 line
    min_val = min(np.nanmin(sm_coarse_agg), np.nanmin(sm_fine_agg))
    max_val = max(np.nanmax(sm_coarse_agg), np.nanmax(sm_fine_agg))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    ax.set_xlabel('SM Coarse (aggregated) [m3/m3]', fontsize=11)
    ax.set_ylabel('SM Fine (aggregated to coarse) [m3/m3]', fontsize=11)
    ax.set_title('Pixel-wise Comparison', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect('equal')
    
    # Calculate correlation
    if mask.sum() > 0:
        corr = np.corrcoef(sm_coarse_agg.ravel()[mask], sm_fine_agg.ravel()[mask])[0, 1]
        rmse = np.sqrt(np.mean((sm_coarse_agg.ravel()[mask] - sm_fine_agg.ravel()[mask])**2))
        bias = np.mean(sm_fine_agg.ravel()[mask] - sm_coarse_agg.ravel()[mask])
        
        stats_text = f'Correlation: {corr:.3f}\n'
        stats_text += f'RMSE: {rmse:.4f}\n'
        stats_text += f'Bias: {bias:.4f}\n'
        stats_text += f'N pixels: {mask.sum()}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    
    plt.tight_layout()
    
    # Save
    output_path = OUTPUT_DIR / f"SM_fine_vs_coarse_{date}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n  Plot saved: {output_path.name}")

print(f"\n{'='*70}")
print("ALL COMPARISONS COMPLETED")
print(f"{'='*70}")
