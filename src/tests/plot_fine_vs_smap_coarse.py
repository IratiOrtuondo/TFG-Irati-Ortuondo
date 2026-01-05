#!/usr/bin/env python3
"""
Plot SM_fine (ATBD disaggregated) vs SMAP L3 SM coarse (original 36km).
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import h5py
from scipy.stats import pearsonr

# Configuration
DATES = ['20150607', '20150610', '20150615', '20150618', '20150620']
RAW_DIR = Path(r"c:\Users\ortuo\tfgirati\tfg-nisar\data\raw")
PROCESSED_DIR = Path(r"c:\Users\ortuo\tfgirati\tfg-nisar\data\processed")
OUTPUT_DIR = PROCESSED_DIR / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Region bounds
LON_MIN = -104.8884912
LAT_MIN = 39.8008444
LON_MAX = -103.7115088
LAT_MAX = 40.6991556

# Crop for fine data
CROP_Y = 27
CROP_X = 35

def aggregate_to_coarse(fine_data, coarse_shape=(3, 3)):
    """
    Aggregate fine resolution data to coarse resolution by averaging blocks.
    """
    native_shape = fine_data.shape
    
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
            coarse[i, j] = np.nanmean(block)
    
    return coarse

def extract_smap_sm_3x3(date):
    """Extract the 3x3 coarse SM from SMAP L3 for the region."""
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
        
        # Replace fill values
        fill_value = -9999.0
        sm = np.where(sm == fill_value, np.nan, sm)
        
        # Crop to region
        mask = ((lat >= LAT_MIN) & (lat <= LAT_MAX) & 
                (lon >= LON_MIN) & (lon <= LON_MAX))
        
        if not np.any(mask):
            return None
        
        rows, cols = np.where(mask)
        row_min, row_max = rows.min(), rows.max() + 1
        col_min, col_max = cols.min(), cols.max() + 1
        
        sm_crop = sm[row_min:row_max, col_min:col_max]
        
        return sm_crop

print("="*70)
print("SM FINE (ATBD) vs SMAP L3 COARSE (36km) COMPARISON")
print("="*70)

# Store statistics for summary
all_stats = []

for date in DATES:
    print(f"\n{'='*70}")
    print(f"Processing: {date}")
    print(f"{'='*70}")
    
    # Load SM_fine: prefer relaxed outputs
    candidates = [
        PROCESSED_DIR / f"SM_fine_{date}_TBV_tauomega_ATBD_reg_relaxed.npz",
        PROCESSED_DIR / f"SM_fine_{date}_relaxed.npz",
        PROCESSED_DIR / f"SM_fine_{date}_TBV_tauomega_ATBD_reg.npz",
    ]
    data = None
    for sm_fine_file in candidates:
        if sm_fine_file.exists():
            data = np.load(sm_fine_file)
            break
    if data is None:
        print(f"[WARN] SM_fine not found (tried relaxed and regular): {candidates}")
        continue

    # pick common keys
    if 'soil_moisture' in data:
        sm_fine_full = data['soil_moisture']
    elif 'SM_fine' in data:
        sm_fine_full = data['SM_fine']
    else:
        # fallback: first ndarray
        for k in data.keys():
            if isinstance(data[k], np.ndarray):
                sm_fine_full = data[k]
                break
    sm_fine = np.asarray(sm_fine_full)
    if sm_fine.ndim > 2:
        sm_fine = sm_fine.squeeze()
    sm_fine = sm_fine[:CROP_Y, :CROP_X]
    
    # Load SMAP coarse
    sm_coarse = extract_smap_sm_3x3(date)
    if sm_coarse is None:
        print(f"[WARN] Could not extract SMAP coarse for {date}")
        continue
    
    # Aggregate SM_fine to same resolution as coarse for comparison
    sm_fine_agg = aggregate_to_coarse(sm_fine, coarse_shape=sm_coarse.shape)
    
    print(f"  SM_fine shape: {sm_fine.shape}")
    print(f"  SM_coarse shape: {sm_coarse.shape}")
    print(f"  SM_fine mean: {np.nanmean(sm_fine):.4f} m3/m3")
    print(f"  SM_coarse mean: {np.nanmean(sm_coarse):.4f} m3/m3")
    print(f"  SM_fine_agg mean: {np.nanmean(sm_fine_agg):.4f} m3/m3")
    
    # Calculate statistics (pixel-wise comparison at coarse resolution)
    mask = np.isfinite(sm_fine_agg) & np.isfinite(sm_coarse)
    if np.sum(mask) > 1:
        correlation, p_value = pearsonr(sm_fine_agg[mask].ravel(), 
                                        sm_coarse[mask].ravel())
        rmse = np.sqrt(np.mean((sm_fine_agg[mask] - sm_coarse[mask])**2))
        bias = np.mean(sm_fine_agg[mask] - sm_coarse[mask])
        mae = np.mean(np.abs(sm_fine_agg[mask] - sm_coarse[mask]))
    else:
        correlation, p_value, rmse, bias, mae = np.nan, np.nan, np.nan, np.nan, np.nan
    
    print(f"\n  VALIDATION STATISTICS (at coarse resolution):")
    print(f"    Correlation (R): {correlation:.4f}")
    print(f"    RMSE: {rmse:.4f} m3/m3")
    print(f"    Bias: {bias:.4f} m3/m3")
    print(f"    MAE: {mae:.4f} m3/m3")
    
    # Store stats
    all_stats.append({
        'date': date,
        'correlation': correlation,
        'rmse': rmse,
        'bias': bias,
        'mae': mae,
        'fine_mean': np.nanmean(sm_fine),
        'coarse_mean': np.nanmean(sm_coarse)
    })
    
    # Create plot with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f'Soil Moisture Comparison - {date}', fontsize=16, fontweight='bold')
    
    # 1. SM_fine (ATBD disaggregated)
    ax = axes[0]
    im1 = ax.imshow(sm_fine, cmap='YlGnBu', vmin=0, vmax=0.5, aspect='auto')
    ax.set_title(f'SM Fine - ATBD Disaggregated (~3km)\n{sm_fine.shape[0]}×{sm_fine.shape[1]} pixels', 
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('X (pixel)')
    ax.set_ylabel('Y (pixel)')
    cbar1 = plt.colorbar(im1, ax=ax, label='SM (m³/m³)', fraction=0.046, pad=0.04)
    
    sm_valid = sm_fine[(sm_fine > 0) & (sm_fine < 1) & np.isfinite(sm_fine)]
    stats_text = f'Mean: {np.nanmean(sm_valid):.3f}\n'
    stats_text += f'Std: {np.nanstd(sm_valid):.3f}\n'
    stats_text += f'Range: [{np.nanmin(sm_valid):.3f}, {np.nanmax(sm_valid):.3f}]'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    
    # 2. SM_coarse (SMAP L3 original)
    ax = axes[1]
    im2 = ax.imshow(sm_coarse, cmap='YlGnBu', vmin=0, vmax=0.5, aspect='auto')
    ax.set_title(f'SM Coarse - SMAP L3 Original (~36km)\n{sm_coarse.shape[0]}×{sm_coarse.shape[1]} pixels', 
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('X (pixel)')
    ax.set_ylabel('Y (pixel)')
    cbar2 = plt.colorbar(im2, ax=ax, label='SM (m³/m³)', fraction=0.046, pad=0.04)
    
    sm_coarse_valid = sm_coarse[np.isfinite(sm_coarse)]
    stats_text = f'Mean: {np.nanmean(sm_coarse_valid):.3f}\n'
    stats_text += f'Std: {np.nanstd(sm_coarse_valid):.3f}\n'
    stats_text += f'Range: [{np.nanmin(sm_coarse_valid):.3f}, {np.nanmax(sm_coarse_valid):.3f}]'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    
    # 3. Scatter plot (aggregated fine vs coarse)
    ax = axes[2]
    mask = np.isfinite(sm_fine_agg.ravel()) & np.isfinite(sm_coarse.ravel())
    ax.scatter(sm_coarse.ravel()[mask], sm_fine_agg.ravel()[mask],
               s=200, alpha=0.7, edgecolors='black', linewidths=2, c='steelblue')
    
    # 1:1 line (no legend entry)
    min_val = min(np.nanmin(sm_coarse), np.nanmin(sm_fine_agg))
    max_val = max(np.nanmax(sm_coarse), np.nanmax(sm_fine_agg))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    ax.set_xlabel('SMAP L3 Coarse (m³/m³)', fontsize=11, fontweight='bold')
    ax.set_ylabel('SM Fine Aggregated (m³/m³)', fontsize=11, fontweight='bold')
    ax.set_title('Pixel-wise Comparison\n(at coarse resolution)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Add statistics to scatter plot
    stats_text = f'R = {correlation:.3f}\n'
    stats_text += f'RMSE = {rmse:.4f}\n'
    stats_text += f'Bias = {bias:.4f}\n'
    stats_text += f'MAE = {mae:.4f}\n'
    stats_text += f'N = {np.sum(mask)}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    plt.tight_layout()
    
    # Save
    output_file = OUTPUT_DIR / f"SM_fine_vs_SMAP_coarse_{date}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Plot saved: {output_file.name}")

print("\n" + "="*70)
print("SUMMARY STATISTICS - ALL DATES")
print("="*70)
print(f"\n{'Date':<12} {'R':<8} {'RMSE':<10} {'Bias':<10} {'MAE':<10} {'Fine Mean':<12} {'Coarse Mean':<12}")
print("-"*86)
for stat in all_stats:
    print(f"{stat['date']:<12} {stat['correlation']:<8.4f} {stat['rmse']:<10.4f} "
          f"{stat['bias']:<10.4f} {stat['mae']:<10.4f} {stat['fine_mean']:<12.4f} "
          f"{stat['coarse_mean']:<12.4f}")

# Overall statistics
if len(all_stats) > 0:
    mean_r = np.mean([s['correlation'] for s in all_stats if np.isfinite(s['correlation'])])
    mean_rmse = np.mean([s['rmse'] for s in all_stats if np.isfinite(s['rmse'])])
    mean_bias = np.mean([s['bias'] for s in all_stats if np.isfinite(s['bias'])])
    mean_mae = np.mean([s['mae'] for s in all_stats if np.isfinite(s['mae'])])
    
    print("-"*86)
    print(f"{'MEAN':<12} {mean_r:<8.4f} {mean_rmse:<10.4f} {mean_bias:<10.4f} {mean_mae:<10.4f}")
    print("\n" + "="*70)
    print(f"Overall Performance:")
    print(f"  Mean Correlation: {mean_r:.4f}")
    print(f"  Mean RMSE: {mean_rmse:.4f} m3/m3")
    print(f"  Mean Bias: {mean_bias:.4f} m3/m3")
    print(f"  Mean MAE: {mean_mae:.4f} m3/m3")

print("\n" + "="*70)
print("ALL PLOTS COMPLETED")
print("="*70)
