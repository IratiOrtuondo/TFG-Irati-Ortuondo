#!/usr/bin/env python3
"""
Comprehensive Statistical Analysis of Soil Moisture ATBD Algorithm
=================================================================
This script performs detailed statistical analysis on soil moisture estimates
derived from NISAR SMAP data using the ATBD (Algorithm Theoretical Basis Document)
methodology. It processes multiple dates and generates temporal evolution plots
with comprehensive statistical summaries.
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================
# List of dates to analyze (YYYYMMDD format)
DATES = ['20150607', '20150610', '20150615', '20150618', '20150620']

# Base directory containing processed data files
DATA_DIR = Path(r"c:\Users\ortuo\tfgirati\tfg-nisar\data\processed")

# Spatial cropping dimensions (pixels) to focus on region of interest
CROP_Y = 27  # Number of rows to keep
CROP_X = 35  # Number of columns to keep

print("="*80)
print("STATISTICAL ANALYSIS OF SOIL MOISTURE - ATBD ALGORITHM")
print("="*80)

# ============================================================================
# DATA PROCESSING LOOP
# ============================================================================
# List to accumulate statistical summaries across all dates for temporal analysis
stats_summary = []

# Process each date in the dataset
for date in DATES:
    # Construct file path for the NPZ (NumPy compressed) file containing
    # the ATBD-derived soil moisture and related geophysical parameters
    npz_path = DATA_DIR / f"SM_fine_{date}_TBV_tauomega_ATBD_reg.npz"
    
    # Skip processing if the data file does not exist
    if not npz_path.exists():
        print(f"\n[SKIP] {date}: file not found")
        continue
    
    print(f"\n{'='*80}")
    print(f"DATE: {date}")
    print(f"{'='*80}")
    
    # Load all variables from the compressed NPZ file
    # allow_pickle=True enables loading of complex Python objects if present
    data = np.load(npz_path, allow_pickle=True)
    
    # ========================================================================
    # EXTRACT AND CROP GEOPHYSICAL VARIABLES
    # ========================================================================
    # Retrieve and spatially crop each variable to the region of interest
    # Crops to the first CROP_Y rows and CROP_X columns
    
    # Soil moisture content in m³/m³ (volumetric water content)
    SM = data['soil_moisture'][:CROP_Y, :CROP_X]
    
    # Smooth dielectric constant (complex permittivity of soil)
    epsilon = data['epsilon_smooth'][:CROP_Y, :CROP_X]
    
    # Surface roughness-induced emissivity
    emissivity = data['e_rough'][:CROP_Y, :CROP_X]
    
    # Surface roughness-induced reflectivity
    reflectivity = data['r_rough'][:CROP_Y, :CROP_X]
    
    # Brightness temperature measured at top of atmosphere (K)
    TB = data['TB'][:CROP_Y, :CROP_X]
    
    # Extract polarization metadata (e.g., "VV", "HH")
    pol = str(data.get('tb_pol', 'unknown'))
    
    # ========================================================================
    # SOIL MOISTURE STATISTICS
    # ========================================================================
    # Filter for physically valid SM values: 0-1 m³/m³ and non-NaN
    sm_valid = SM[(SM > 0) & (SM < 1) & np.isfinite(SM)]
    
    print(f"\n1. SOIL MOISTURE (m³/m³)")
    print(f"   Grid dimensions: {SM.shape} ({SM.size} total pixels)")
    print(f"   Valid pixels: {sm_valid.size} ({100*sm_valid.size/SM.size:.1f}% of total)")
    print(f"   Mean: {np.nanmean(sm_valid):.4f}")
    print(f"   Median: {np.nanmedian(sm_valid):.4f}")
    print(f"   Standard deviation: {np.nanstd(sm_valid):.4f}")
    print(f"   Minimum: {np.nanmin(sm_valid):.4f}")
    print(f"   Maximum: {np.nanmax(sm_valid):.4f}")
    print(f"   Range (Max-Min): {np.nanmax(sm_valid) - np.nanmin(sm_valid):.4f}")
    print(f"   25th percentile: {np.nanpercentile(sm_valid, 25):.4f}")
    print(f"   75th percentile: {np.nanpercentile(sm_valid, 75):.4f}")
    # Coefficient of variation: ratio of standard deviation to mean (%)
    print(f"   Coefficient of variation: {100*np.nanstd(sm_valid)/np.nanmean(sm_valid):.2f}%")
    
    # ========================================================================
    # DIELECTRIC CONSTANT STATISTICS
    # ========================================================================
    # Filter for valid epsilon values (positive and non-NaN)
    eps_valid = epsilon[(epsilon > 0) & np.isfinite(epsilon)]
    
    print(f"\n2. DIELECTRIC CONSTANT (epsilon)")
    print(f"   Mean: {np.nanmean(eps_valid):.3f}")
    print(f"   Standard deviation: {np.nanstd(eps_valid):.3f}")
    print(f"   Range: [{np.nanmin(eps_valid):.3f}, {np.nanmax(eps_valid):.3f}]")
    
    # ========================================================================
    # EMISSIVITY STATISTICS
    # ========================================================================
    # Filter for physically valid emissivity values: 0-1 and non-NaN
    e_valid = emissivity[(emissivity > 0) & (emissivity < 1) & np.isfinite(emissivity)]
    
    print(f"\n3. EMISSIVITY")
    print(f"   Mean: {np.nanmean(e_valid):.4f}")
    print(f"   Standard deviation: {np.nanstd(e_valid):.4f}")
    print(f"   Range: [{np.nanmin(e_valid):.4f}, {np.nanmax(e_valid):.4f}]")
    
    # ========================================================================
    # REFLECTIVITY STATISTICS
    # ========================================================================
    # Filter for physically valid reflectivity values: 0-1 and non-NaN
    r_valid = reflectivity[(reflectivity >= 0) & (reflectivity < 1) & np.isfinite(reflectivity)]
    
    print(f"\n4. REFLECTIVITY")
    print(f"   Mean: {np.nanmean(r_valid):.4f}")
    print(f"   Standard deviation: {np.nanstd(r_valid):.4f}")
    print(f"   Range: [{np.nanmin(r_valid):.4f}, {np.nanmax(r_valid):.4f}]")
    
    # ========================================================================
    # BRIGHTNESS TEMPERATURE STATISTICS
    # ========================================================================
    # Filter for valid TB values (non-NaN)
    tb_valid = TB[np.isfinite(TB)]
    
    print(f"\n5. BRIGHTNESS TEMPERATURE (K)")
    print(f"   Mean: {np.nanmean(tb_valid):.2f}")
    print(f"   Standard deviation: {np.nanstd(tb_valid):.2f}")
    print(f"   Range: [{np.nanmin(tb_valid):.2f}, {np.nanmax(tb_valid):.2f}]")
    
    # ========================================================================
    # SOIL MOISTURE CLASSIFICATION BY MOISTURE LEVELS
    # ========================================================================
    # Categorize pixels into moisture classes based on threshold values
    very_dry = np.sum(sm_valid < 0.10)
    dry = np.sum((sm_valid >= 0.10) & (sm_valid < 0.20))
    moderate = np.sum((sm_valid >= 0.20) & (sm_valid < 0.30))
    wet = np.sum((sm_valid >= 0.30) & (sm_valid < 0.40))
    very_wet = np.sum(sm_valid >= 0.40)
    
    print(f"\n6. SOIL MOISTURE CLASSIFICATION")
    print(f"   Very dry (<0.10): {very_dry} pixels ({100*very_dry/sm_valid.size:.1f}%)")
    print(f"   Dry (0.10-0.20): {dry} pixels ({100*dry/sm_valid.size:.1f}%)")
    print(f"   Moderate (0.20-0.30): {moderate} pixels ({100*moderate/sm_valid.size:.1f}%)")
    print(f"   Wet (0.30-0.40): {wet} pixels ({100*wet/sm_valid.size:.1f}%)")
    print(f"   Very wet (>=0.40): {very_wet} pixels ({100*very_wet/sm_valid.size:.1f}%)")
    
    # ========================================================================
    # ACCUMULATE STATISTICS FOR TEMPORAL ANALYSIS
    # ========================================================================
    # Store key statistics in dictionary for later temporal comparison
    stats_summary.append({
        'date': date,
        'sm_mean': np.nanmean(sm_valid),
        'sm_std': np.nanstd(sm_valid),
        'sm_min': np.nanmin(sm_valid),
        'sm_max': np.nanmax(sm_valid),
        'sm_median': np.nanmedian(sm_valid),
        'tb_mean': np.nanmean(tb_valid),
        'tb_std': np.nanstd(tb_valid),
        'eps_mean': np.nanmean(eps_valid),
        'e_mean': np.nanmean(e_valid),
        'n_valid': sm_valid.size,
    })

# ============================================================================
# TEMPORAL SUMMARY ANALYSIS
# ============================================================================
# Analyze statistical trends across all dates
print(f"\n{'='*80}")
print(f"TEMPORAL SUMMARY - SOIL MOISTURE EVOLUTION")
print(f"{'='*80}")
print(f"\n{'Date':<12} {'SM Mean':<10} {'SM Std':<10} {'SM Min':<10} {'SM Max':<10} {'TB Mean':<10}")
print("-"*80)

# Print statistics for each date in tabular format
for s in stats_summary:
    print(f"{s['date']:<12} {s['sm_mean']:<10.4f} {s['sm_std']:<10.4f} "
          f"{s['sm_min']:<10.4f} {s['sm_max']:<10.4f} {s['tb_mean']:<10.2f}")

# ============================================================================
# GLOBAL STATISTICS ACROSS ALL DATES
# ============================================================================
# Extract time series of key parameters
all_sm_means = [s['sm_mean'] for s in stats_summary]
all_sm_stds = [s['sm_std'] for s in stats_summary]
all_tb_means = [s['tb_mean'] for s in stats_summary]

print(f"\n{'='*80}")
print(f"GLOBAL STATISTICS (5 dates analyzed)")
print(f"{'='*80}")

print(f"\nSoil Moisture across temporal period:")
print(f"  Temporal mean: {np.mean(all_sm_means):.4f} m³/m³")
print(f"  Std of daily means: {np.std(all_sm_means):.4f}")
print(f"  Range of daily means: [{np.min(all_sm_means):.4f}, {np.max(all_sm_means):.4f}]")
# Average spatial variability (mean of standard deviations for each date)
print(f"  Average spatial variability: {np.mean(all_sm_stds):.4f}")

print(f"\nBrightness Temperature across temporal period:")
print(f"  Temporal mean: {np.mean(all_tb_means):.2f} K")
print(f"  Std of daily means: {np.std(all_tb_means):.2f} K")
print(f"  Range of daily means: [{np.min(all_tb_means):.2f}, {np.max(all_tb_means):.2f}] K")

# ============================================================================
# TEMPORAL TREND ANALYSIS
# ============================================================================
# Calculate absolute and relative changes from first to last date
print(f"\nTemporal trend (evolution over the observation period):")
sm_change = all_sm_means[-1] - all_sm_means[0]
tb_change = all_tb_means[-1] - all_tb_means[0]

print(f"  SM change ({DATES[0]} -> {DATES[-1]}): {sm_change:+.4f} m³/m³ ({100*sm_change/all_sm_means[0]:+.1f}%)")
print(f"  TB change ({DATES[0]} -> {DATES[-1]}): {tb_change:+.2f} K ({100*tb_change/all_tb_means[0]:+.1f}%)")

# ============================================================================
# PLOTTING: TEMPORAL EVOLUTION VISUALIZATION
# ============================================================================
# Create a two-panel figure showing temporal evolution of SM and TB with errors
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Convert date indices to numeric values for x-axis plotting
dates_num = range(len(DATES))

# ============================================================================
# PANEL 1: Soil Moisture Temporal Evolution
# ============================================================================
# Plot SM with error bars representing spatial standard deviation
ax1.errorbar(dates_num, all_sm_means, yerr=all_sm_stds, 
             marker='o', markersize=8, linewidth=2, capsize=5, color='steelblue')
ax1.set_ylabel('Soil Moisture (m³/m³)', fontsize=12, fontweight='bold')
ax1.set_title('Temporal Evolution of Soil Moisture', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(dates_num)
ax1.set_xticklabels(DATES, rotation=45)

# Add horizontal reference line showing temporal mean
ax1.axhline(y=np.mean(all_sm_means), color='red', linestyle='--', linewidth=1.5, 
           label=f'Mean: {np.mean(all_sm_means):.3f}')
ax1.legend()

# ============================================================================
# PANEL 2: Brightness Temperature Temporal Evolution
# ============================================================================
# Plot TB daily means showing temporal variation
ax2.plot(dates_num, all_tb_means, marker='s', markersize=8, linewidth=2, color='orangered')
ax2.set_ylabel('Brightness Temperature (K)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
ax2.set_title('Temporal Evolution of Brightness Temperature', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(dates_num)
ax2.set_xticklabels(DATES, rotation=45)

# Add horizontal reference line showing temporal mean
ax2.axhline(y=np.mean(all_tb_means), color='red', linestyle='--', linewidth=1.5,
           label=f'Mean: {np.mean(all_tb_means):.1f} K')
ax2.legend()

# Optimize layout spacing between subplots
plt.tight_layout()

# ============================================================================
# SAVE VISUALIZATION
# ============================================================================
# Save the temporal evolution plot to the output directory
output_plot = DATA_DIR / "plots" / "SM_temporal_evolution.png"
plt.savefig(output_plot, dpi=150, bbox_inches='tight')
print(f"\n[OK] Plot saved: {output_plot}")

# ============================================================================
# COMPLETION MESSAGE
# ============================================================================
print(f"\n{'='*80}")
print(f"STATISTICAL ANALYSIS COMPLETED SUCCESSFULLY")
print(f"{'='*80}\n")
