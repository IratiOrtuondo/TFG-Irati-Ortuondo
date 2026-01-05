#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualización de resultados ATBD de soil moisture
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuración
DATES = ['20150607', '20150610', '20150615', '20150618', '20150620']
DATA_DIR = Path(r"c:\Users\ortuo\tfgirati\tfg-nisar\data\processed")
OUTPUT_DIR = DATA_DIR / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CROP_Y = 27
CROP_X = 35

print(f"\n{'='*70}")
print(f"ATBD SOIL MOISTURE VISUALIZATION - ALL DATES")
print(f"{'='*70}\n")

for date in DATES:
    # Cargar datos
    npz_path = DATA_DIR / f"SM_fine_{date}_TBV_tauomega_ATBD_reg.npz"
    
    if not npz_path.exists():
        print(f"[WARN] Skipping {date}: file not found")
        continue
    
    print(f"\nProcessing: {date}")
    print(f"Loading: {npz_path.name}")

    data = np.load(npz_path, allow_pickle=True)
    
    # Ver qué keys tiene
    # print(f"Available keys: {list(data.keys())}\n")
    
    # Extraer variables y cropear
    SM = data['soil_moisture'][:CROP_Y, :CROP_X]
    epsilon = data['epsilon_smooth'][:CROP_Y, :CROP_X]
    emissivity = data['e_rough'][:CROP_Y, :CROP_X]
    reflectivity = data['r_rough'][:CROP_Y, :CROP_X]
    TB = data['TB'][:CROP_Y, :CROP_X]
    
    # Información del archivo
    pol = str(data.get('tb_pol', 'unknown'))
    
    print(f"Date: {date}")
    print(f"Polarization: {pol}")
    print(f"Grid dimensions: {SM.shape}")
    
    # Estadísticas
    sm_valid = SM[(SM > 0) & (SM < 1) & np.isfinite(SM)]
    eps_valid = epsilon[(epsilon > 0) & np.isfinite(epsilon)]
    e_valid = emissivity[(emissivity > 0) & (emissivity < 1) & np.isfinite(emissivity)]
    r_valid = reflectivity[(reflectivity >= 0) & (reflectivity < 1) & np.isfinite(reflectivity)]
    tb_valid = TB[np.isfinite(TB)]
    
    print(f"  SM: Mean={np.nanmean(sm_valid):.4f}, Std={np.nanstd(sm_valid):.4f}, N={sm_valid.size}")
    
    # Crear figura con subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'ATBD Soil Moisture Retrieval - {date} - {pol} Polarization', 
                 fontsize=16, fontweight='bold')

    # 1. Soil Moisture
    ax = axes[0, 0]
    im1 = ax.imshow(SM, cmap='YlGnBu', vmin=0, vmax=0.5, aspect='auto')
    ax.set_title('Soil Moisture', fontsize=12, fontweight='bold')
    ax.set_xlabel('X (pixel)')
    ax.set_ylabel('Y (pixel)')
    cbar1 = plt.colorbar(im1, ax=ax, label='SM (m³/m³)')

    # Estadísticas en el plot
    stats_text = f'Mean: {np.nanmean(sm_valid):.3f}\n'
    stats_text += f'Std: {np.nanstd(sm_valid):.3f}\n'
    stats_text += f'Min: {np.nanmin(sm_valid):.3f}\n'
    stats_text += f'Max: {np.nanmax(sm_valid):.3f}\n'
    stats_text += f'N: {sm_valid.size}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    # 2. Epsilon (permittividad dieléctrica)
    ax = axes[0, 1]
    im2 = ax.imshow(epsilon, cmap='viridis', vmin=1, vmax=30, aspect='auto')
    ax.set_title('Dielectric Constant', fontsize=12, fontweight='bold')
    ax.set_xlabel('X (pixel)')
    ax.set_ylabel('Y (pixel)')
    cbar2 = plt.colorbar(im2, ax=ax, label='ε (dimensionless)')

    stats_text = f'Mean: {np.nanmean(eps_valid):.2f}\n'
    stats_text += f'Std: {np.nanstd(eps_valid):.2f}\n'
    stats_text += f'Min: {np.nanmin(eps_valid):.2f}\n'
    stats_text += f'Max: {np.nanmax(eps_valid):.2f}\n'
    stats_text += f'N: {eps_valid.size}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    # 3. Emissivity
    ax = axes[0, 2]
    im3 = ax.imshow(emissivity, cmap='plasma', vmin=0.5, vmax=1.0, aspect='auto')
    ax.set_title('Emissivity', fontsize=12, fontweight='bold')
    ax.set_xlabel('X (pixel)')
    ax.set_ylabel('Y (pixel)')
    cbar3 = plt.colorbar(im3, ax=ax, label='e (dimensionless)')

    stats_text = f'Mean: {np.nanmean(e_valid):.3f}\n'
    stats_text += f'Std: {np.nanstd(e_valid):.3f}\n'
    stats_text += f'Min: {np.nanmin(e_valid):.3f}\n'
    stats_text += f'Max: {np.nanmax(e_valid):.3f}\n'
    stats_text += f'N: {e_valid.size}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    # 4. Reflectivity
    ax = axes[1, 0]
    im4 = ax.imshow(reflectivity, cmap='coolwarm', vmin=0, vmax=0.5, aspect='auto')
    ax.set_title('Reflectivity', fontsize=12, fontweight='bold')
    ax.set_xlabel('X (pixel)')
    ax.set_ylabel('Y (pixel)')
    cbar4 = plt.colorbar(im4, ax=ax, label='r (dimensionless)')

    stats_text = f'Mean: {np.nanmean(r_valid):.3f}\n'
    stats_text += f'Std: {np.nanstd(r_valid):.3f}\n'
    stats_text += f'Min: {np.nanmin(r_valid):.3f}\n'
    stats_text += f'Max: {np.nanmax(r_valid):.3f}\n'
    stats_text += f'N: {r_valid.size}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    # 5. Brightness Temperature
    ax = axes[1, 1]
    im5 = ax.imshow(TB, cmap='hot', vmin=220, vmax=290, aspect='auto')
    ax.set_title('Brightness Temperature', fontsize=12, fontweight='bold')
    ax.set_xlabel('X (pixel)')
    ax.set_ylabel('Y (pixel)')
    cbar5 = plt.colorbar(im5, ax=ax, label='TB (K)')

    stats_text = f'Mean: {np.nanmean(tb_valid):.1f} K\n'
    stats_text += f'Std: {np.nanstd(tb_valid):.1f} K\n'
    stats_text += f'Min: {np.nanmin(tb_valid):.1f} K\n'
    stats_text += f'Max: {np.nanmax(tb_valid):.1f} K\n'
    stats_text += f'N: {tb_valid.size}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    # 6. Histograma de SM
    ax = axes[1, 2]
    ax.hist(sm_valid.ravel(), bins=40, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Soil Moisture (m³/m³)', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title('SM Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axvline(np.nanmean(sm_valid), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {np.nanmean(sm_valid):.3f}')
    ax.axvline(np.nanmedian(sm_valid), color='orange', linestyle='--', linewidth=2,
               label=f'Median: {np.nanmedian(sm_valid):.3f}')
    ax.legend(fontsize=9)

    plt.tight_layout()

    # Guardar
    output_path = OUTPUT_DIR / f"SM_ATBD_{date}_{pol}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Plot saved: {output_path.name}")
print(f"✓ ALL PLOTS COMPLETED")
print(f"{'='*70}\n")
