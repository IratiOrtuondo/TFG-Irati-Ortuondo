#!/usr/bin/env python3
"""
Visualización individual por fecha: TB fine y SM fine recalculados
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

dates = ['20150607', '20150610', '20150615', '20150618', '20150620']
data_dir = Path('c:/Users/ortuo/tfgirati/tfg-nisar/data/processed')

for date in dates:
    print(f'Procesando {date}...')
    
    # Cargar datos
    sm_file = data_dir / f'SM_fine_{date}_TBV_tauomega_ATBD.npz'
    tb_file = data_dir / f'TB_fine_{date}_VV_native.npz'
    
    sm_data = np.load(sm_file)
    tb_data = np.load(tb_file)
    
    sm = sm_data['soil_moisture'][:27, :35]
    tb = tb_data['TB_fine'][:27, :35]
    
    # Crear figura con 2 paneles
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Panel 1: TB fine
    im1 = ax1.imshow(tb, cmap='RdYlBu_r', vmin=200, vmax=300, origin='upper')
    ax1.set_title(f'TB Fine - {date}\nMean: {np.nanmean(tb):.2f} K', 
                  fontsize=14, weight='bold')
    ax1.set_xlabel('X (pixels)', fontsize=11)
    ax1.set_ylabel('Y (pixels)', fontsize=11)
    ax1.grid(True, alpha=0.3, color='white', linewidth=0.5)
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('K', fontsize=11)
    
    # Panel 2: SM fine
    im2 = ax2.imshow(sm, cmap='YlGnBu', vmin=0.0, vmax=0.5, origin='upper')
    ax2.set_title(f'SM Fine - {date}\nMean: {np.nanmean(sm):.4f} m³/m³', 
                  fontsize=14, weight='bold')
    ax2.set_xlabel('X (pixels)', fontsize=11)
    ax2.set_ylabel('Y (pixels)', fontsize=11)
    ax2.grid(True, alpha=0.3, color='white', linewidth=0.5)
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('m³/m³', fontsize=11)
    
    # Título general
    fig.suptitle(f'TB Fine and SM Fine - {date}', 
                 fontsize=16, weight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Guardar
    output_file = data_dir / f'TB_SM_fine_{date}_recalc.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f'  ✓ Guardado: {output_file.name}')
    
    # Stats
    print(f'  TB: min={np.nanmin(tb):.2f}K, max={np.nanmax(tb):.2f}K, mean={np.nanmean(tb):.2f}K')
    print(f'  SM: min={np.nanmin(sm):.4f}, max={np.nanmax(sm):.4f}, mean={np.nanmean(sm):.4f} m³/m³\n')
    
    plt.close()

print('✓ Generación completada - 5 archivos creados')
