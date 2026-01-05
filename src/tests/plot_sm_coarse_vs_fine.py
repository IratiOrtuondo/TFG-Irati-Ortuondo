#!/usr/bin/env python3
"""
Comparación SM coarse (SMAP L3) vs fine (disaggregated) para 20150607
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import h5py

date = '20150607'
raw_dir = Path('c:/Users/ortuo/tfgirati/tfg-nisar/data/raw')
interim_dir = Path('c:/Users/ortuo/tfgirati/tfg-nisar/data/interim')
processed_dir = Path('c:/Users/ortuo/tfgirati/tfg-nisar/data/processed')

# Primero cargar TB coarse para obtener los índices EASE2 correctos
tb_coarse_file = interim_dir / f'smap-tb36-{date}-v.npz'
tb_coarse_data = np.load(tb_coarse_file)
print(f'TB coarse shape: {tb_coarse_data["TB_36km"].shape}')

# Cargar SM coarse desde SMAP L3
l3_file = raw_dir / f'SMAP_L3_SM_A_{date}_R13080_001.h5'
print(f'Cargando SM coarse desde {l3_file.name}...')

with h5py.File(l3_file, 'r') as f:
    sm_group = f['Soil_Moisture_Retrieval_Data']
    
    # Cargar datos con estructura de array 1D
    sm_coarse_1d = sm_group['soil_moisture'][:]
    lat_1d = sm_group['latitude'][:]
    lon_1d = sm_group['longitude'][:]
    
    # Filtrar valores inválidos
    valid_mask = (sm_coarse_1d > 0) & (sm_coarse_1d < 1)
    
    lon_min = -104.8884912
    lat_min = 39.8008444
    lon_max = -103.7115088
    lat_max = 40.6991556
    
    # Filtrar región
    region_mask = valid_mask & (lat_1d >= lat_min) & (lat_1d <= lat_max) & (lon_1d >= lon_min) & (lon_1d <= lon_max)
    
    sm_region = sm_coarse_1d[region_mask]
    lat_region = lat_1d[region_mask]
    lon_region = lon_1d[region_mask]
    
    print(f'Pixels válidos en región: {len(sm_region)}')
    
    if len(sm_region) > 15:  # Debería haber exactamente 15 pixels para 3x5
        # Ordenar por lat y lon para crear grid
        from scipy.spatial import cKDTree
        
        # Agrupar pixels cercanos para obtener el grid 3x5
        lat_round = np.round(lat_region, 1)
        lon_round = np.round(lon_region, 1)
        
        lat_unique = np.unique(lat_round)
        lon_unique = np.unique(lon_round)
        
        # Tomar los 3x5 centrales si hay más
        if len(lat_unique) > 3:
            lat_idx = np.argsort(lat_unique)
            mid_idx = len(lat_unique) // 2
            lat_sel = lat_unique[lat_idx[mid_idx-1:mid_idx+2]]
        else:
            lat_sel = lat_unique
            
        if len(lon_unique) > 5:
            lon_idx = np.argsort(lon_unique)
            mid_idx = len(lon_unique) // 2
            lon_sel = lon_unique[lon_idx[mid_idx-2:mid_idx+3]]
        else:
            lon_sel = lon_unique
        
        # Crear grid 3x5
        sm_coarse_grid = np.full((len(lat_sel), len(lon_sel)), np.nan)
        
        for i, lat_val in enumerate(sorted(lat_sel, reverse=True)):
            for j, lon_val in enumerate(sorted(lon_sel)):
                mask_pixel = (np.abs(lat_round - lat_val) < 0.05) & (np.abs(lon_round - lon_val) < 0.05)
                if np.any(mask_pixel):
                    sm_coarse_grid[i, j] = np.mean(sm_region[mask_pixel])
        
        print(f'Grid final: {sm_coarse_grid.shape}')
    else:
        sm_coarse_grid = np.full((3, 5), np.nan)

# Cargar SM fine
sm_fine_file = processed_dir / f'SM_fine_{date}_VV_tauomega.npz'
sm_fine_data = np.load(sm_fine_file)
sm_fine = sm_fine_data['soil_moisture'][:27, :35]  # Crop

# Crear figura comparativa
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# SM Coarse
im1 = ax1.imshow(sm_coarse_grid, cmap='YlGnBu', vmin=0.0, vmax=0.5, origin='upper')
ax1.set_title(f'SM Coarse (36 km, SMAP L3) - {date}\nMean: {np.nanmean(sm_coarse_grid):.4f} m³/m³', 
              fontsize=14, weight='bold')
ax1.set_xlabel('X (pixels)', fontsize=11)
ax1.set_ylabel('Y (pixels)', fontsize=11)
ax1.grid(True, alpha=0.3, color='white', linewidth=0.5)
cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cbar1.set_label('m³/m³', fontsize=11)

# SM Fine
im2 = ax2.imshow(sm_fine, cmap='YlGnBu', vmin=0.0, vmax=0.5, origin='upper')
ax2.set_title(f'SM Fine (3 km, Disaggregated) - {date}\nMean: {np.nanmean(sm_fine):.4f} m³/m³', 
              fontsize=14, weight='bold')
ax2.set_xlabel('X (pixels)', fontsize=11)
ax2.set_ylabel('Y (pixels)', fontsize=11)
ax2.grid(True, alpha=0.3, color='white', linewidth=0.5)
cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label('m³/m³', fontsize=11)

# Título general
fig.suptitle(f'SM Coarse vs Fine - {date}', fontsize=16, weight='bold', y=0.98)

plt.tight_layout()

# Guardar
output_file = processed_dir / f'SM_coarse_vs_fine_{date}.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f'\n✓ Guardado: {output_file.name}')

# Stats
print(f'\nSM Coarse: shape={sm_coarse_grid.shape}, min={np.nanmin(sm_coarse_grid):.4f}, max={np.nanmax(sm_coarse_grid):.4f}, mean={np.nanmean(sm_coarse_grid):.4f} m³/m³')
print(f'SM Fine:   shape={sm_fine.shape}, min={np.nanmin(sm_fine):.4f}, max={np.nanmax(sm_fine):.4f}, mean={np.nanmean(sm_fine):.4f} m³/m³')

plt.close()
