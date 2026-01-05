#!/usr/bin/env python3
"""
Comparación TB coarse vs fine para 20150607
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

date = '20150607'
interim_dir = Path('c:/Users/ortuo/tfgirati/tfg-nisar/data/interim')
processed_dir = Path('c:/Users/ortuo/tfgirati/tfg-nisar/data/processed')

# Cargar TB coarse (36km)
tb_coarse_file = interim_dir / f'smap-tb36-{date}-v.npz'
tb_coarse_data = np.load(tb_coarse_file)
tb_coarse = tb_coarse_data['TB_36km']

# Cargar TB fine (3km native)
tb_fine_file = processed_dir / f'TB_fine_{date}_VV_native.npz'
tb_fine_data = np.load(tb_fine_file)
tb_fine = tb_fine_data['TB_fine'][:27, :35]  # Crop

# Crear figura comparativa
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# TB Coarse
im1 = ax1.imshow(tb_coarse, cmap='RdYlBu_r', vmin=200, vmax=300, origin='upper')
ax1.set_title(f'TB Coarse (36 km) - {date}\nMean: {np.nanmean(tb_coarse):.2f} K', 
              fontsize=14, weight='bold')
ax1.set_xlabel('X (pixels)', fontsize=11)
ax1.set_ylabel('Y (pixels)', fontsize=11)
ax1.grid(True, alpha=0.3, color='white', linewidth=0.5)
cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cbar1.set_label('K', fontsize=11)

# TB Fine
im2 = ax2.imshow(tb_fine, cmap='RdYlBu_r', vmin=200, vmax=300, origin='upper')
ax2.set_title(f'TB Fine (3 km) - {date}\nMean: {np.nanmean(tb_fine):.2f} K', 
              fontsize=14, weight='bold')
ax2.set_xlabel('X (pixels)', fontsize=11)
ax2.set_ylabel('Y (pixels)', fontsize=11)
ax2.grid(True, alpha=0.3, color='white', linewidth=0.5)
cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label('K', fontsize=11)

# Título general
fig.suptitle(f'TB Coarse vs Fine - {date}', fontsize=16, weight='bold', y=0.98)

plt.tight_layout()

# Guardar
output_file = processed_dir / f'TB_coarse_vs_fine_{date}.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f'✓ Guardado: {output_file.name}')

# Stats
print(f'\nTB Coarse: shape={tb_coarse.shape}, min={np.nanmin(tb_coarse):.2f}K, max={np.nanmax(tb_coarse):.2f}K, mean={np.nanmean(tb_coarse):.2f}K')
print(f'TB Fine:   shape={tb_fine.shape}, min={np.nanmin(tb_fine):.2f}K, max={np.nanmax(tb_fine):.2f}K, mean={np.nanmean(tb_fine):.2f}K')

plt.close()
