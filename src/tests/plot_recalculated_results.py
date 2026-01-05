#!/usr/bin/env python3
"""
Visualización de SM fine y TB fine recalculados con nuevos beta/gamma
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

dates = ['20150607', '20150610', '20150615', '20150618', '20150620']
data_dir = Path('c:/Users/ortuo/tfgirati/tfg-nisar/data/processed')

# ===============================
# 1. SOIL MOISTURE FINE
# ===============================
print('=== Cargando SM fine recalculados ===')
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, date in enumerate(dates):
    sm_file = data_dir / f'SM_fine_{date}_TBV_tauomega_ATBD.npz'
    data = np.load(sm_file)
    sm = data['soil_moisture'][:27, :35]  # Crop
    
    im = axes[i].imshow(sm, cmap='YlGnBu', vmin=0.0, vmax=0.5, origin='upper')
    axes[i].set_title(f'SM {date}\nMedia: {np.nanmean(sm):.4f} m³/m³', fontsize=12, weight='bold')
    axes[i].set_xlabel('X (pixels)', fontsize=10)
    axes[i].set_ylabel('Y (pixels)', fontsize=10)
    axes[i].grid(True, alpha=0.3, color='white', linewidth=0.5)
    plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04, label='m³/m³')
    
    # Estadísticas
    valid = np.isfinite(sm)
    print(f'{date}: min={np.nanmin(sm):.4f}, max={np.nanmax(sm):.4f}, mean={np.nanmean(sm):.4f}, valid={valid.sum()}/{sm.size}')

# Ocultar el subplot sobrante
axes[5].axis('off')

plt.suptitle('Soil Moisture Fine - Recalculado con β=-3.71 K/dB, γ=0.19', 
             fontsize=16, weight='bold', y=0.995)
plt.tight_layout()
plt.savefig(data_dir / 'SM_fine_recalculated_all_dates.png', dpi=150, bbox_inches='tight')
print('[OK] Guardado: SM_fine_recalculated_all_dates.png\n')
plt.close()

# ===============================
# 2. BRIGHTNESS TEMPERATURE FINE
# ===============================
print('=== Cargando TB fine recalculados ===')
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, date in enumerate(dates):
    tb_file = data_dir / f'TB_fine_{date}_VV_native.npz'
    data = np.load(tb_file)
    tb = data['TB_fine'][:27, :35]  # Crop
    
    im = axes[i].imshow(tb, cmap='RdYlBu_r', vmin=200, vmax=300, origin='upper')
    axes[i].set_title(f'TB {date}\nMedia: {np.nanmean(tb):.2f} K', fontsize=12, weight='bold')
    axes[i].set_xlabel('X (pixels)', fontsize=10)
    axes[i].set_ylabel('Y (pixels)', fontsize=10)
    axes[i].grid(True, alpha=0.3, color='white', linewidth=0.5)
    plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04, label='K')
    
    # Estadísticas
    valid = np.isfinite(tb)
    print(f'{date}: min={np.nanmin(tb):.2f}K, max={np.nanmax(tb):.2f}K, mean={np.nanmean(tb):.2f}K, valid={valid.sum()}/{tb.size}')

# Ocultar el subplot sobrante
axes[5].axis('off')

plt.suptitle('Brightness Temperature Fine (VV) - Recalculado con β=-3.71 K/dB, γ=0.19', 
             fontsize=16, weight='bold', y=0.995)
plt.tight_layout()
plt.savefig(data_dir / 'TB_fine_recalculated_all_dates.png', dpi=150, bbox_inches='tight')
print('[OK] Guardado: TB_fine_recalculated_all_dates.png\n')
plt.close()

# ===============================
# 3. SERIES TEMPORALES
# ===============================
print('=== Generando series temporales ===')
sm_means = []
tb_means = []

for date in dates:
    sm_file = data_dir / f'SM_fine_{date}_TBV_tauomega_ATBD.npz'
    tb_file = data_dir / f'TB_fine_{date}_VV_native.npz'
    
    sm = np.load(sm_file)['soil_moisture'][:27, :35]
    tb = np.load(tb_file)['TB_fine'][:27, :35]
    
    sm_means.append(np.nanmean(sm))
    tb_means.append(np.nanmean(tb))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# SM time series
ax1.plot(dates, sm_means, 'o-', linewidth=2.5, markersize=10, color='steelblue')
ax1.fill_between(range(len(dates)), sm_means, alpha=0.3, color='steelblue')
ax1.set_ylabel('Soil Moisture (m³/m³)', fontsize=13, weight='bold')
ax1.set_title('Serie Temporal SM - Recalculado', fontsize=14, weight='bold')
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)
for i, (date, sm) in enumerate(zip(dates, sm_means)):
    ax1.text(i, sm + 0.005, f'{sm:.4f}', ha='center', fontsize=9, weight='bold')

# TB time series
ax2.plot(dates, tb_means, 's-', linewidth=2.5, markersize=10, color='orangered')
ax2.fill_between(range(len(dates)), tb_means, alpha=0.3, color='orangered')
ax2.set_ylabel('Brightness Temperature (K)', fontsize=13, weight='bold')
ax2.set_xlabel('Date', fontsize=13)
ax2.set_title('Serie Temporal TB - Recalculado', fontsize=14, weight='bold')
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='x', rotation=45)
for i, (date, tb) in enumerate(zip(dates, tb_means)):
    ax2.text(i, tb + 2, f'{tb:.1f}K', ha='center', fontsize=9, weight='bold')

plt.tight_layout()
plt.savefig(data_dir / 'Timeseries_SM_TB_recalculated.png', dpi=150, bbox_inches='tight')
print('[OK] Guardado: Timeseries_SM_TB_recalculated.png\n')
plt.close()

print('=== RESUMEN ===')
print(f'SM medio (todas las fechas): {np.mean(sm_means):.4f} m³/m³')
print(f'TB medio (todas las fechas): {np.mean(tb_means):.2f} K')
print(f'Tendencia SM: {sm_means[-1] - sm_means[0]:.4f} m³/m³')
print(f'Tendencia TB: {tb_means[-1] - tb_means[0]:.2f} K')
print('\n✓ Visualización completada')
