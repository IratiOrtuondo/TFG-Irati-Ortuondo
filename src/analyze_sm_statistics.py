#!/usr/bin/env python3
"""
Análisis estadístico completo de Soil Moisture ATBD
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Configuración
DATES = ['20150607', '20150610', '20150615', '20150618', '20150620']
DATA_DIR = Path(r"c:\Users\ortuo\tfgirati\tfg-nisar\data\processed")
CROP_Y = 27
CROP_X = 35

print("="*80)
print("ANALISIS ESTADISTICO DE SOIL MOISTURE - ATBD")
print("="*80)

# Arrays para almacenar estadísticas
stats_summary = []

for date in DATES:
    npz_path = DATA_DIR / f"SM_fine_{date}_TBV_tauomega_ATBD_reg.npz"
    
    if not npz_path.exists():
        print(f"\n[SKIP] {date}: file not found")
        continue
    
    print(f"\n{'='*80}")
    print(f"FECHA: {date}")
    print(f"{'='*80}")
    
    data = np.load(npz_path, allow_pickle=True)
    
    # Extraer variables (cropped)
    SM = data['soil_moisture'][:CROP_Y, :CROP_X]
    epsilon = data['epsilon_smooth'][:CROP_Y, :CROP_X]
    emissivity = data['e_rough'][:CROP_Y, :CROP_X]
    reflectivity = data['r_rough'][:CROP_Y, :CROP_X]
    TB = data['TB'][:CROP_Y, :CROP_X]
    
    # Metadatos
    pol = str(data.get('tb_pol', 'unknown'))
    
    # Estadísticas de Soil Moisture
    sm_valid = SM[(SM > 0) & (SM < 1) & np.isfinite(SM)]
    print(f"\n1. SOIL MOISTURE (m³/m³)")
    print(f"   Grid: {SM.shape} ({SM.size} pixels)")
    print(f"   Pixels validos: {sm_valid.size} ({100*sm_valid.size/SM.size:.1f}%)")
    print(f"   Media: {np.nanmean(sm_valid):.4f}")
    print(f"   Mediana: {np.nanmedian(sm_valid):.4f}")
    print(f"   Desv. std: {np.nanstd(sm_valid):.4f}")
    print(f"   Minimo: {np.nanmin(sm_valid):.4f}")
    print(f"   Maximo: {np.nanmax(sm_valid):.4f}")
    print(f"   Rango: {np.nanmax(sm_valid) - np.nanmin(sm_valid):.4f}")
    print(f"   Percentil 25: {np.nanpercentile(sm_valid, 25):.4f}")
    print(f"   Percentil 75: {np.nanpercentile(sm_valid, 75):.4f}")
    print(f"   Coef. variacion: {100*np.nanstd(sm_valid)/np.nanmean(sm_valid):.2f}%")
    
    # Estadísticas de constante dieléctrica
    eps_valid = epsilon[(epsilon > 0) & np.isfinite(epsilon)]
    print(f"\n2. CONSTANTE DIELECTRICA (epsilon)")
    print(f"   Media: {np.nanmean(eps_valid):.3f}")
    print(f"   Desv. std: {np.nanstd(eps_valid):.3f}")
    print(f"   Rango: [{np.nanmin(eps_valid):.3f}, {np.nanmax(eps_valid):.3f}]")
    
    # Estadísticas de emisividad
    e_valid = emissivity[(emissivity > 0) & (emissivity < 1) & np.isfinite(emissivity)]
    print(f"\n3. EMISIVIDAD")
    print(f"   Media: {np.nanmean(e_valid):.4f}")
    print(f"   Desv. std: {np.nanstd(e_valid):.4f}")
    print(f"   Rango: [{np.nanmin(e_valid):.4f}, {np.nanmax(e_valid):.4f}]")
    
    # Estadísticas de reflectividad
    r_valid = reflectivity[(reflectivity >= 0) & (reflectivity < 1) & np.isfinite(reflectivity)]
    print(f"\n4. REFLECTIVIDAD")
    print(f"   Media: {np.nanmean(r_valid):.4f}")
    print(f"   Desv. std: {np.nanstd(r_valid):.4f}")
    print(f"   Rango: [{np.nanmin(r_valid):.4f}, {np.nanmax(r_valid):.4f}]")
    
    # Estadísticas de TB
    tb_valid = TB[np.isfinite(TB)]
    print(f"\n5. BRIGHTNESS TEMPERATURE (K)")
    print(f"   Media: {np.nanmean(tb_valid):.2f}")
    print(f"   Desv. std: {np.nanstd(tb_valid):.2f}")
    print(f"   Rango: [{np.nanmin(tb_valid):.2f}, {np.nanmax(tb_valid):.2f}]")
    
    # Clasificación de humedad
    very_dry = np.sum(sm_valid < 0.10)
    dry = np.sum((sm_valid >= 0.10) & (sm_valid < 0.20))
    moderate = np.sum((sm_valid >= 0.20) & (sm_valid < 0.30))
    wet = np.sum((sm_valid >= 0.30) & (sm_valid < 0.40))
    very_wet = np.sum(sm_valid >= 0.40)
    
    print(f"\n6. CLASIFICACION DE HUMEDAD")
    print(f"   Muy seco (<0.10): {very_dry} pixels ({100*very_dry/sm_valid.size:.1f}%)")
    print(f"   Seco (0.10-0.20): {dry} pixels ({100*dry/sm_valid.size:.1f}%)")
    print(f"   Moderado (0.20-0.30): {moderate} pixels ({100*moderate/sm_valid.size:.1f}%)")
    print(f"   Humedo (0.30-0.40): {wet} pixels ({100*wet/sm_valid.size:.1f}%)")
    print(f"   Muy humedo (>=0.40): {very_wet} pixels ({100*very_wet/sm_valid.size:.1f}%)")
    
    # Guardar para resumen
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

# RESUMEN TEMPORAL
print(f"\n{'='*80}")
print(f"RESUMEN TEMPORAL - EVOLUCION DE SM")
print(f"{'='*80}")
print(f"\n{'Fecha':<12} {'SM Mean':<10} {'SM Std':<10} {'SM Min':<10} {'SM Max':<10} {'TB Mean':<10}")
print("-"*80)
for s in stats_summary:
    print(f"{s['date']:<12} {s['sm_mean']:<10.4f} {s['sm_std']:<10.4f} "
          f"{s['sm_min']:<10.4f} {s['sm_max']:<10.4f} {s['tb_mean']:<10.2f}")

# Estadísticas globales (todas las fechas)
all_sm_means = [s['sm_mean'] for s in stats_summary]
all_sm_stds = [s['sm_std'] for s in stats_summary]
all_tb_means = [s['tb_mean'] for s in stats_summary]

print(f"\n{'='*80}")
print(f"ESTADISTICAS GLOBALES (5 fechas)")
print(f"{'='*80}")
print(f"\nSoil Moisture:")
print(f"  Media temporal: {np.mean(all_sm_means):.4f} m³/m³")
print(f"  Std de las medias: {np.std(all_sm_means):.4f}")
print(f"  Rango de medias: [{np.min(all_sm_means):.4f}, {np.max(all_sm_means):.4f}]")
print(f"  Variabilidad espacial promedio: {np.mean(all_sm_stds):.4f}")

print(f"\nBrightness Temperature:")
print(f"  Media temporal: {np.mean(all_tb_means):.2f} K")
print(f"  Std de las medias: {np.std(all_tb_means):.2f} K")
print(f"  Rango de medias: [{np.min(all_tb_means):.2f}, {np.max(all_tb_means):.2f}] K")

# Tendencia temporal
print(f"\nTendencia temporal:")
sm_change = all_sm_means[-1] - all_sm_means[0]
tb_change = all_tb_means[-1] - all_tb_means[0]
print(f"  Cambio en SM ({DATES[0]} -> {DATES[-1]}): {sm_change:+.4f} m³/m³ ({100*sm_change/all_sm_means[0]:+.1f}%)")
print(f"  Cambio en TB ({DATES[0]} -> {DATES[-1]}): {tb_change:+.2f} K ({100*tb_change/all_tb_means[0]:+.1f}%)")

# Crear plot de series temporales
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

dates_num = range(len(DATES))

# SM
ax1.errorbar(dates_num, all_sm_means, yerr=all_sm_stds, 
             marker='o', markersize=8, linewidth=2, capsize=5, color='steelblue')
ax1.set_ylabel('Soil Moisture (m³/m³)', fontsize=12, fontweight='bold')
ax1.set_title('Temporal Evolution of Soil Moisture', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(dates_num)
ax1.set_xticklabels(DATES, rotation=45)
ax1.axhline(y=np.mean(all_sm_means), color='red', linestyle='--', linewidth=1.5, 
           label=f'Mean: {np.mean(all_sm_means):.3f}')
ax1.legend()

# TB
ax2.plot(dates_num, all_tb_means, marker='s', markersize=8, linewidth=2, color='orangered')
ax2.set_ylabel('Brightness Temperature (K)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
ax2.set_title('Temporal Evolution of Brightness Temperature', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(dates_num)
ax2.set_xticklabels(DATES, rotation=45)
ax2.axhline(y=np.mean(all_tb_means), color='red', linestyle='--', linewidth=1.5,
           label=f'Mean: {np.mean(all_tb_means):.1f} K')
ax2.legend()

plt.tight_layout()
output_plot = DATA_DIR / "plots" / "SM_temporal_evolution.png"
plt.savefig(output_plot, dpi=150, bbox_inches='tight')
print(f"\n[OK] Plot guardado: {output_plot}")

print(f"\n{'='*80}")
print(f"ANALISIS COMPLETADO")
print(f"{'='*80}\n")
