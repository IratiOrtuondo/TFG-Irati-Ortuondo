#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import rasterio
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"   # <- añadimos esto

def main():
    npz_path = INTERIM / "aligned_step2_smap.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"No encuentro {npz_path}")

    data = np.load(npz_path, allow_pickle=True)

    TB  = data["TBc_2d"]   # radiometer TB [K]
    Spp = data["S_pp_dB"]  # radar sigma0 [dB]
    meta = data["meta"]

    print("Meta:")
    for m in meta:
        print("  -", m)

    print("\nShapes:")
    print("  TBc_2d :", TB.shape)
    print("  S_pp_dB:", Spp.shape)

    # Estadísticas rápidas
    def stats(arr, name):
        mask = np.isfinite(arr)
        if not mask.any():
            print(f"  {name}: todo NaN")
            return
        vals = arr[mask]
        print(f"  {name}: min={vals.min():.2f}, max={vals.max():.2f}, mean={vals.mean():.2f}")

    print("\nStats:")
    stats(TB,  "TBc_2d (K)")
    stats(Spp, "S_pp_dB (dB)")

    # Buscar un píxel donde haya TB y sigma0 válidos
    mask_valid = np.isfinite(TB) & np.isfinite(Spp)
    if not mask_valid.any():
        print("\n[WARN] No hay ningún píxel con TB y σ0 válidos a la vez.")
        return

    i, j = np.argwhere(mask_valid)[0]
    print(f"\nPrimer píxel válido encontrado en (i={i}, j={j}):")
    print(f"  TBc_2d  = {float(TB[i, j]):.2f} K")
    print(f"  S_pp_dB = {float(Spp[i, j]):.2f} dB")

    # ============================================================
    # MAPITAS RÁPIDOS: SUBCONJUNTO EN TORNO A BOULDER
    # ============================================================

    # 1) Sacar nombre de la plantilla desde meta (template=...)
    tmpl_rel = None
    for m in meta:
        if isinstance(m, str) and m.startswith("template="):
            tmpl_rel = m.split("=", 1)[1]
            break

    if tmpl_rel is None:
        print("\n[WARN] No encuentro 'template=...' en meta, no puedo hacer mapitas.")
        return

    # Primero probamos en data/processed, luego en data/interim por si acaso
    cand1 = PROCESSED / tmpl_rel
    cand2 = INTERIM / tmpl_rel

    if cand1.exists():
        tmpl_path = cand1
    elif cand2.exists():
        tmpl_path = cand2
    else:
        print(f"\n[WARN] No encuentro la plantilla ni en {cand1} ni en {cand2}.")
        return

    # 2) Leer transform y CRS de la plantilla
    with rasterio.open(tmpl_path) as src:
        transform = src.transform
        crs = src.crs

    print(f"\nUsando plantilla: {tmpl_path}")
    print(f"CRS plantilla: {crs}")

    # Bbox approx de Boulder en lon/lat (mismo que usas en las descargas)
    min_lon, min_lat, max_lon, max_lat = -105.6, 39.8, -104.9, 40.2

    # 3) Transformar lon/lat -> coords del CRS de la plantilla
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    x_min, y_min = transformer.transform(min_lon, min_lat)
    x_max, y_max = transformer.transform(max_lon, max_lat)

    # 4) Pasar de coords (x, y) a índices (col, row)
    inv_transform = ~transform

    col_min_f, row_max_f = inv_transform * (x_min, y_min)
    col_max_f, row_min_f = inv_transform * (x_max, y_max)

    row_min = int(np.floor(min(row_min_f, row_max_f)))
    row_max = int(np.ceil(max(row_min_f, row_max_f)))
    col_min = int(np.floor(min(col_min_f, col_max_f)))
    col_max = int(np.ceil(max(col_min_f, col_max_f)))

    ny, nx = TB.shape
    row_min = max(row_min, 0)
    row_max = min(row_max, ny)
    col_min = max(col_min, 0)
    col_max = min(col_max, nx)

    print(f"\nSubconjunto Boulder: rows [{row_min}:{row_max}], cols [{col_min}:{col_max}]")

    if row_min >= row_max or col_min >= col_max:
        print("[WARN] Subconjunto vacío después de recortar. Algo raro con el bbox/CRS.")
        return

    TB_boulder = TB[row_min:row_max, col_min:col_max]
    Spp_boulder = Spp[row_min:row_max, col_min:col_max]

    # 5) Extent en coordenadas del CRS para que imshow quede bien
    x0, y0 = transform * (col_min, row_min)
    x1, y1 = transform * (col_max, row_max)
    extent = [x0, x1, y1, y0]  # [xmin, xmax, ymin, ymax]

    # 6) Mapas
    plt.figure(figsize=(9, 4))

    plt.subplot(1, 2, 1)
    im1 = plt.imshow(TB_boulder, extent=extent, origin="upper")
    plt.title("SMAP TBc (K) – Boulder")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.colorbar(im1, label="K")

    plt.subplot(1, 2, 2)
    im2 = plt.imshow(Spp_boulder, extent=extent, origin="upper")
    plt.title("SMAP σ⁰ (dB) – Boulder")
    plt.xlabel("x (m)")
    plt.colorbar(im2, label="dB")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
