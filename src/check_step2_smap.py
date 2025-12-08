#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import rasterio
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"


def main():
    # Prefer the standard step2 output.
    npz_path = INTERIM / "aligned_step2_smap.npz"

    if not npz_path.exists():
        alt = INTERIM / "aligned_step1.npz"

        if alt.exists():
            print(f"[WARN] {npz_path.name} not found. Using fallback {alt.name}.")
            npz_path = alt
        else:
            # NEW: look for files like aligned-smap-YYYYMMDD.npz
            aligned_files = sorted(INTERIM.glob("aligned-smap-*.npz"))
            if aligned_files:
                # Use the last one (latest date by filename)
                npz_path = aligned_files[-1]
                print(
                    f"[WARN] {INTERIM / 'aligned_step2_smap.npz'} and {alt.name} not found. "
                    f"Using last aligned-smap-*.npz found: {npz_path.name}"
                )
            else:
                # If none of the above exist, raise a helpful error listing .npz files
                files = list(INTERIM.glob("*.npz"))
                sample = "none"
                if files:
                    sample = ", ".join(f.name for f in files[:10])
                raise FileNotFoundError(
                    f"Could not find {INTERIM / 'aligned_step2_smap.npz'} or fallback {alt}. "
                    f".npz files in {INTERIM}: {sample}"
                )

    data = np.load(npz_path, allow_pickle=True)

    TB = data["TBc_2d"]   # radiometer TB [K]
    Spp = data["S_pp_dB"]  # radar sigma0 [dB]
    meta = data["meta"]

    print("Meta:")
    for m in meta:
        print("  -", m)

    print("\nShapes:")
    print("  TBc_2d :", TB.shape)
    print("  S_pp_dB:", Spp.shape)

    # Quick statistics helper
    def stats(arr, name):
        mask = np.isfinite(arr)
        if not mask.any():
            print(f"  {name}: all NaN")
            return
        vals = arr[mask]
        print(
            f"  {name}: min={vals.min():.2f}, max={vals.max():.2f}, "
            f"mean={vals.mean():.2f}"
        )

    print("\nStats:")
    stats(TB, "TBc_2d (K)")
    stats(Spp, "S_pp_dB (dB)")

    # Look for a pixel where both TB and sigma0 are valid
    mask_valid = np.isfinite(TB) & np.isfinite(Spp)
    if not mask_valid.any():
        print("\n[WARN] No pixel has both valid TB and σ0.")
        return

    i, j = np.argwhere(mask_valid)[0]
    print(f"\nFirst valid pixel found at (i={i}, j={j}):")
    print(f"  TBc_2d  = {float(TB[i, j]):.2f} K")
    print(f"  S_pp_dB = {float(Spp[i, j]):.2f} dB")

    # ============================================================
    # QUICK MAPS: SUBSET AROUND BOULDER
    # ============================================================

    # 1) Get template name from meta (template=...)
    tmpl_rel = None
    for m in meta:
        if isinstance(m, str) and m.startswith("template="):
            tmpl_rel = m.split("=", 1)[1]
            break

    if tmpl_rel is None:
        print("\n[WARN] 'template=...' not found in meta, cannot make maps.")
        return

    # First try data/processed, then data/interim
    cand1 = PROCESSED / tmpl_rel
    cand2 = INTERIM / tmpl_rel

    if cand1.exists():
        tmpl_path = cand1
    elif cand2.exists():
        tmpl_path = cand2
    else:
        print(f"\n[WARN] Template not found in {cand1} nor in {cand2}.")
        return

    # 2) Read transform and CRS from the template
    with rasterio.open(tmpl_path) as src:
        transform = src.transform
        crs = src.crs

    print(f"\nUsing template: {tmpl_path}")
    print(f"Template CRS: {crs}")

    # Boulder 100 km × 100 km (same bbox as in smap.py --bbox)
    min_lon, min_lat, max_lon, max_lat = -104.8885, 39.8008, -103.7115, 40.6992


    # 3) Transform lon/lat -> template CRS coordinates
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    x_min, y_min = transformer.transform(min_lon, min_lat)
    x_max, y_max = transformer.transform(max_lon, max_lat)

    # 4) Convert from (x, y) to (col, row)
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

    print(f"\nBoulder subset: rows [{row_min}:{row_max}], cols [{col_min}:{col_max}]")

    if row_min >= row_max or col_min >= col_max:
        print("[WARN] Empty subset after clipping. Something is odd with the bbox/CRS.")
        return

    TB_boulder = TB[row_min:row_max, col_min:col_max]
    Spp_boulder = Spp[row_min:row_max, col_min:col_max]

    # 5) Extent in CRS coordinates so imshow is correctly georeferenced
    x0, y0 = transform * (col_min, row_min)
    x1, y1 = transform * (col_max, row_max)
    extent = [x0, x1, y1, y0]  # [xmin, xmax, ymin, ymax]

    # 6) Plots
    plt.figure(figsize=(9, 4))

    plt.subplot(1, 2, 1)
    im1 = plt.imshow(TB_boulder, extent=extent, origin="upper")
    plt.title("SMAP TBc (K) – Colorado")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.colorbar(im1, label="K")

    plt.subplot(1, 2, 2)
    im2 = plt.imshow(Spp_boulder, extent=extent, origin="upper")
    plt.title("SMAP σ⁰ (dB) – Colorado")
    plt.xlabel("x (m)")
    plt.colorbar(im2, label="dB")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
