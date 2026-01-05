#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import h5py

XPOL_KEYS = ("vh", "hv", "xpol", "cross")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--contains", default=None, help="filtra por substring (case-insensitive)")
    args = ap.parse_args()

    with h5py.File(args.h5, "r") as h5:
        paths = []

        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                paths.append((name, obj.shape, obj.dtype))
        h5.visititems(visitor)

    # 1) imprime TODO lo relacionado con XPOL
    print("\n[DATASETS QUE PARECEN XPOL (vh/hv/xpol/cross)]")
    found = 0
    for name, shape, dtype in sorted(paths, key=lambda x: x[0].lower()):
        low = name.lower()
        if any(k in low for k in XPOL_KEYS):
            print(f"  - {name}  shape={shape} dtype={dtype}")
            found += 1
    if found == 0:
        print("  (ninguno)")

    # 2) imprime TODO lo relacionado con sigma0 (por si el naming es raro)
    print("\n[DATASETS QUE CONTIENEN 'sigma0' O 's0']")
    found2 = 0
    for name, shape, dtype in sorted(paths, key=lambda x: x[0].lower()):
        low = name.lower()
        if ("sigma0" in low) or ("/s0" in low) or low.endswith("s0"):
            print(f"  - {name}  shape={shape} dtype={dtype}")
            found2 += 1
    if found2 == 0:
        print("  (ninguno)")

    # 3) filtro opcional
    if args.contains:
        key = args.contains.lower()
        print(f"\n[DATASETS QUE CONTIENEN '{args.contains}']")
        f = 0
        for name, shape, dtype in sorted(paths, key=lambda x: x[0].lower()):
            if key in name.lower():
                print(f"  - {name}  shape={shape} dtype={dtype}")
                f += 1
        if f == 0:
            print("  (ninguno)")

# =========================
# QUICK XPOL PLOT (aft)
# =========================
import numpy as np
import matplotlib.pyplot as plt
import h5py

H5 = r"..\data\raw\SMAP_L1C_S0_HIRES_02252_D_20150704T132507_R11850_001.h5"
XPOL_PATH = "Sigma0_Data/cell_sigma0_xpol_aft"

with h5py.File(H5, "r") as h5:
    xpol = np.array(h5[XPOL_PATH][...], dtype=np.float64)
    lon  = np.array(h5["Sigma0_Data/cell_lon"][...], dtype=np.float64)
    lat  = np.array(h5["Sigma0_Data/cell_lat"][...], dtype=np.float64)

# fills típicos
for fv in [0, -9999, -999, -32768, 65535]:
    xpol[xpol == fv] = np.nan

m = np.isfinite(xpol) & np.isfinite(lon) & np.isfinite(lat)
print("valid points:", np.count_nonzero(m), "/", xpol.size)
print("xpol min/max/mean:", np.nanmin(xpol), np.nanmax(xpol), np.nanmean(xpol))

idx = np.where(m.ravel())[0]
if idx.size > 200_000:
    idx = np.random.choice(idx, 200_000, replace=False)

plt.figure()
plt.title("XPOL aft (lon/lat)")
plt.scatter(lon.ravel()[idx], lat.ravel()[idx], s=1, c=xpol.ravel()[idx])
plt.xlabel("lon"); plt.ylabel("lat")
plt.colorbar(label="sigma0 xpol")
plt.show()

if __name__ == "__main__":
    main()
