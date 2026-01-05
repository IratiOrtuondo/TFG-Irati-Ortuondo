#!/usr/bin/env python3
"""
plot_gamma_mask.py
Plot finite-mask of gamma_K_per_dB from step3_beta_gamma.npz and overlay the Colorado ROI box.
"""
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from affine import Affine
from pyproj import CRS, Transformer
from matplotlib.patches import Rectangle

INTERIM = Path(__file__).resolve().parents[1] / 'data' / 'interim'
OUTDIR = INTERIM / 'plots'
OUTDIR.mkdir(parents=True, exist_ok=True)
npz = np.load(INTERIM / 'step3_beta_gamma.npz', allow_pickle=True)
H = int(npz['height']); W = int(npz['width'])
crs_wkt = str(npz['crs_wkt'])
transform_arr = np.array(npz['transform']).ravel()[:6]
a,b,c,d,e,f = transform_arr
transform = Affine(a,b,c,d,e,f)

gamma = np.asarray(npz['gamma_K_per_dB'])
mask = np.isfinite(gamma).astype(int)

# ROI lon/lat
LON_MIN, LAT_MIN, LON_MAX, LAT_MAX = -104.8885, 39.8008, -103.7115, 40.6992
crs_grid = CRS.from_wkt(crs_wkt)
crs_ll = CRS.from_epsg(4326)
transformer = Transformer.from_crs(crs_ll, crs_grid, always_xy=True)

# convert bbox to pixel coords
corners = [(LON_MIN, LAT_MIN), (LON_MIN, LAT_MAX), (LON_MAX, LAT_MIN), (LON_MAX, LAT_MAX)]
rows, cols = [], []
for lon, lat in corners:
    x, y = transformer.transform(lon, lat)
    col, row = ~transform * (x, y)
    rows.append(row); cols.append(col)
import math
r0 = int(math.floor(min(rows))); r1 = int(math.ceil(max(rows)))+1
c0 = int(math.floor(min(cols))); c1 = int(math.ceil(max(cols)))+1
row_min, row_max = max(0, min(r0, r1)), min(H, max(r0, r1))
col_min, col_max = max(0, min(c0, c1)), min(W, max(c0, c1))

plt.figure(figsize=(8,6))
plt.imshow(mask, origin='upper', cmap='gray')
plt.title('Finite gamma mask (1=finite)')
plt.colorbar()
rect = Rectangle((col_min, row_min), col_max - col_min, row_max - row_min,
                 linewidth=2, edgecolor='r', facecolor='none')
plt.gca().add_patch(rect)
plt.tight_layout()
out = OUTDIR / 'gamma_mask.png'
plt.savefig(out, dpi=200)
print('[OK] Saved', out)
plt.close()
