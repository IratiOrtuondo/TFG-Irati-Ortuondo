"""Plot native XPOL NPZ for a given date.
Usage: python plot_xpol_native_date.py 20150503
Writes: data/interim/xpol_native_<date>.png
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print('Usage: python plot_xpol_native_date.py YYYYMMDD')
    raise SystemExit(1)

DATE = sys.argv[1]
NPZ = Path('..') / 'data' / 'interim' / f'aligned-smap-xpol-{DATE}-native.npz'
OUT = Path('..') / 'data' / 'interim' / f'xpol_native_{DATE}.png'

if not NPZ.exists():
    print('NPZ not found:', NPZ)
    raise SystemExit(1)

a = np.load(NPZ, allow_pickle=True)
if 'S_xpol_dB_native' not in a.files:
    print('Key S_xpol_dB_native not found in', NPZ)
    print('Available keys:', a.files)
    raise SystemExit(1)

arr = a['S_xpol_dB_native']
print('Loaded', NPZ, 'shape=', arr.shape, 'finite=', int(np.isfinite(arr).sum()))

# For large arrays, decimate for quick plotting to keep file sizes reasonable
h, w = arr.shape
max_pixels = 2000  # target max dimension for image
step_h = max(1, h // max_pixels)
step_w = max(1, w // max_pixels)
arr_small = arr[::step_h, ::step_w]

arr_plot = np.where(np.isfinite(arr_small), arr_small, np.nan)

plt.figure(figsize=(8,6))
im = plt.imshow(arr_plot, cmap='RdYlBu_r')
plt.colorbar(im, label='S_xpol (dB)')
plt.title(f'XPOL native {DATE} shape={arr.shape} decimated={arr_small.shape} finite={int(np.isfinite(arr).sum())}')
# overlay user's bbox from NPZ meta if available
if 'meta' in a.files:
    meta = a['meta'].tolist() if isinstance(a['meta'], np.ndarray) else a['meta']
    bbox = None
    if isinstance(meta, dict) and 'bbox_lonlat' in meta:
        bbox = meta['bbox_lonlat']
    elif isinstance(meta, (list, tuple)):
        # try to find a nested dict
        try:
            md = dict(meta)
            if 'bbox_lonlat' in md:
                bbox = md['bbox_lonlat']
        except Exception:
            bbox = None

    if bbox is not None:
        try:
            # if transform_native exists, compute pixel coords
            if 'transform_native' in a.files:
                tr = a['transform_native']
                from affine import Affine
                T = Affine(*tr.tolist()) if hasattr(tr, 'tolist') else Affine(*tr)
                lonmin, latmin, lonmax, latmax = bbox
                # convert lon/lat to col/row
                inv = ~T
                c0, r0 = inv * (lonmin, latmax)
                c1, r1 = inv * (lonmax, latmin)
                # scale according to decimation
                c0s, c1s = int(c0 // step_w), int(c1 // step_w)
                r0s, r1s = int(r0 // step_h), int(r1 // step_h)
                import matplotlib.patches as patches
                rect = patches.Rectangle((min(c0s, c1s), min(r0s, r1s)), abs(c1s - c0s), abs(r1s - r0s), linewidth=1.5, edgecolor='yellow', facecolor='none')
                plt.gca().add_patch(rect)
            else:
                # fallback: try lat_native/lon_native arrays to find index ranges
                if 'lat_native' in a.files and 'lon_native' in a.files:
                    lat_n = a['lat_native']
                    lon_n = a['lon_native']
                    lonmin, latmin, lonmax, latmax = bbox
                    # find indices within bbox
                    mask = (lon_n >= lonmin) & (lon_n <= lonmax) & (lat_n >= latmin) & (lat_n <= latmax)
                    if mask.any():
                        rows, cols = np.where(mask)
                        r0s, r1s = rows.min() // step_h, rows.max() // step_h
                        c0s, c1s = cols.min() // step_w, cols.max() // step_w
                        import matplotlib.patches as patches
                        rect = patches.Rectangle((c0s, r0s), c1s - c0s, r1s - r0s, linewidth=1.5, edgecolor='yellow', facecolor='none')
                        plt.gca().add_patch(rect)
        except Exception as _e:
            print('Could not overlay bbox:', _e)
plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=150)
print('WROTE', OUT)
