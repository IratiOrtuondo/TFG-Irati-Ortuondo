#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
check_step3_smap_colorado.py

Inspect and plot Step 3 β(C) and γ(C) results
ONLY over the Colorado ROI, cropped from the global SMAP grid.

NO QA filtering except removal of clearly non-physical values.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from affine import Affine
from pyproj import CRS, Transformer

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"

# <-- adjust if needed
# Use the bbox Step3 output by default (this is the working grid for the Colorado ROI)
NPZ_NAME = "step3_beta_gamma_bbox.npz"

# optional output folder for images
OUT_DIR = INTERIM / "plots"
SAVE_PNG = True

# ---------------------------------------------------------
# Colorado ROI in lon/lat
# ---------------------------------------------------------
LON_MIN = -104.8885
LAT_MIN = 39.8008
LON_MAX = -103.7115
LAT_MAX = 40.6992


def lonlat_to_rowcol(
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
    crs_wkt: str,
    transform_arr: np.ndarray,
):
    """Convert lon/lat bounding box to row/col indices (min/max inclusive-ish)."""
    crs_grid = CRS.from_wkt(crs_wkt)
    crs_ll = CRS.from_epsg(4326)

    transformer = Transformer.from_crs(crs_ll, crs_grid, always_xy=True)

    a, b, c, d, e, f = np.array(transform_arr).ravel()[:6]
    transform = Affine(a, b, c, d, e, f)

    corners_ll = [
        (lon_min, lat_min),
        (lon_min, lat_max),
        (lon_max, lat_min),
        (lon_max, lat_max),
    ]

    rows, cols = [], []
    for lon, lat in corners_ll:
        x, y = transformer.transform(lon, lat)
        col, row = ~transform * (x, y)
        rows.append(row)
        cols.append(col)

    r0 = int(np.floor(min(rows)))
    r1 = int(np.ceil(max(rows))) + 1
    c0 = int(np.floor(min(cols)))
    c1 = int(np.ceil(max(cols))) + 1

    # handle negative pixel sizes / flipped transforms robustly
    row_min, row_max = (min(r0, r1), max(r0, r1))
    col_min, col_max = (min(c0, c1), max(c0, c1))
    return row_min, row_max, col_min, col_max


def clean_for_plot(arr: np.ndarray, abs_max: float = 50.0) -> np.ndarray:
    """
    Minimal sanity cleaning:
      - remove NaN / inf
      - remove absurdly large values
    """
    out = arr.astype(np.float32, copy=True)
    out[~np.isfinite(out)] = np.nan
    out[np.abs(out) > abs_max] = np.nan
    return out


def plot_map(data: np.ndarray, title: str, cbar_label: str, out_path: Path | None = None):
    plt.figure(figsize=(6, 5))

    # display scaling that doesn't "filter" values, only sets the color stretch
    finite = np.isfinite(data)
    if np.any(finite):
        vmin, vmax = np.nanpercentile(data, [2, 98])
    else:
        vmin, vmax = None, None

    im = plt.imshow(data, origin="upper", vmin=vmin, vmax=vmax)
    plt.colorbar(im, label=cbar_label)
    plt.title(title)
    plt.tight_layout()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200)
        print(f"[OK] Saved: {out_path}")

    plt.show()


def main() -> None:
    # ---------------------------------------------
    # Load Step 3 output
    # ---------------------------------------------
    npz_path = INTERIM / NPZ_NAME
    if not npz_path.exists():
        raise FileNotFoundError(f"Cannot find {npz_path}")

    npz = np.load(npz_path, allow_pickle=True)

    # expected keys
    beta = np.asarray(npz["beta_K_per_dB"], dtype=np.float32)
    gamma = np.asarray(npz["gamma_K_per_dB"], dtype=np.float32)
    crs_wkt = str(npz["crs_wkt"])
    transform_arr = npz["transform"]

    print(f"[INFO] Loaded {npz_path}")
    print("[INFO] Global beta shape :", beta.shape)
    print("[INFO] Global gamma shape:", gamma.shape)

    # ---------------------------------------------
    # Convert ROI to row/col
    # ---------------------------------------------
    row_min, row_max, col_min, col_max = lonlat_to_rowcol(
        LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, crs_wkt, transform_arr
    )

    H, W = beta.shape
    row_min = max(0, row_min)
    col_min = max(0, col_min)
    row_max = min(H, row_max)
    col_max = min(W, col_max)

    if row_min >= row_max or col_min >= col_max:
        raise RuntimeError(
            f"ROI crop is empty. Got rows [{row_min}:{row_max}] cols [{col_min}:{col_max}]. "
            "Check ROI coords or CRS/transform."
        )

    # ---------------------------------------------
    # Crop to Colorado ROI
    # ---------------------------------------------
    beta_roi = beta[row_min:row_max, col_min:col_max]
    gamma_roi = gamma[row_min:row_max, col_min:col_max]

    print("[INFO] ROI beta shape :", beta_roi.shape)
    print("[INFO] ROI gamma shape:", gamma_roi.shape)

    # ---------------------------------------------
    # Minimal sanity cleaning (ONLY extreme nonsense)
    # ---------------------------------------------
    beta_plot = clean_for_plot(beta_roi, abs_max=50.0)
    gamma_plot = clean_for_plot(gamma_roi, abs_max=50.0)

    # ---------------------------------------------
    # Plot maps
    # ---------------------------------------------
    beta_out = (OUT_DIR / "beta_colorado.png") if SAVE_PNG else None
    gamma_out = (OUT_DIR / "gamma_colorado.png") if SAVE_PNG else None

    plot_map(beta_plot, "β(C) — Colorado ROI", "β (K / dB)", beta_out)
    plot_map(gamma_plot, "γ(C) — Colorado ROI", "γ (K / dB)", gamma_out)


if __name__ == "__main__":
    main()
