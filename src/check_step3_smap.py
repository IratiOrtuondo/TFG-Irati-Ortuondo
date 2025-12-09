#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
check_step3_smap_colorado.py — Inspect and plot Step 3 (β-only) results
ONLY over the Colorado ROI, cropped from the global SMAP grid.

Expected input:
    data/interim/step3_colorado.npz   (or any Step 3 NPZ)

Contains:
    - beta_K_per_dB : 2D array (H, W) of β(C) [K/dB]
    - n_samples     : 2D array (H, W)
    - r2            : 2D array (H, W)
    - crs_wkt, transform, height, width, files, meta
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

# <-- change this if your Step 3 file has another name
NPZ_NAME = "step3_colorado.npz"

# ---------------------------------------------------------
# Colorado ROI in lon/lat (same bbox you used before)
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
    """
    Convert a lon/lat bounding box to row/col indices on the SMAP grid.

    Returns:
        row_min, row_max, col_min, col_max  (int indices for slicing)
    """
    crs_grid = CRS.from_wkt(crs_wkt)
    crs_ll = CRS.from_epsg(4326)  # WGS84 lon/lat

    transformer = Transformer.from_crs(crs_ll, crs_grid, always_xy=True)

    flat = np.array(transform_arr).ravel()
    if flat.size < 6:
        raise ValueError("transform array has unexpected size; cannot build Affine")
    a, b, c, d, e, f = flat[:6]
    transform = Affine(a, b, c, d, e, f)

    # bbox corners in lon/lat
    corners_ll = [
        (lon_min, lat_min),
        (lon_min, lat_max),
        (lon_max, lat_min),
        (lon_max, lat_max),
    ]

    rows, cols = [], []
    for lon, lat in corners_ll:
        x, y = transformer.transform(lon, lat)  # to grid CRS
        col, row = ~transform * (x, y)          # to pixel indices (float)
        rows.append(row)
        cols.append(col)

    row_min = int(np.floor(min(rows)))
    row_max = int(np.ceil(max(rows))) + 1  # +1 so slicing is inclusive
    col_min = int(np.floor(min(cols)))
    col_max = int(np.ceil(max(cols))) + 1

    return row_min, row_max, col_min, col_max


def main() -> None:
    # ---------------------------------------------
    # Load global Step 3 output
    # ---------------------------------------------
    npz_path = INTERIM / NPZ_NAME
    if not npz_path.exists():
        raise FileNotFoundError(f"Cannot find {npz_path}")

    npz = np.load(npz_path, allow_pickle=True)

    beta = npz["beta_K_per_dB"]
    n_samples = npz["n_samples"]
    r2 = npz["r2"]
    crs_wkt = str(npz["crs_wkt"])
    transform_arr = npz["transform"]

    print(f"Loaded: {npz_path}")
    print("GLOBAL beta shape:", beta.shape)

    # ---------------------------------------------
    # Convert lon/lat ROI -> row/col indices
    # ---------------------------------------------
    row_min, row_max, col_min, col_max = lonlat_to_rowcol(
        LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, crs_wkt, transform_arr
    )

    print("\nROI indices (row/col):")
    print(f"  rows: {row_min} : {row_max}")
    print(f"  cols: {col_min} : {col_max}")

    # clip to array bounds (safety)
    H, W = beta.shape
    row_min = max(0, row_min)
    col_min = max(0, col_min)
    row_max = min(H, row_max)
    col_max = min(W, col_max)

    # ---------------------------------------------
    # Crop to Colorado ROI
    # ---------------------------------------------
    beta_roi = beta[row_min:row_max, col_min:col_max]
    n_samples_roi = n_samples[row_min:row_max, col_min:col_max]
    r2_roi = r2[row_min:row_max, col_min:col_max]

    print("\nROI shapes:")
    print("  beta_roi     :", beta_roi.shape)
    print("  n_samples_roi:", n_samples_roi.shape)
    print("  r2_roi       :", r2_roi.shape)

    # ---------------------------------------------
    # Plot β(C) over Colorado ROI only (no QA)
    # ---------------------------------------------
    plt.figure(figsize=(5, 5))
    im = plt.imshow(beta_roi, origin="upper")
    plt.colorbar(im, label="β(C) [K/dB]")
    plt.title("β(C) — Colorado ROI")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
