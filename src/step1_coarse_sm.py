#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collocate SMAP L3 soil moisture to the analysis grid.

This module extracts SMAP L3 soil moisture (coarse resolution, ~36 km),
crops it to a user-defined study area, and collocates it onto a fixed
analysis grid using nearest-neighbour resampling.

Purpose and assumptions:
- The script uses the Passive-only product when available and falls back
  to Active–Passive when necessary.
- Missing/fill values are represented by -9999 in the product and are
  replaced with ``np.nan`` before processing.
- Nearest-neighbour resampling is used to preserve the intrinsic coarse
  resolution; no upscaling or interpolation is performed to add detail.

Outputs are simple NPZ files intended for validation and comparison.

Notes for maintainers:
- The code is intentionally simple and deterministic; it is suitable for
  unit testing and integration into larger pipelines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DATES = ["20150607", "20150610", "20150615", "20150618", "20150620"]

# Local raw/processed directories (project-specific)
RAW_DIR = Path(r"c:\Users\ortuo\tfgirati\tfg-nisar\data\raw")
OUTPUT_DIR = Path(r"c:\Users\ortuo\tfgirati\tfg-nisar\data\processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Study area bounds (lon/lat, EPSG:4326)
LON_MIN = -104.8884912
LON_MAX = -103.7115088
LAT_MIN = 39.8008444
LAT_MAX = 40.6991556

CRS_WGS84 = CRS.from_epsg(4326)


# -----------------------------------------------------------------------------
# Grid utilities
# -----------------------------------------------------------------------------


def create_analysis_grid(
    n_lat: int = 30,
    n_lon: int = 39,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a fixed analysis grid covering the study area.

    The returned arrays follow the common raster convention: `lat_grid` and
    `lon_grid` are 2D arrays with shape (n_lat, n_lon). `lat` is created using
    `np.linspace(LAT_MAX, LAT_MIN, n_lat)` so that the first row corresponds
    to the northern edge of the study area.

    Args:
        n_lat: Number of latitude pixels.
        n_lon: Number of longitude pixels.

    Returns:
        Tuple (lat_grid, lon_grid) both 2D float arrays suitable for plotting
        and reprojecting results.
    """
    lat = np.linspace(LAT_MAX, LAT_MIN, n_lat)
    lon = np.linspace(LON_MIN, LON_MAX, n_lon)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    return lat_grid, lon_grid


# -----------------------------------------------------------------------------
# SMAP L3 extraction
# -----------------------------------------------------------------------------


def load_smap_l3_soil_moisture(
    date: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | Tuple[None, None, None]:
    """Load and crop SMAP L3 soil moisture for a given YYYYMMDD date.

    Behavior / heuristics:
    - Checks for a Passive-only file first (filename pattern used in this
      project); if absent, it tries an Active–Passive filename.
    - Within the HDF5 file, looks for AM or PM retrieval groups and uses
      the first found. If neither group exists the function returns None.
    - Replaces the commonly used fill value -9999.0 with ``np.nan``.
    - Crops the arrays to the configured study-area bounding box and returns
      the cropped sm, lat, lon arrays.

    Args:
        date: Date string YYYYMMDD.

    Returns:
        (sm_cropped, lat_cropped, lon_cropped) or (None, None, None) when
        data are missing or no samples fall in the study area.
    """
    file_p = RAW_DIR / f"SMAP_L3_SM_P_{date}_R19240_001.h5"
    file_a = RAW_DIR / f"SMAP_L3_SM_A_{date}_R13080_001.h5"

    if file_p.exists():
        smap_file = file_p
    elif file_a.exists():
        smap_file = file_a
    else:
        print(f"[WARN] No SMAP L3 file found for {date}")
        return None, None, None

    with h5py.File(smap_file, "r") as f:
        # Prefer AM group if present; otherwise try PM
        if "Soil_Moisture_Retrieval_Data_AM" in f:
            group = f["Soil_Moisture_Retrieval_Data_AM"]
        elif "Soil_Moisture_Retrieval_Data_PM" in f:
            group = f["Soil_Moisture_Retrieval_Data_PM"]
        else:
            print(f"[WARN] No soil moisture group in {smap_file.name}")
            return None, None, None

        # Standard dataset names in SMAP L3 groups
        sm = group["soil_moisture"][:]
        lat = group["latitude"][:]
        lon = group["longitude"][:]

    # Replace SMAP fill values with NaN before any computations
    sm = np.where(sm == -9999.0, np.nan, sm)

    # Boolean mask of points inside the study area (inclusive bounds)
    mask = (
        (lat >= LAT_MIN) & (lat <= LAT_MAX) &
        (lon >= LON_MIN) & (lon <= LON_MAX)
    )

    if not np.any(mask):
        print("[WARN] No SMAP pixels inside study area")
        return None, None, None

    rows, cols = np.where(mask)
    r0, r1 = rows.min(), rows.max() + 1
    c0, c1 = cols.min(), cols.max() + 1

    # Return contiguous crop (r0:r1, c0:c1) to preserve original array shape
    return (
        sm[r0:r1, c0:c1],
        lat[r0:r1, c0:c1],
        lon[r0:r1, c0:c1],
    )


# -----------------------------------------------------------------------------
# Collocation
# -----------------------------------------------------------------------------


def collocate_to_analysis_grid(
    sm_coarse: np.ndarray,
    lat_coarse: np.ndarray,
    lon_coarse: np.ndarray,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
) -> np.ndarray:
    """Collocate coarse SMAP soil moisture onto the analysis grid.

    Uses nearest-neighbour resampling (rasterio.reproject with Resampling.nearest)
    which preserves the coarse product's cell values rather than interpolating.

    The function builds simple bounds-based transforms for the source and
    destination grids using `from_bounds` and relies on CRS=EPSG:4326.

    Args:
        sm_coarse: Coarse-resolution SMAP soil moisture array.
        lat_coarse: Latitude array corresponding to sm_coarse.
        lon_coarse: Longitude array corresponding to sm_coarse.
        lat_grid: Target latitude grid (2D).
        lon_grid: Target longitude grid (2D).

    Returns:
        The collocated soil moisture on the analysis grid as float32 with
        np.nan where no data are present.
    """
    src_transform = from_bounds(
        lon_coarse.min(),
        lat_coarse.min(),
        lon_coarse.max(),
        lat_coarse.max(),
        sm_coarse.shape[1],
        sm_coarse.shape[0],
    )

    dst_transform = from_bounds(
        LON_MIN,
        LAT_MIN,
        LON_MAX,
        LAT_MAX,
        lon_grid.shape[1],
        lat_grid.shape[0],
    )

    sm_on_grid = np.full(lat_grid.shape, np.nan, dtype=np.float32)

    reproject(
        source=sm_coarse.astype(np.float32),
        destination=sm_on_grid,
        src_transform=src_transform,
        src_crs=CRS_WGS84,
        dst_transform=dst_transform,
        dst_crs=CRS_WGS84,
        resampling=Resampling.nearest,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )

    return sm_on_grid


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    """Top-level entry point: run collocation for configured dates and save NPZs."""
    print("=" * 70)
    print("SMAP L3 COARSE SOIL MOISTURE COLLOCATION")
    print("=" * 70)

    lat_grid, lon_grid = create_analysis_grid()

    for date in DATES:
        print(f"\nProcessing {date}...")

        sm, lat, lon = load_smap_l3_soil_moisture(date)
        if sm is None:
            continue

        sm_on_grid = collocate_to_analysis_grid(
            sm, lat, lon, lat_grid, lon_grid
        )

        out_file = OUTPUT_DIR / f"smap_l3_sm_coarse_on_grid_{date}.npz"
        np.savez(
            out_file,
            sm_coarse_on_grid=sm_on_grid,
            latitude=lat_grid,
            longitude=lon_grid,
            source_product="SMAP_L3",
            nominal_resolution_km=36,
        )

        print(f"  Saved: {out_file.name}")

    print("\n✓ Collocation complete")


if __name__ == "__main__":
    main()
