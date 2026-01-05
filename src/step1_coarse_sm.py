#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collocate SMAP L3 soil moisture to the analysis grid.

This script extracts SMAP L3 soil moisture (coarse resolution, ~36 km),
crops it to the study area, and collocates it onto a fixed analysis grid
using nearest-neighbor resampling. The intrinsic spatial resolution of
the SMAP product is preserved.

The output is intended for validation and comparison purposes.
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
    """Creates the fixed analysis grid covering the study area.

    Args:
        n_lat: Number of latitude pixels.
        n_lon: Number of longitude pixels.

    Returns:
        Tuple of (lat_grid, lon_grid), both 2D arrays.
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
    """Loads and crops SMAP L3 soil moisture for a given date.

    Prefers the Passive-only (P) product; falls back to Active–Passive (A).

    Args:
        date: Acquisition date (YYYYMMDD).

    Returns:
        Cropped soil moisture, latitude, and longitude arrays,
        or (None, None, None) if data are unavailable.
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
        if "Soil_Moisture_Retrieval_Data_AM" in f:
            group = f["Soil_Moisture_Retrieval_Data_AM"]
        elif "Soil_Moisture_Retrieval_Data_PM" in f:
            group = f["Soil_Moisture_Retrieval_Data_PM"]
        else:
            print(f"[WARN] No soil moisture group in {smap_file.name}")
            return None, None, None

        sm = group["soil_moisture"][:]
        lat = group["latitude"][:]
        lon = group["longitude"][:]

    sm = np.where(sm == -9999.0, np.nan, sm)

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
    """Collocates coarse SMAP soil moisture onto the analysis grid.

    Nearest-neighbor resampling is used. No resolution enhancement is performed.

    Args:
        sm_coarse: Coarse-resolution SMAP soil moisture.
        lat_coarse: Latitude array (coarse).
        lon_coarse: Longitude array (coarse).
        lat_grid: Target latitude grid.
        lon_grid: Target longitude grid.

    Returns:
        Soil moisture collocated to the analysis grid.
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
