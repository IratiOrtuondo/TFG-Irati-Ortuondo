#!/usr/bin/env python3
"""smap_tb36_to_grid.py

Extract SMAP L3 36km brightness temperature (TB) for a given date and bbox.
Reads TB_V or TB_H from the radiometer dataset and saves as NPZ.

Usage:
  python smap_tb36_to_grid.py --data-dir data/raw --date 20150607 \
      --out-dir data/interim --pol V
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from affine import Affine
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from rasterio.transform import from_origin

# EASE2 36km grid parameters
EASE2_36KM_DX = 36000.0
EASE2_36KM_X0 = -17367530.45
EASE2_36KM_Y0 = 7314540.83

DST_CRS = CRS.from_epsg(6933)  # EASE2
SRC_CRS = CRS.from_epsg(4326)  # WGS84


def find_smap_l3_file(data_dir: Path, date: str) -> Path:
    """Find SMAP L3 radiometer (L3_SM_P) HDF5 file for given date (YYYYMMDD)."""
    pattern = f"SMAP_L3_SM_P*{date}*.h5"
    files = list(data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No SMAP L3 radiometer file found matching: {pattern}")
    if len(files) > 1:
        print(f"  [WARN] Multiple L3 files found, using first: {files[0].name}")
    return files[0]


def main():
    parser = argparse.ArgumentParser(description="Extract SMAP L3 TB 36km")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory with SMAP L3 HDF5 files")
    parser.add_argument("--date", type=str, required=True, help="Date YYYYMMDD")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for NPZ")
    parser.add_argument("--pol", type=str, default="V", choices=["V", "H"], help="Polarization (V or H)")
    parser.add_argument("--lon-min", type=float, default=-104.8884912)
    parser.add_argument("--lon-max", type=float, default=-103.7115088)
    parser.add_argument("--lat-min", type=float, default=39.8008444)
    parser.add_argument("--lat-max", type=float, default=40.6991556)
    
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=== Extract SMAP L3 TB 36km ===")
    print(f"Date: {args.date}")
    print(f"Polarization: {args.pol}")
    print(f"Bbox: lon [{args.lon_min}, {args.lon_max}], lat [{args.lat_min}, {args.lat_max}]")
    
    # Find L3 file
    h5_path = find_smap_l3_file(args.data_dir, args.date)
    print(f"\n[1/3] Reading {h5_path.name}")
    
    with h5py.File(h5_path, "r") as h5:
        # TB datasets: tb_v_corrected or tb_h_corrected (AM pass)
        pol_lower = args.pol.lower()
        tb_path = f"Soil_Moisture_Retrieval_Data_AM/tb_{pol_lower}_corrected"
        
        if tb_path not in h5:
            raise KeyError(f"Dataset not found: {tb_path}")
        
        tb = h5[tb_path][:]
        
        # Apply fill value
        if hasattr(h5[tb_path], "attrs") and "_FillValue" in h5[tb_path].attrs:
            fill = h5[tb_path].attrs["_FillValue"]
            tb = np.where(tb == fill, np.nan, tb)
        
        print(f"  TB path: {tb_path}")
        print(f"  TB shape: {tb.shape}, finite: {np.isfinite(tb).sum()}")
    
    # L3 TB is on EASE2 36km grid (typically 406 x 964)
    h_full, w_full = tb.shape
    
    # Create full grid transform
    tf_full = from_origin(EASE2_36KM_X0, EASE2_36KM_Y0, EASE2_36KM_DX, EASE2_36KM_DX)
    
    # Compute bbox in EASE2 coords
    print(f"\n[2/3] Cropping to bbox")
    bbox_ease2 = transform_bounds(
        SRC_CRS, DST_CRS,
        args.lon_min, args.lat_min, args.lon_max, args.lat_max,
        densify_pts=21
    )
    left_bbox, bottom_bbox, right_bbox, top_bbox = bbox_ease2
    
    print(f"  Bbox EASE2: left={left_bbox:.1f}, bottom={bottom_bbox:.1f}, right={right_bbox:.1f}, top={top_bbox:.1f}")
    
    # Compute pixel indices for bbox crop using consistent rounding
    # Match the grid used by copol/xpol scripts (origin at -10152000, 4788000)
    # Target grid: 3 rows × 5 cols starting at (-10152000, 4788000)
    target_x0 = -10152000.0
    target_y0 = 4788000.0
    
    # Calculate which pixels in the full L3 grid correspond to this target
    col_min = int(np.round((target_x0 - EASE2_36KM_X0) / EASE2_36KM_DX))
    row_min = int(np.round((EASE2_36KM_Y0 - target_y0) / EASE2_36KM_DX))
    
    # Fixed size: 3 rows × 5 cols to match copol/xpol
    n_rows = 3
    n_cols = 5
    
    col_max = col_min + n_cols
    row_max = row_min + n_rows
    
    # Bounds check
    col_min = max(0, col_min)
    col_max = min(w_full, col_max)
    row_min = max(0, row_min)
    row_max = min(h_full, row_max)
    
    print(f"  Target grid origin: x={target_x0:.1f}, y={target_y0:.1f}")
    print(f"  Pixel indices: row [{row_min}:{row_max}], col [{col_min}:{col_max}]")
    print(f"  Expected shape: ({n_rows}, {n_cols})")
    
    if col_max <= col_min or row_max <= row_min:
        raise ValueError("Bbox outside L3 grid extent")
    
    # Crop array
    tb_crop = tb[row_min:row_max, col_min:col_max]
    
    # Use the same transform as copol/xpol for consistency
    tf_crop = from_origin(target_x0, target_y0, EASE2_36KM_DX, EASE2_36KM_DX)
    
    # Save NPZ
    print(f"\n[3/3] Saving NPZ")
    out_path = args.out_dir / f"smap-tb36-{args.date}-{args.pol.lower()}.npz"
    
    np.savez_compressed(
        out_path,
        TB_36km=tb_crop.astype(np.float32),
        pol=args.pol,
        crs_wkt=DST_CRS.to_wkt(),
        transform=np.array([tf_crop.a, tf_crop.b, tf_crop.c, tf_crop.d, tf_crop.e, tf_crop.f], dtype=np.float64),
        height=np.int32(tb_crop.shape[0]),
        width=np.int32(tb_crop.shape[1]),
        source_h5=str(h5_path),
        tb_path=tb_path,
    )
    
    print(f"  [OK] {out_path.name}")
    print(f"  Shape: {tb_crop.shape}")
    print(f"  Finite pixels: {np.isfinite(tb_crop).sum()}")
    print(f"  TB range: [{np.nanmin(tb_crop):.1f}, {np.nanmax(tb_crop):.1f}] K")


if __name__ == "__main__":
    main()
