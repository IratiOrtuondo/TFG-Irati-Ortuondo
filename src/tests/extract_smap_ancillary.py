#!/usr/bin/env python3
"""extract_smap_ancillary.py

Extract ancillary data from SMAP L3 (L3_SM_P for radiometer ancillary) for RTM soil moisture retrieval.
Extracts: surface temperature, vegetation water content, roughness, incidence angle, etc.

Usage:
  python extract_smap_ancillary.py --data-dir data/raw --date 20150607 --out-dir data/interim
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
    parser = argparse.ArgumentParser(description="Extract SMAP L3 ancillary data for RTM")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory with SMAP L3 HDF5 files")
    parser.add_argument("--date", type=str, required=True, help="Date YYYYMMDD")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for NPZ")
    parser.add_argument("--lon-min", type=float, default=-104.8884912)
    parser.add_argument("--lon-max", type=float, default=-103.7115088)
    parser.add_argument("--lat-min", type=float, default=39.8008444)
    parser.add_argument("--lat-max", type=float, default=40.6991556)
    
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=== Extract SMAP L3 Ancillary Data ===")
    print(f"Date: {args.date}")
    print(f"Bbox: lon [{args.lon_min}, {args.lon_max}], lat [{args.lat_min}, {args.lat_max}]")
    
    # Find L3 file
    h5_path = find_smap_l3_file(args.data_dir, args.date)
    print(f"\n[1/3] Reading {h5_path.name}")
    
    ancillary_data = {}
    
    with h5py.File(h5_path, "r") as h5:
        # Key ancillary datasets for RTM (AM pass)
        datasets_to_extract = {
            'surface_temperature': 'Soil_Moisture_Retrieval_Data_AM/surface_temperature',
            'vegetation_water_content': 'Soil_Moisture_Retrieval_Data_AM/vegetation_water_content',
            'roughness_coefficient': 'Soil_Moisture_Retrieval_Data_AM/roughness_coefficient',
            'boresight_incidence': 'Soil_Moisture_Retrieval_Data_AM/boresight_incidence',
            'vegetation_opacity': 'Soil_Moisture_Retrieval_Data_AM/vegetation_opacity',
            'albedo': 'Soil_Moisture_Retrieval_Data_AM/albedo',
        }
        
        for key, path in datasets_to_extract.items():
            if path not in h5:
                print(f"  [WARN] Dataset not found: {path}")
                continue
            
            data = h5[path][:]
            
            # Apply fill value
            if hasattr(h5[path], "attrs") and "_FillValue" in h5[path].attrs:
                fill = h5[path].attrs["_FillValue"]
                data = np.where(data == fill, np.nan, data)
            
            ancillary_data[key] = data
            print(f"  {key}: shape {data.shape}, finite: {np.isfinite(data).sum()}")
    
    # L3 data is on EASE2 36km grid (typically 406 x 964)
    first_key = list(ancillary_data.keys())[0]
    h_full, w_full = ancillary_data[first_key].shape
    
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
    
    # Compute pixel indices for bbox crop
    col_min = max(0, int(np.floor((left_bbox - EASE2_36KM_X0) / EASE2_36KM_DX)))
    col_max = min(w_full, int(np.ceil((right_bbox - EASE2_36KM_X0) / EASE2_36KM_DX)))
    row_min = max(0, int(np.floor((EASE2_36KM_Y0 - top_bbox) / EASE2_36KM_DX)))
    row_max = min(h_full, int(np.ceil((EASE2_36KM_Y0 - bottom_bbox) / EASE2_36KM_DX)))
    
    if col_max <= col_min or row_max <= row_min:
        raise ValueError("Bbox outside L3 grid extent")
    
    # Crop all arrays
    ancillary_cropped = {}
    for key, data in ancillary_data.items():
        data_crop = data[row_min:row_max, col_min:col_max]
        ancillary_cropped[key] = data_crop.astype(np.float32)
        print(f"  {key}: cropped shape {data_crop.shape}, finite: {np.isfinite(data_crop).sum()}")
    
    # Compute cropped transform
    x0_crop = EASE2_36KM_X0 + col_min * EASE2_36KM_DX
    y0_crop = EASE2_36KM_Y0 - row_min * EASE2_36KM_DX
    tf_crop = from_origin(x0_crop, y0_crop, EASE2_36KM_DX, EASE2_36KM_DX)
    
    # Save NPZ
    print(f"\n[3/3] Saving NPZ")
    out_path = args.out_dir / f"smap-ancillary-{args.date}.npz"
    
    save_dict = {
        **ancillary_cropped,
        'crs_wkt': DST_CRS.to_wkt(),
        'transform': np.array([tf_crop.a, tf_crop.b, tf_crop.c, tf_crop.d, tf_crop.e, tf_crop.f], dtype=np.float64),
        'height': np.int32(list(ancillary_cropped.values())[0].shape[0]),
        'width': np.int32(list(ancillary_cropped.values())[0].shape[1]),
        'source_h5': str(h5_path),
        'date': args.date,
    }
    
    np.savez_compressed(out_path, **save_dict)
    
    print(f"  [OK] {out_path.name}")
    print(f"  Shape: {save_dict['height']} x {save_dict['width']}")
    print(f"\nExtracted ancillary variables:")
    for key in ancillary_cropped.keys():
        data = ancillary_cropped[key]
        print(f"  {key}: range [{np.nanmin(data):.2f}, {np.nanmax(data):.2f}]")


if __name__ == "__main__":
    main()
