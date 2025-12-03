#!/usr/bin/env python3
"""Compute WGS84 bounding box from NISAR GCOV grid coordinates.

This script reads the x/y coordinate arrays from a NISAR L2 GCOV HDF5 file,
assumes a known source CRS, and computes the geographic (WGS84) bounding box
in [lon_min, lat_min, lon_max, lat_max] form.

Example:
    python nisar_gcov_bbox.py \
        --file NISAR_L2_PR_GCOV.h5 \
        --x-path /science/LSAR/GCOV/grids/frequencyA/xCoordinates \
        --y-path /science/LSAR/GCOV/grids/frequencyA/yCoordinates \
        --epsg 32611
"""

from __future__ import annotations

import argparse
from typing import Tuple

import h5py
import numpy as np
from pyproj import CRS, Transformer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compute WGS84 bounding box from NISAR GCOV x/y coordinate arrays."
        ),
    )
    parser.add_argument(
        "--file",
        required=True,
        help="Path to the NISAR GCOV HDF5 file.",
    )
    parser.add_argument(
        "--x-path",
        default="/science/LSAR/GCOV/grids/frequencyA/xCoordinates",
        help=(
            "HDF5 path to the x-coordinate dataset "
            "(default: /science/LSAR/GCOV/grids/frequencyA/xCoordinates)."
        ),
    )
    parser.add_argument(
        "--y-path",
        default="/science/LSAR/GCOV/grids/frequencyA/yCoordinates",
        help=(
            "HDF5 path to the y-coordinate dataset "
            "(default: /science/LSAR/GCOV/grids/frequencyA/yCoordinates)."
        ),
    )
    parser.add_argument(
        "--epsg",
        type=int,
        default=32611,
        help=(
            "EPSG code of the source CRS for (x, y) coordinates "
            "(default: 32611 = WGS 84 / UTM zone 11N)."
        ),
    )
    return parser.parse_args()


def read_xy(
    filename: str,
    x_path: str,
    y_path: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Read x and y coordinate arrays from an HDF5 file.

    Args:
        filename: Path to the HDF5 file.
        x_path: HDF5 path to the x-coordinate dataset.
        y_path: HDF5 path to the y-coordinate dataset.

    Returns:
        Tuple of (x_array, y_array).

    Raises:
        KeyError: If any of the requested datasets is missing.
        OSError: If the file cannot be opened.
    """
    with h5py.File(filename, "r") as h5_file:
        if x_path not in h5_file or y_path not in h5_file:
            missing = [
                path
                for path in (x_path, y_path)
                if path not in h5_file
            ]
            raise KeyError(f"Missing datasets in file: {', '.join(missing)}")

        x = h5_file[x_path][()]  # (nx,)
        y = h5_file[y_path][()]  # (ny,)

    return x, y


def compute_bbox_wgs84(
    x: np.ndarray,
    y: np.ndarray,
    source_epsg: int,
) -> Tuple[float, float, float, float]:
    """Compute WGS84 bounding box from projected x/y coordinates.

    Args:
        x: 1D array of x coordinates in source CRS.
        y: 1D array of y coordinates in source CRS.
        source_epsg: EPSG code of the source CRS.

    Returns:
        Tuple (lon_min, lat_min, lon_max, lat_max) in WGS84.
    """
    crs_xy = CRS.from_epsg(source_epsg)
    to_wgs84 = Transformer.from_crs(
        crs_xy,
        CRS.from_epsg(4326),  # WGS84 (lat/lon)
        always_xy=True,
    )

    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))

    # Transform the two opposite corners
    lon1, lat1 = to_wgs84.transform(x_min, y_min)
    lon2, lat2 = to_wgs84.transform(x_max, y_max)

    lon_min, lon_max = sorted([lon1, lon2])
    lat_min, lat_max = sorted([lat1, lat2])

    return lon_min, lat_min, lon_max, lat_max


def main() -> None:
    """Main entry point for the NISAR GCOV bounding box utility."""
    args = parse_args()

    x, y = read_xy(args.file, args.x_path, args.y_path)
    lon_min, lat_min, lon_max, lat_max = compute_bbox_wgs84(
        x,
        y,
        args.epsg,
    )

    print(f"Source CRS: EPSG:{args.epsg}")
    print("BBox WGS84 [lon_min, lat_min, lon_max, lat_max]:")
    print(f"[{lon_min:.6f}, {lat_min:.6f}, {lon_max:.6f}, {lat_max:.6f}]")


if __name__ == "__main__":
    main()
