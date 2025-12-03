#!/usr/bin/env python3
"""Create a UTM GeoTIFF template inferred from a NISAR GCOV product.

This script builds a regular grid in the native UTM projection of a NISAR
L2 GCOV file (assuming the file is already geocoded in UTM). It is useful
when your NISAR scene covers a relatively small area and you want to avoid
reprojecting to a global EASE2 grid.

Example:
  python make_utm_template.py \
      --nisar data/raw/NISAR_L2_PR_GCOV.h5 \
      --out data/processed/UTM_template.tif \
      --resolution 1000
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS


def infer_utm_template_from_nisar(
    nisar_path: str,
    resolution: float = 1000.0,
    buffer_km: float = 10.0,
) -> tuple[CRS, Affine, int, int]:
    """Infer a UTM template grid (CRS, transform, height, width) from NISAR GCOV.

    This function:
      * Opens the NISAR L2 GCOV HDF5 file.
      * Finds a GCOV grid group (frequencyA or frequencyB).
      * Reads the x/y coordinates (assumed to be in meters in UTM).
      * Reads the projection/CRS information.
      * Computes a bounding box with an additional buffer.
      * Builds an affine transform and dimensions for a regular grid at the
        requested resolution.

    Args:
      nisar_path: Path to the NISAR L2 GCOV HDF5 file.
      resolution: Grid resolution in meters (pixel size). Defaults to 1000.
      buffer_km: Buffer around the scene in kilometers. Defaults to 10.

    Returns:
      A tuple (crs, transform, height, width) where:
        crs: Rasterio CRS object describing the projection.
        transform: Affine transform for the upper-left corner.
        height: Number of rows (pixels in y dimension).
        width: Number of columns (pixels in x dimension).

    Raises:
      ValueError: If required groups/datasets or CRS information cannot be found.
    """
    with h5py.File(nisar_path, "r") as h5_file:
        # Look for /science/LSAR/GCOV/grids/frequencyA or frequencyB.
        grid_bases = (
            "/science/LSAR/GCOV/grids/frequencyA",
            "/science/LSAR/GCOV/grids/frequencyB",
        )
        grid_group = None
        for base in grid_bases:
            if base in h5_file:
                grid_group = h5_file[base]
                break

        if grid_group is None:
            raise ValueError(
                "Could not find 'frequencyA' or 'frequencyB' under "
                "/science/LSAR/GCOV/grids in the NISAR file."
            )

        # Read x/y coordinates.
        x_dataset = grid_group.get("xCoordinates") or grid_group.get("x")
        y_dataset = grid_group.get("yCoordinates") or grid_group.get("y")
        if x_dataset is None or y_dataset is None:
            raise ValueError("Could not find 'xCoordinates'/'yCoordinates' in GCOV.")

        x_values = x_dataset[...]
        y_values = y_dataset[...]

        # Attempt to read CRS from a 'projection' dataset or from grid_mapping attrs.
        crs = None
        projection_node = grid_group.get("projection")
        if projection_node is not None:
            proj_raw = projection_node[()]
            proj_str = (
                proj_raw.decode() if isinstance(proj_raw, (bytes, bytearray)) else str(proj_raw)
            )
            try:
                crs = CRS.from_wkt(proj_str)
            except Exception:
                # Fallback: try pyproj for more flexible parsing.
                try:
                    from pyproj import CRS as PyprojCRS

                    pj_crs = PyprojCRS.from_user_input(proj_str)
                    crs = CRS.from_wkt(pj_crs.to_wkt())
                except Exception:
                    crs = None

        if crs is None:
            # Fallback: search for a dataset with a grid_mapping attribute.
            for dataset_name, dataset in grid_group.items():
                if not isinstance(dataset, h5py.Dataset):
                    continue

                grid_mapping = dataset.attrs.get("grid_mapping")
                if not grid_mapping:
                    continue

                gm_name = (
                    grid_mapping.decode()
                    if isinstance(grid_mapping, (bytes, bytearray))
                    else str(grid_mapping)
                )
                gm_node = h5_file.get(gm_name)
                if gm_node is None:
                    continue

                wkt = gm_node.attrs.get("spatial_ref")
                if not wkt:
                    continue

                wkt_str = (
                    wkt.decode() if isinstance(wkt, (bytes, bytearray)) else str(wkt)
                )
                try:
                    crs = CRS.from_wkt(wkt_str)
                    break
                except Exception:
                    crs = None

        if crs is None:
            raise ValueError(
                "Could not read CRS from NISAR GCOV metadata. "
                "Use --crs to override it explicitly."
            )

        # Bounding box in native coordinates (meters).
        xmin, xmax = float(x_values.min()), float(x_values.max())
        ymin, ymax = float(y_values.min()), float(y_values.max())

        # Expand with buffer (meters).
        buffer_m = buffer_km * 1000.0
        xmin -= buffer_m
        xmax += buffer_m
        ymin -= buffer_m
        ymax += buffer_m

        # Number of pixels (ensure they fully cover the buffered extent).
        width = int(np.ceil((xmax - xmin) / resolution))
        height = int(np.ceil((ymax - ymin) / resolution))

        # Affine transform: north-up, origin at upper-left corner.
        transform = Affine(resolution, 0.0, xmin, 0.0, -resolution, ymax)

        return crs, transform, height, width


def main() -> None:
    """Parses command-line arguments and writes the UTM template GeoTIFF."""
    parser = argparse.ArgumentParser(
        description="Create a UTM GeoTIFF template from a NISAR L2 GCOV file."
    )
    parser.add_argument(
        "--nisar",
        required=True,
        help="Path to the NISAR L2 GCOV HDF5 file.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output path for the template GeoTIFF file.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1000.0,
        help="Grid resolution in meters (default: 1000).",
    )
    parser.add_argument(
        "--buffer-km",
        type=float,
        default=10.0,
        help="Buffer around the scene in kilometers (default: 10).",
    )
    parser.add_argument(
        "--crs",
        type=str,
        default=None,
        help=(
            "Optional CRS override (e.g., 'EPSG:32611'). "
            "If provided, this CRS will be used instead of the one "
            "inferred from the NISAR file."
        ),
    )
    args = parser.parse_args()

    crs, transform, height, width = infer_utm_template_from_nisar(
        nisar_path=args.nisar,
        resolution=args.resolution,
        buffer_km=args.buffer_km,
    )

    if args.crs:
        from pyproj import CRS as PyprojCRS

        pj_crs = PyprojCRS.from_user_input(args.crs)
        crs = CRS.from_wkt(pj_crs.to_wkt())

    print("[INFO] UTM template parameters:")
    print(f"  CRS: {crs}")
    print(f"  Shape (height, width): ({height}, {width})")
    print(f"  Resolution: {args.resolution} m")
    print(f"  Transform: {transform}")

    # Write an empty single-band GeoTIFF filled with NaNs.
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "height": height,
        "width": width,
        "transform": transform,
        "crs": crs,
        "compress": "deflate",
        "predictor": 2,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "nodata": np.nan,
    }

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(np.full((height, width), np.nan, dtype=np.float32), 1)

    print(f"[OK] Template written to: {output_path}")


if __name__ == "__main__":
    main()
