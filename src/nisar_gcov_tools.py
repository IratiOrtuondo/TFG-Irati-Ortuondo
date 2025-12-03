#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NISAR L2 GCOV utilities in a single script.

Subcommands:

  1) find-backscatter
     Scan an HDF5 and propose candidate backscatter-like datasets
     (sigma0/gamma0/backscatter) per polarization, including band, X/Y coords
     and projection paths.

  2) bbox
     Compute a geographic bounding box in WGS84
     (lon_min, lat_min, lon_max, lat_max) from X/Y coordinate arrays in a
     known projected CRS (EPSG).

  3) plot
     Automatically find a Sigma0 (or gamma0 + rtcGammaToSigmaFactor) grid,
     convert to dB, and display it with matplotlib.

Examples:

  # (1) List backscatter candidates per polarization
  python nisar_gcov_tools.py find-backscatter NISAR_L2_PR_GCOV.h5 --only-grids --top 10

  # (2) Bounding box using frequencyA x/y in UTM 11N
  python nisar_gcov_tools.py bbox \
      --file NISAR_L2_PR_GCOV.h5 \
      --x-path /science/LSAR/GCOV/grids/frequencyA/xCoordinates \
      --y-path /science/LSAR/GCOV/grids/frequencyA/yCoordinates \
      --epsg 32611

  # (3) Automatic Sigma0 plot in dB
  python nisar_gcov_tools.py plot NISAR_L2_PR_GCOV.h5
"""

from __future__ import annotations

import argparse
import re
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from pyproj import CRS, Transformer


# ----------------------------------------------------------------------
# Shared constants
# ----------------------------------------------------------------------

# Generic patterns for backscatter/sigma0/gamma0
PATTERN_PATHS = [
    r"/science/LSAR/GCOV/.*/Sigma0.*",
    r"/science/LSAR/GCOV/.*/sigma0.*",
    r"/science/LSAR/GCOV/.*/Backscatter.*",
    r"/science/LSAR/GCOV/.*/backscatter.*",
    r"/science/LSAR/GCOV/.*/gamma0.*",
    r"/science/LSAR/GCOV/.*/VV$",
    r"/science/LSAR/GCOV/.*/HH$",
    r"/science/LSAR/GCOV/.*/VH$",
    r"/science/LSAR/GCOV/.*/HV$",
    # Common covariance/power variants
    r"/science/LSAR/GCOV/.*/VVVV$",
    r"/science/LSAR/GCOV/.*/HHHH$",
    r"/science/LSAR/GCOV/.*/HVHV$",
    r"/science/LSAR/GCOV/.*/VHVH$",
]

ATTR_HINTS = ["sigma", "sigma0", "backscatter", "gamma0", "nrcs", "nrsc"]

# Typical keys for coordinates and projection
X_KEYS = ["xCoordinates", "xCoordinate", "x", "longitude", "lon"]
Y_KEYS = ["yCoordinates", "yCoordinate", "y", "latitude", "lat"]
PROJECTION_KEYS = ["projection", "crs"]

POL_VARIANTS: Dict[str, List[str]] = {
    "VV": ["VV", "VVVV"],
    "HH": ["HH", "HHHH"],
    "VH": ["VH", "VHVH"],
    "HV": ["HV", "HVHV"],
}

# Patterns used by the "plot" subcommand
PAT_SIGMA: List[str] = [
    r"/science/LSAR/GCOV/grids/.*/Sigma0.*",
    r"/science/LSAR/GCOV/grids/.*/sigma0.*",
    r"/science/LSAR/GCOV/grids/.*/backscatter.*sigma.*",
]

PAT_GAMMA: List[str] = [
    r"/science/LSAR/GCOV/grids/.*/gamma0.*",
    r"/science/LSAR/GCOV/grids/.*/backscatter.*",
]

_FILL_ATTR_KEYS: Tuple[str, ...] = (
    "_FillValue",
    "fillValue",
    "noDataValue",
    "NaNValue",
)


# ----------------------------------------------------------------------
# Generic helpers
# ----------------------------------------------------------------------

def is_2d_large(shape: Tuple[int, ...]) -> bool:
    """Return True if the dataset looks like a large 2D grid."""
    return (len(shape) == 2) and (shape[0] >= 256 and shape[1] >= 256)


def detect_pol_from_name(path: str) -> Optional[str]:
    """Infer polarization (VV/HH/VH/HV) from the dataset name."""
    name = path.split("/")[-1].upper()
    for pol, variants in POL_VARIANTS.items():
        for variant in variants:
            if name == variant:
                return pol
    # Fallback: ends with VV/HH/VH/HV
    match = re.search(r"(VV|HH|VH|HV)$", name)
    return match.group(1) if match else None


def score_dset(path: str, dset: h5py.Dataset) -> int:
    """Compute a heuristic score for how 'backscatter-like' a dataset is."""
    score = 0
    path_lower = path.lower()

    if "/science/lsar/gcov/grids" in path_lower:
        score += 3

    for pattern in PATTERN_PATHS:
        if re.fullmatch(pattern.replace(".*", ".*"), path):
            score += 5

    # Attribute hints
    for _, value in dset.attrs.items():
        try:
            str_value = (
                value.decode()
                if isinstance(value, (bytes, bytearray))
                else str(value)
            )
        except Exception:
            str_value = str(value)
        str_value = str_value.lower()
        if any(hint in str_value for hint in ATTR_HINTS):
            score += 3

    # Prefer large 2D arrays
    if is_2d_large(dset.shape):
        score += 4

    return score


def find_nearby(
    h5_file: h5py.File,
    group_path: str,
    keys: List[str],
) -> Optional[str]:
    """Search for coordinate/projection datasets near a given group."""
    # 1) Same group
    if group_path in h5_file:
        for key in keys:
            candidate = f"{group_path}/{key}"
            if candidate in h5_file and isinstance(h5_file[candidate], h5py.Dataset):
                return candidate

    # 2) One level up
    parts = [p for p in group_path.split("/") if p]
    if parts:
        parent = "/" + "/".join(parts[:-1]) if len(parts) > 1 else "/"
        if parent in h5_file:
            for key in keys:
                candidate = f"{parent}/{key}"
                if candidate in h5_file and isinstance(h5_file[candidate], h5py.Dataset):
                    return candidate

    return None


def _find_first_dataset(
    h5_file: h5py.File,
    patterns: Iterable[str],
) -> Optional[str]:
    """Find the first large 2D dataset matching any of the regex patterns."""
    candidates: List[Tuple[str, Tuple[int, ...]]] = []

    def visit(name: str, obj: h5py.Dataset | h5py.Group) -> None:
        if not isinstance(obj, h5py.Dataset):
            return
        path = f"/{name}"
        for pattern in patterns:
            if re.fullmatch(pattern, path):
                candidates.append((path, obj.shape))
                break

    h5_file.visititems(visit)

    # Prefer large 2D grids (for plotting)
    candidates = [
        c for c in candidates
        if len(c[1]) == 2 and min(c[1]) > 1000
    ]

    return candidates[0][0] if candidates else None


def _find_factor_dataset(h5_file: h5py.File) -> Optional[str]:
    """Find rtcGammaToSigmaFactor dataset in typical locations or via a generic search."""
    preferred_paths = [
        "/science/LSAR/GCOV/grids/frequencyA/rtcGammaToSigmaFactor",
        "/science/LSAR/GCOV/grids/frequencyB/rtcGammaToSigmaFactor",
    ]
    for path in preferred_paths:
        if path in h5_file:
            return path

    found: Optional[str] = None

    def visit(name: str, obj: h5py.Dataset | h5py.Group) -> None:
        nonlocal found
        if found is not None:
            return
        if isinstance(obj, h5py.Dataset) and name.endswith("rtcGammaToSigmaFactor"):
            found = f"/{name}"

    h5_file.visititems(visit)
    return found


def _load_coordinates_auto(h5_file: h5py.File) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Try to load X/Y coordinate arrays from typical locations."""
    def find_coord(candidates: List[str]) -> Optional[np.ndarray]:
        for dataset_name in candidates:
            paths = [
                f"/science/LSAR/GCOV/grids/frequencyA/{dataset_name}",
                f"/science/LSAR/GCOV/grids/{dataset_name}",
                f"/science/LSAR/GCOV/{dataset_name}",
            ]
            for path in paths:
                if path in h5_file:
                    return h5_file[path][()]
        return None

    x_array = find_coord(["xCoordinates", "longitude"])
    y_array = find_coord(["yCoordinates", "latitude"])
    return x_array, y_array


def _to_db(arr: np.ndarray) -> np.ndarray:
    """Convert linear values to dB, masking non-positive values."""
    out = np.full(arr.shape, np.nan, dtype=float)
    mask = arr > 0
    out[mask] = 10.0 * np.log10(arr[mask])
    return out


def _clean_fill_values(
    h5_file: h5py.File,
    dataset_path: str,
    data: np.ndarray,
) -> np.ndarray:
    """Replace fill-value-like entries with NaN when metadata is available."""
    if dataset_path not in h5_file:
        return data

    dset = h5_file[dataset_path]
    fill_value = None
    for key in _FILL_ATTR_KEYS:
        if key in dset.attrs:
            fill_value = dset.attrs[key]
            break

    if fill_value is None:
        return data

    cleaned = np.where(data == fill_value, np.nan, data)
    return cleaned


# ----------------------------------------------------------------------
# Subcommand 1: find-backscatter
# ----------------------------------------------------------------------

def cmd_find_backscatter(args: argparse.Namespace) -> None:
    """Implementation of the find-backscatter subcommand."""
    try:
        h5_file = h5py.File(args.file, "r")
    except Exception as exc:
        sys.exit(f"Could not open {args.file}: {exc}")

    candidates: List[Tuple[int, str, Tuple[int, ...]]] = []
    all_dsets: Dict[str, h5py.Dataset] = {}

    def visit(name: str, obj: h5py.Dataset | h5py.Group) -> None:
        if isinstance(obj, h5py.Dataset):
            path = "/" + name
            if args.only_grids and not path.lower().startswith("/science/lsar/gcov/grids"):
                return
            all_dsets[path] = obj
            dataset_score = score_dset(path, obj)
            if dataset_score > 0:
                candidates.append((dataset_score, path, obj.shape))

    h5_file.visititems(visit)

    # Look for rtcGammaToSigmaFactor-like datasets
    gamma_to_sigma_paths: List[str] = [
        path
        for path in all_dsets.keys()
        if path.lower().endswith("/rtcgammasigmatofactor")
        or path.lower().endswith("/rtcgtosigmafactor")
        or "rtcgammatosigmafactor" in path.lower()
    ]

    for path in all_dsets.keys():
        if path.lower().endswith("/rtcgammtosigmafactor") or (
            "rtcgammatosigmafactor" in path.lower()
        ):
            if path not in gamma_to_sigma_paths:
                gamma_to_sigma_paths.append(path)

    if not candidates:
        print("No sigma0/gamma0/backscatter-like dataset found.")
        if gamma_to_sigma_paths:
            print(
                "Note: rtcGammaToSigmaFactor exists, "
                "but no gamma0/sigma0 bands were found."
            )
        h5_file.close()
        return

    # Sort by score and area
    candidates.sort(
        key=lambda item: (
            item[0],
            (item[2][0] * item[2][1] if len(item[2]) == 2 else 0),
        ),
        reverse=True,
    )

    print("Top candidates:")
    for score, path, shape in candidates[: args.top]:
        print(f"  score={score:2d}  path={path}  shape={shape}")

    # Per-polarization summary
    per_pol: Dict[str, List[Tuple[int, str, Tuple[int, ...]]]] = {
        "VV": [],
        "HH": [],
        "VH": [],
        "HV": [],
    }
    for score, path, shape in candidates:
        pol = detect_pol_from_name(path)
        if pol in per_pol:
            per_pol[pol].append((score, path, shape))

    print("\nPer-polarization summary (best candidate first):")
    for pol in ["VV", "HH", "VH", "HV"]:
        pol_list = per_pol[pol]
        if not pol_list:
            print(f"  {pol}: (not found)")
            continue

        def pol_sort_key(item: Tuple[int, str, Tuple[int, ...]]) -> Tuple[int, int]:
            score, path, shape = item
            path_lower = path.lower()
            bonus = 0
            if "/science/lsar/gcov/grids" in path_lower:
                bonus += 3
            if (
                "backscatter" in path_lower
                or "gamma0" in path_lower
                or "sigma0" in path_lower
            ):
                bonus += 2
            area = (shape[0] * shape[1]) if len(shape) == 2 else 0
            return score + bonus, area

        sorted_pol_list = sorted(pol_list, key=pol_sort_key, reverse=True)

        best_score, best_path, best_shape = sorted_pol_list[0]
        group = "/".join(best_path.split("/")[:-1]) or "/"

        x_path = find_nearby(h5_file, group, X_KEYS)
        y_path = find_nearby(h5_file, group, Y_KEYS)

        proj_path: Optional[str] = None
        # Projection in the same group or ascending
        for key in PROJECTION_KEYS:
            candidate = f"{group}/{key}"
            if candidate in h5_file:
                proj_path = candidate
                break

        if proj_path is None:
            parts = [p for p in group.split("/") if p]
            for idx in range(len(parts), 0, -1):
                candidate = "/" + "/".join(parts[:idx]) + "/projection"
                if candidate in h5_file:
                    proj_path = candidate
                    break

        print(f"  {pol}:")
        print(f"    band: {best_path}  shape={best_shape}")
        print(f"    x:    {x_path or '(not found)'}")
        print(f"    y:    {y_path or '(not found)'}")
        print(f"    proj: {proj_path or '(not found)'}")

    if gamma_to_sigma_paths:
        print("\nFound rtcGammaToSigmaFactor (for gamma0→sigma0):")
        for path in gamma_to_sigma_paths:
            shape = all_dsets[path].shape
            print(f"  {path}  shape={shape}")

    h5_file.close()


# ----------------------------------------------------------------------
# Subcommand 2: bbox
# ----------------------------------------------------------------------

def read_xy(
    filename: str,
    x_path: str,
    y_path: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Read X and Y coordinate arrays from an HDF5 file."""
    with h5py.File(filename, "r") as h5_file:
        if x_path not in h5_file or y_path not in h5_file:
            missing = [p for p in (x_path, y_path) if p not in h5_file]
            raise KeyError(f"Missing datasets in file: {', '.join(missing)}")

        x = h5_file[x_path][()]
        y = h5_file[y_path][()]

    return x, y


def compute_bbox_wgs84(
    x: np.ndarray,
    y: np.ndarray,
    source_epsg: int,
) -> Tuple[float, float, float, float]:
    """Compute a WGS84 bounding box from projected X/Y coordinates."""
    crs_xy = CRS.from_epsg(source_epsg)
    to_wgs84 = Transformer.from_crs(
        crs_xy,
        CRS.from_epsg(4326),
        always_xy=True,
    )

    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))

    lon1, lat1 = to_wgs84.transform(x_min, y_min)
    lon2, lat2 = to_wgs84.transform(x_max, y_max)

    lon_min, lon_max = sorted([lon1, lon2])
    lat_min, lat_max = sorted([lat1, lat2])

    return lon_min, lat_min, lon_max, lat_max


def cmd_bbox(args: argparse.Namespace) -> None:
    """Implementation of the bbox subcommand."""
    x, y = read_xy(args.file, args.x_path, args.y_path)
    lon_min, lat_min, lon_max, lat_max = compute_bbox_wgs84(
        x,
        y,
        args.epsg,
    )
    print(f"Source CRS: EPSG:{args.epsg}")
    print("BBox WGS84 [lon_min, lat_min, lon_max, lat_max]:")
    print(f"[{lon_min:.6f}, {lat_min:.6f}, {lon_max:.6f}, {lat_max:.6f}]")


# ----------------------------------------------------------------------
# Subcommand 3: plot
# ----------------------------------------------------------------------

def cmd_plot(args: argparse.Namespace) -> None:
    """Implementation of the plot subcommand (Sigma0/gamma0 in dB)."""
    path = args.file
    with h5py.File(path, "r") as h5_file:
        sigma_path = _find_first_dataset(h5_file, PAT_SIGMA)
        mode: str

        if sigma_path is not None:
            data = h5_file[sigma_path][()]
            data = _clean_fill_values(h5_file, sigma_path, data)
            mode = "direct Sigma0"
        else:
            gamma_path = _find_first_dataset(h5_file, PAT_GAMMA)
            if gamma_path is None:
                sys.exit(
                    "Could not find Sigma0 or gamma0/backscatter grids. "
                    "Try 'find-backscatter' to inspect candidates."
                )

            factor_path = _find_factor_dataset(h5_file)
            if factor_path is None:
                sys.exit("Found gamma0 but not rtcGammaToSigmaFactor.")

            gamma = h5_file[gamma_path][()].astype("float64")
            factor = h5_file[factor_path][()].astype("float64")
            if gamma.shape != factor.shape:
                sys.exit(
                    f"Shape mismatch: gamma0 {gamma.shape} vs factor {factor.shape}"
                )

            gamma = _clean_fill_values(h5_file, gamma_path, gamma)
            data = gamma * factor
            sigma_path = "(computed) gamma0 * rtcGammaToSigmaFactor"
            mode = "computed from gamma0"

        x, y = _load_coordinates_auto(h5_file)
        img_db = _to_db(data)

        extent = None
        origin = "upper"
        if (
            x is not None
            and y is not None
            and x.ndim == 1
            and y.ndim == 1
        ):
            extent = [float(x.min()), float(x.max()),
                      float(y.min()), float(y.max())]

        plt.figure(figsize=(8, 6))
        im = plt.imshow(img_db, extent=extent, origin=origin)
        plt.title(f"Sigma0 dB ({mode})\n{sigma_path}")
        plt.xlabel("Longitude / X" if extent is not None else "Sample")
        plt.ylabel("Latitude / Y" if extent is not None else "Line")
        cb = plt.colorbar(im)
        cb.set_label("Sigma0 [dB]")
        plt.tight_layout()
        plt.show()


# ----------------------------------------------------------------------
# Main CLI with subparsers
# ----------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Combined tools for NISAR L2 GCOV (find, bbox, plot)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # find-backscatter
    p_find = subparsers.add_parser(
        "find-backscatter",
        help="Search for backscatter/sigma0/gamma0-like datasets and summarize by polarization.",
    )
    p_find.add_argument("file", help="Path to the GCOV HDF5 file.")
    p_find.add_argument(
        "--only-grids",
        action="store_true",
        help="Restrict search to /science/LSAR/GCOV/grids.",
    )
    p_find.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top candidates to display.",
    )
    p_find.set_defaults(func=cmd_find_backscatter)

    # bbox
    p_bbox = subparsers.add_parser(
        "bbox",
        help="Compute WGS84 bounding box from X/Y coordinate arrays.",
    )
    p_bbox.add_argument(
        "--file",
        required=True,
        help="Path to the GCOV HDF5 file.",
    )
    p_bbox.add_argument(
        "--x-path",
        default="/science/LSAR/GCOV/grids/frequencyA/xCoordinates",
        help="HDF5 path to the X-coordinate dataset.",
    )
    p_bbox.add_argument(
        "--y-path",
        default="/science/LSAR/GCOV/grids/frequencyA/yCoordinates",
        help="HDF5 path to the Y-coordinate dataset.",
    )
    p_bbox.add_argument(
        "--epsg",
        type=int,
        default=32611,
        help="EPSG code of the source CRS (X,Y).",
    )
    p_bbox.set_defaults(func=cmd_bbox)

    # plot
    p_plot = subparsers.add_parser(
        "plot",
        help="Find and plot Sigma0 (or gamma0+factor) in dB.",
    )
    p_plot.add_argument(
        "file",
        help="Path to the GCOV HDF5 file.",
    )
    p_plot.set_defaults(func=cmd_plot)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
