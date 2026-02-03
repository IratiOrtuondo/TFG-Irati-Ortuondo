#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NISAR L2 GCOV (Geocoded Covariance Matrix) Utilities
====================================================

A comprehensive command-line tool suite for working with NISAR Level-2 GCOV
HDF5 data products. This module provides three main functionalities:

Subcommands:

  1) find-backscatter
     Scans an HDF5 file and identifies candidate backscatter-like datasets
     (sigma0, gamma0, backscatter) organized by polarization (VV/HH/VH/HV).
     For each polarization, reports the primary dataset band, associated X/Y
     coordinate arrays, and projection information (CRS/EPSG code).

  2) bbox
     Computes a geographic bounding box in WGS84 latitude/longitude coordinates
     from X/Y coordinate arrays stored in a known projected CRS (EPSG code).
     Uses coordinate transformation to convert from projected to geographic space.

  3) plot
     Automatically detects Sigma0 (or gamma0 with rtcGammaToSigmaFactor) data,
     converts to decibel scale, and displays with matplotlib including coordinate
     system axes and colorbar.

Usage Examples:

  # (1) List backscatter candidates with top 10 scoring matches per polarization
  python nisar_gcov_tools.py find-backscatter NISAR_L2_PR_GCOV.h5 --only-grids --top 10

  # (2) Compute WGS84 bounding box from UTM 11N coordinates (frequencyA)
  python nisar_gcov_tools.py bbox \
      --file NISAR_L2_PR_GCOV.h5 \
      --x-path /science/LSAR/GCOV/grids/frequencyA/xCoordinates \
      --y-path /science/LSAR/GCOV/grids/frequencyA/yCoordinates \
      --epsg 32611

  # (3) Automatically find and display Sigma0 plot in decibel scale
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


# ============================================================================
# SHARED CONSTANTS AND CONFIGURATION
# ============================================================================
# This section defines regex patterns, attribute hints, and key names used
# across all subcommands to identify and locate relevant datasets in the HDF5
# file structure.

# ============================================================================
# REGEX PATTERNS FOR BACKSCATTER DATASET DISCOVERY
# ============================================================================
# List of regex patterns used to match potential backscatter-like dataset
# paths in the HDF5 hierarchy. Includes common naming conventions for:
# - Sigma0 products (linear backscatter coefficient)
# - Gamma0 products (radar brightness with incident angle correction)
# - Covariance matrix elements (VVVV, HHHH, VHVH, HVHV)
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

# ============================================================================
# ATTRIBUTE HINTS FOR BACKSCATTER IDENTIFICATION
# ============================================================================
# Keywords that appear in dataset attributes to identify backscatter-like data.
# Used to score datasets during the discovery process.
ATTR_HINTS = ["sigma", "sigma0", "backscatter", "gamma0", "nrcs", "nrsc"]

# ============================================================================
# COORDINATE AND PROJECTION DATASET IDENTIFIERS
# ============================================================================
# Common dataset names to search for when locating X/Y coordinates and
# projection/CRS information in the HDF5 hierarchy.
X_KEYS = ["xCoordinates", "xCoordinate", "x", "longitude", "lon"]
Y_KEYS = ["yCoordinates", "yCoordinate", "y", "latitude", "lat"]
PROJECTION_KEYS = ["projection", "crs"]

# ============================================================================
# POLARIZATION VARIANTS MAPPING
# ============================================================================
# Dictionary mapping polarization abbreviations to alternative naming conventions
# used in the HDF5 file. Accounts for different ways to represent the same
# polarization (e.g., "VV" vs "VVVV" for covariance matrix diagonal elements).
POL_VARIANTS: Dict[str, List[str]] = {
    "VV": ["VV", "VVVV"],
    "HH": ["HH", "HHHH"],
    "VH": ["VH", "VHVH"],
    "HV": ["HV", "HVHV"],
}

# ============================================================================
# REGEX PATTERNS FOR SIGMA0 AND GAMMA0 DETECTION (PLOT SUBCOMMAND)
# ============================================================================
# Specialized patterns used by the 'plot' subcommand to locate Sigma0
# (direct backscatter coefficient) or gamma0 (angle-normalized backscatter).
PAT_SIGMA: List[str] = [
    r"/science/LSAR/GCOV/grids/.*/Sigma0.*",
    r"/science/LSAR/GCOV/grids/.*/sigma0.*",
    r"/science/LSAR/GCOV/grids/.*/backscatter.*sigma.*",
]

PAT_GAMMA: List[str] = [
    r"/science/LSAR/GCOV/grids/.*/gamma0.*",
    r"/science/LSAR/GCOV/grids/.*/backscatter.*",
]

# ============================================================================
# FILL VALUE ATTRIBUTE NAMES
# ============================================================================
# Tuple of possible HDF5 attribute names that indicate fill/no-data values.
# These are checked to mask invalid data points when processing arrays.
_FILL_ATTR_KEYS: Tuple[str, ...] = (
    "_FillValue",
    "fillValue",
    "noDataValue",
    "NaNValue",
)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def is_2d_large(shape: Tuple[int, ...]) -> bool:
    """
    Check if a dataset shape represents a large 2D grid.
    
    This heuristic helps identify data-like arrays (images) as opposed to
    small metadata or coordinate arrays. Backscatter data is typically stored
    as 2D arrays with dimensions >= 256 pixels in each direction.
    
    Args:
        shape: Tuple of array dimensions from h5py.Dataset.shape
        
    Returns:
        True if shape is exactly 2D with both dimensions >= 256, False otherwise
    """
    return (len(shape) == 2) and (shape[0] >= 256 and shape[1] >= 256)


def detect_pol_from_name(path: str) -> Optional[str]:
    """
    Extract polarization identifier from dataset path or name.
    
    Attempts to identify the radar polarization (VV, HH, VH, or HV) from
    the dataset name using the POL_VARIANTS dictionary. First checks for
    exact matches, then falls back to regex pattern matching at end of name.
    
    Args:
        path: Full HDF5 dataset path (e.g., "/science/LSAR/GCOV/grids/.../VV")
        
    Returns:
        Polarization string ("VV", "HH", "VH", "HV") if found, None otherwise
    """
    name = path.split("/")[-1].upper()
    for pol, variants in POL_VARIANTS.items():
        for variant in variants:
            if name == variant:
                return pol
    # Fallback: ends with VV/HH/VH/HV
    match = re.search(r"(VV|HH|VH|HV)$", name)
    return match.group(1) if match else None


def score_dset(path: str, dset: h5py.Dataset) -> int:
    """
    Compute a heuristic relevance score for a dataset as a backscatter product.
    
    Scores datasets based on multiple criteria:
    - Path location (grids vs. other groups)
    - Pattern matching against PATTERN_PATHS
    - Attribute keywords (hints from metadata)
    - Array dimensions (larger 2D arrays are preferred)
    
    Args:
        path: Full HDF5 dataset path
        dset: h5py.Dataset object to evaluate
        
    Returns:
        Integer score (higher = more likely to be backscatter data)
    """
    score = 0
    path_lower = path.lower()

    # Bonus points if dataset is in the GCOV grids section
    if "/science/lsar/gcov/grids" in path_lower:
        score += 3

    # Check against known pattern paths and add score if matched
    for pattern in PATTERN_PATHS:
        if re.fullmatch(pattern.replace(".*", ".*"), path):
            score += 5

    # ========================================================================
    # SCORE BASED ON ATTRIBUTE HINTS
    # ========================================================================
    # Examine dataset attributes for keywords that indicate backscatter data
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

    # Prefer large 2D arrays (actual data grids, not metadata)
    if is_2d_large(dset.shape):
        score += 4

    return score


def find_nearby(
    h5_file: h5py.File,
    group_path: str,
    keys: List[str],
) -> Optional[str]:
    """
    Search for a dataset with specific names near a given HDF5 group path.
    
    This function helps locate coordinate or projection datasets near a
    backscatter dataset. It searches in the same group first, then in the
    parent group, using the provided list of possible dataset names.
    
    Args:
        h5_file: Opened h5py.File object
        group_path: Path to the group where the search starts
        keys: List of possible dataset names to search for
        
    Returns:
        Full path to the first found dataset, or None if not found
    """
    # 1) Search in the same group first
    if group_path in h5_file:
        for key in keys:
            candidate = f"{group_path}/{key}"
            if candidate in h5_file and isinstance(h5_file[candidate], h5py.Dataset):
                return candidate

    # 2) Search one level up (parent group)
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
    """
    Find the first large 2D dataset matching any of the provided regex patterns.
    
    Traverses the HDF5 file tree searching for datasets that match the given
    regex patterns. Returns the first dataset found that also meets size
    criteria (>1000 pixels in each dimension).
    
    Args:
        h5_file: Opened h5py.File object
        patterns: Iterable of regex patterns to match against dataset paths
        
    Returns:
        Full path to the matching dataset, or None if none found
    """
    candidates: List[Tuple[str, Tuple[int, ...]]] = []

    # Define visitor function to check each dataset in the HDF5 hierarchy
    def visit(name: str, obj: h5py.Dataset | h5py.Group) -> None:
        if not isinstance(obj, h5py.Dataset):
            return
        path = f"/{name}"
        # Check if path matches any of the provided patterns
        for pattern in patterns:
            if re.fullmatch(pattern, path):
                candidates.append((path, obj.shape))
                break

    # Traverse the entire HDF5 file
    h5_file.visititems(visit)

    # Filter for large 2D arrays suitable for plotting/display
    candidates = [
        c for c in candidates
        if len(c[1]) == 2 and min(c[1]) > 1000
    ]

    return candidates[0][0] if candidates else None


def _find_factor_dataset(h5_file: h5py.File) -> Optional[str]:
    """
    Locate the rtcGammaToSigmaFactor conversion dataset.
    
    When working with gamma0 data, this correction factor is needed to convert
    to sigma0 (backscatter coefficient). Searches preferred locations first,
    then falls back to a full file scan.
    
    Args:
        h5_file: Opened h5py.File object
        
    Returns:
        Path to rtcGammaToSigmaFactor dataset, or None if not found
    """
    # Check typical locations first for efficiency
    preferred_paths = [
        "/science/LSAR/GCOV/grids/frequencyA/rtcGammaToSigmaFactor",
        "/science/LSAR/GCOV/grids/frequencyB/rtcGammaToSigmaFactor",
    ]
    for path in preferred_paths:
        if path in h5_file:
            return path

    # Fall back to full search if not in preferred locations
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
    """
    Automatically load X/Y coordinate arrays from standard locations.
    
    Searches for X (longitude) and Y (latitude) coordinate datasets in
    common HDF5 path locations within the GCOV structure.
    
    Args:
        h5_file: Opened h5py.File object
        
    Returns:
        Tuple of (X array, Y array), with None for any not found
    """
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
    """
    Convert linear backscatter values to decibel scale.
    
    Applies the standard dB conversion formula: 10 * log10(linear_value).
    Non-positive values are masked as NaN to avoid log of zero/negative.
    
    Args:
        arr: NumPy array of linear backscatter values
        
    Returns:
        Array of values in decibel scale, with NaN for non-positive inputs
    """
    out = np.full(arr.shape, np.nan, dtype=float)
    mask = arr > 0
    out[mask] = 10.0 * np.log10(arr[mask])
    return out


def _clean_fill_values(
    h5_file: h5py.File,
    dataset_path: str,
    data: np.ndarray,
) -> np.ndarray:
    """
    Replace fill/no-data values with NaN based on HDF5 metadata.
    
    Checks for HDF5 fill-value attributes and replaces matching values
    in the data array with NaN to ensure proper analysis and visualization.
    
    Args:
        h5_file: Opened h5py.File object
        dataset_path: Path to the dataset being processed
        data: NumPy array to clean
        
    Returns:
        Data array with fill values replaced by NaN
    """
    if dataset_path not in h5_file:
        return data

    dset = h5_file[dataset_path]
    fill_value = None
    
    # Search for fill value in standard HDF5 attribute names
    for key in _FILL_ATTR_KEYS:
        if key in dset.attrs:
            fill_value = dset.attrs[key]
            break

    if fill_value is None:
        return data

    # Replace fill values with NaN
    cleaned = np.where(data == fill_value, np.nan, data)
    return cleaned


# ============================================================================
# SUBCOMMAND 1: find-backscatter
# ============================================================================
# This subcommand scans the HDF5 file and identifies backscatter-like datasets,
# organizing results by polarization and providing metadata paths.

def cmd_find_backscatter(args: argparse.Namespace) -> None:
    """
    Implementation of the find-backscatter subcommand.
    
    Scans the HDF5 file for backscatter/sigma0/gamma0 datasets and scores them
    based on path patterns, attributes, and array size. Results are grouped by
    polarization and the best candidate per polarization is reported with its
    coordinate and projection metadata.
    
    Args:
        args: Argument namespace from argparse with 'file', 'only_grids', 'top'
    """
    try:
        h5_file = h5py.File(args.file, "r")
    except Exception as exc:
        sys.exit(f"Could not open {args.file}: {exc}")

    candidates: List[Tuple[int, str, Tuple[int, ...]]] = []
    all_dsets: Dict[str, h5py.Dataset] = {}

    # ========================================================================
    # SCAN HDF5 FILE FOR BACKSCATTER CANDIDATES
    # ========================================================================
    def visit(name: str, obj: h5py.Dataset | h5py.Group) -> None:
        if isinstance(obj, h5py.Dataset):
            path = "/" + name
            # Filter by grids if requested
            if args.only_grids and not path.lower().startswith("/science/lsar/gcov/grids"):
                return
            all_dsets[path] = obj
            # Score dataset for backscatter-likeness
            dataset_score = score_dset(path, obj)
            if dataset_score > 0:
                candidates.append((dataset_score, path, obj.shape))

    h5_file.visititems(visit)

    # ========================================================================
    # IDENTIFY GAMMA-TO-SIGMA CONVERSION FACTORS
    # ========================================================================
    # Look for rtcGammaToSigmaFactor-like datasets (for gamma0 -> sigma0 conversion)
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

    # ========================================================================
    # HANDLE CASE WHERE NO CANDIDATES ARE FOUND
    # ========================================================================
    if not candidates:
        print("No sigma0/gamma0/backscatter-like dataset found.")
        if gamma_to_sigma_paths:
            print(
                "Note: rtcGammaToSigmaFactor exists, "
                "but no gamma0/sigma0 bands were found."
            )
        h5_file.close()
        return

    # ========================================================================
    # SORT CANDIDATES BY SCORE AND AREA
    # ========================================================================
    # Higher scores and larger areas are ranked first
    candidates.sort(
        key=lambda item: (
            item[0],
            (item[2][0] * item[2][1] if len(item[2]) == 2 else 0),
        ),
        reverse=True,
    )

    # Print top N candidates
    print("Top candidates:")
    for score, path, shape in candidates[: args.top]:
        print(f"  score={score:2d}  path={path}  shape={shape}")

    # ========================================================================
    # ORGANIZE BY POLARIZATION AND FIND METADATA
    # ========================================================================
    # Group candidates by their detected polarization
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

        # Custom sort for polarization-specific ranking
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

        # Find associated X, Y, and projection datasets
        x_path = find_nearby(h5_file, group, X_KEYS)
        y_path = find_nearby(h5_file, group, Y_KEYS)

        proj_path: Optional[str] = None
        # Check for projection in same group
        for key in PROJECTION_KEYS:
            candidate = f"{group}/{key}"
            if candidate in h5_file:
                proj_path = candidate
                break

        # If not found, search up the hierarchy
        if proj_path is None:
            parts = [p for p in group.split("/") if p]
            for idx in range(len(parts), 0, -1):
                candidate = "/" + "/".join(parts[:idx]) + "/projection"
                if candidate in h5_file:
                    proj_path = candidate
                    break

        # Print results for this polarization
        print(f"  {pol}:")
        print(f"    band: {best_path}  shape={best_shape}")
        print(f"    x:    {x_path or '(not found)'}")
        print(f"    y:    {y_path or '(not found)'}")
        print(f"    proj: {proj_path or '(not found)'}")

    # Print gamma-to-sigma factors if found
    if gamma_to_sigma_paths:
        print("\nFound rtcGammaToSigmaFactor (for gamma0→sigma0 conversion):")
        for path in gamma_to_sigma_paths:
            shape = all_dsets[path].shape
            print(f"  {path}  shape={shape}")

    h5_file.close()


# ============================================================================
# SUBCOMMAND 2: bbox
# ============================================================================
# This subcommand computes geographic bounding boxes by transforming projected
# coordinates to WGS84 latitude/longitude.

def read_xy(
    filename: str,
    x_path: str,
    y_path: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read X and Y coordinate arrays from an HDF5 file.
    
    Args:
        filename: Path to the HDF5 file
        x_path: HDF5 path to X-coordinate dataset
        y_path: HDF5 path to Y-coordinate dataset
        
    Returns:
        Tuple of (X array, Y array)
        
    Raises:
        KeyError: If datasets are not found in the file
    """
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
    """
    Compute a WGS84 bounding box from projected X/Y coordinates.
    
    Transforms the extent of the input arrays from a projected coordinate
    system (specified by EPSG code) to WGS84 (EPSG:4326) geographic coordinates.
    
    Args:
        x: Array of X-coordinates (easting) in the source CRS
        y: Array of Y-coordinates (northing) in the source CRS
        source_epsg: EPSG code of the source coordinate reference system
        
    Returns:
        Tuple of (lon_min, lat_min, lon_max, lat_max) in WGS84
    """
    # Initialize coordinate reference systems
    crs_xy = CRS.from_epsg(source_epsg)
    to_wgs84 = Transformer.from_crs(
        crs_xy,
        CRS.from_epsg(4326),
        always_xy=True,
    )

    # Find extent of the input coordinates
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))

    # Transform all four corners to catch any distortions
    lon1, lat1 = to_wgs84.transform(x_min, y_min)
    lon2, lat2 = to_wgs84.transform(x_max, y_max)

    # Ensure bounds are in correct order (min < max)
    lon_min, lon_max = sorted([lon1, lon2])
    lat_min, lat_max = sorted([lat1, lat2])

    return lon_min, lat_min, lon_max, lat_max


def cmd_bbox(args: argparse.Namespace) -> None:
    """
    Implementation of the bbox subcommand.
    
    Reads X/Y coordinates from an HDF5 file, transforms them from a projected
    CRS to WGS84, and prints the geographic bounding box.
    
    Args:
        args: Argument namespace with 'file', 'x_path', 'y_path', 'epsg'
    """
    x, y = read_xy(args.file, args.x_path, args.y_path)
    lon_min, lat_min, lon_max, lat_max = compute_bbox_wgs84(
        x,
        y,
        args.epsg,
    )
    print(f"Source CRS: EPSG:{args.epsg}")
    print("BBox WGS84 [lon_min, lat_min, lon_max, lat_max]:")
    print(f"[{lon_min:.6f}, {lat_min:.6f}, {lon_max:.6f}, {lat_max:.6f}]")


# ============================================================================
# SUBCOMMAND 3: plot
# ============================================================================
# This subcommand finds and visualizes Sigma0 or gamma0 data in decibel scale.

def cmd_plot(args: argparse.Namespace) -> None:
    """
    Implementation of the plot subcommand.
    
    Automatically finds Sigma0 (or gamma0 with conversion factor) data,
    converts to dB, loads coordinates if available, and displays with matplotlib.
    
    Args:
        args: Argument namespace with 'file' (HDF5 file path)
    """
    path = args.file
    with h5py.File(path, "r") as h5_file:
        # Try to find direct Sigma0 data first
        sigma_path = _find_first_dataset(h5_file, PAT_SIGMA)
        mode: str

        if sigma_path is not None:
            # Sigma0 found directly
            data = h5_file[sigma_path][()]
            data = _clean_fill_values(h5_file, sigma_path, data)
            mode = "direct Sigma0"
        else:
            # Try gamma0 + conversion factor approach
            gamma_path = _find_first_dataset(h5_file, PAT_GAMMA)
            if gamma_path is None:
                sys.exit(
                    "Could not find Sigma0 or gamma0/backscatter grids. "
                    "Try 'find-backscatter' to inspect candidates."
                )

            # Get the conversion factor
            factor_path = _find_factor_dataset(h5_file)
            if factor_path is None:
                sys.exit("Found gamma0 but not rtcGammaToSigmaFactor.")

            # Load and multiply gamma0 by the conversion factor
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

        # Load coordinate arrays for axis labels
        x, y = _load_coordinates_auto(h5_file)
        
        # Convert to decibel scale
        img_db = _to_db(data)

        # Set up extent for proper geographic axis display
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

        # ====================================================================
        # CREATE AND DISPLAY PLOT
        # ====================================================================
        plt.figure(figsize=(8, 6))
        im = plt.imshow(img_db, extent=extent, origin=origin)
        plt.title(f"Sigma0 dB ({mode})\n{sigma_path}")
        plt.xlabel("Longitude / X" if extent is not None else "Sample")
        plt.ylabel("Latitude / Y" if extent is not None else "Line")
        cb = plt.colorbar(im)
        cb.set_label("Sigma0 [dB]")
        plt.tight_layout()
        plt.show()


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================
# Build the argument parser with subcommands for each tool.

def build_parser() -> argparse.ArgumentParser:
    """
    Construct the argument parser with all subcommands.
    
    Returns:
        Configured ArgumentParser object
    """
    parser = argparse.ArgumentParser(
        description="Combined tools for NISAR L2 GCOV (find, bbox, plot)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ========================================================================
    # SUBCOMMAND: find-backscatter
    # ========================================================================
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

    # ========================================================================
    # SUBCOMMAND: bbox
    # ========================================================================
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

    # ========================================================================
    # SUBCOMMAND: plot
    # ========================================================================
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
    """
    Main entry point for the command-line interface.
    
    Parses arguments, validates subcommand, and executes the corresponding
    function.
    """
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
