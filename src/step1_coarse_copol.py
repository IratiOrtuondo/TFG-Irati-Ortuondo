#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step1_coarse_copol.py — Prepare SMAP radiometer TB and SMAP radar backscatter
aligned on a common grid (coarse 36 km default, or bbox-defined template).

This commented version contains a clear module docstring and function-level
docstrings plus inline explanations so that another developer can quickly
understand the assumptions and fallbacks used when reading SMAP L3 and
radar products.

Key behaviors:
- Create a GeoTIFF template either from a SMAP L3 product (global EASE2 36 km)
  or from an explicit lon/lat bounding box + pixel size.
- Load TB from SMAP L3 (several name heuristics and optional AM/PM groups).
- Load gridded radar σ0 (SPL3SMA) and fall back to conservative EASE2 3 km
  transform when geolocation is not available.
- Optionally bin XPOL swath (L1C) to the template using simple averaging.
- Save an NPZ with aligned arrays and metadata for downstream processing.

Outputs:
  data/interim/aligned-smap-YYYYMMDD.npz containing:
    TBc_2d, S_pp_dB, (optional) S_xpol_dB, crs_wkt, transform, height, width, meta

This file is intended for use from the command-line but is also suitable for
importing into unit tests or other pipelines.
"""

from __future__ import annotations

import argparse
import math
import re
import warnings
from pathlib import Path

import h5py
import numpy as np
import rasterio
import xarray as xr
from affine import Affine
from rasterio.crs import CRS
from rasterio.enums import Resampling as ResampEnum
from rasterio.transform import from_origin
from rasterio.warp import reproject
from rasterio.warp import transform as rio_transform
from rasterio.warp import transform_bounds


# -------------------------------------------------------------------------
# Project paths
# -------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
INTERIM.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------------
# Basic utilities
# -------------------------------------------------------------------------


def to_db(arr: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Convert a power-like array to decibels (dB).

    Clips at `eps` to avoid log(0), and returns float32 for storage.
    Use this for radar linear -> dB conversion if the magnitude suggests
    the values are not already in dB (large positive values).
    """
    return (10.0 * np.log10(np.clip(arr, eps, None))).astype(np.float32)


def is_mono_eq(v: np.ndarray, rtol: float = 1e-6, atol: float = 1e-6) -> bool:
    """Return True if 1D coordinate `v` is monotonic and equally spaced.

    This helper is used to detect regular grid coordinates in xarray/NetCDF
    datasets and to build Affine transforms when spacing is uniform.
    """
    if v.ndim != 1 or v.size < 2:
        return False
    d = np.diff(v.astype(np.float64))
    return np.all(d > -atol) and np.allclose(d, d[0], rtol=rtol, atol=atol)


def open_h5(path: str) -> h5py.File:
    """Open an HDF5 file or raise a helpful error if it is missing."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File does not exist: {p}")
    return h5py.File(p, "r")


def _write_debug_tif(path: Path, arr: np.ndarray, transform: Affine, crs: CRS) -> None:
    """Write a single-band float32 GeoTIFF for QA and debugging.

    Produces a compressed tiled GeoTIFF (deflate) suitable for quick visual
    inspection in QGIS or similar tools.
    """
    prof = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "height": int(arr.shape[0]),
        "width": int(arr.shape[1]),
        "transform": transform,
        "crs": crs,
        "compress": "deflate",
        "predictor": 2,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "nodata": np.nan,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype(np.float32), 1)


# -------------------------------------------------------------------------
# Template handling
# -------------------------------------------------------------------------


def load_template(template_path: str) -> tuple[CRS, Affine, int, int, dict]:
    """Load a GeoTIFF template and return its CRS, transform, shape and profile."""
    with rasterio.open(template_path) as ds:
        crs = ds.crs
        transform = ds.transform
        height, width = ds.height, ds.width
        profile = ds.profile
    return crs, transform, height, width, profile


def make_template_from_smap_l3(smap_l3_path: str, out_template: str) -> str:
    """Create a global EASE2-like GeoTIFF template from a SMAP L3 file.

    This function attempts to inspect the SMAP L3 file to determine the
    2D array shape automatically; if that fails it falls back to the
    canonical global EASE2 36 km shape (ny=406, nx=964). The output CRS
    defaults to EPSG:6933 (EASE2 projection) and pixel size to ~36 km.
    """
    ny = nx = None

    try:
        with xr.open_dataset(smap_l3_path, engine="h5netcdf", phony_dims="sort") as ds_root:
            cand_vars = [v for v in ds_root.data_vars if ds_root[v].ndim == 2]
            if cand_vars:
                v0 = ds_root[cand_vars[0]]
                ny, nx = v0.shape
                print(f"[INFO] Template shape from root group: ny={ny}, nx={nx}")
    except Exception as e:
        print(f"[WARN] Could not read root group of SMAP L3: {e}")

    if ny is None or nx is None:
        groups_to_try = [
            "Soil_Moisture_Retrieval_Data_AM",
            "Soil_Moisture_Retrieval_Data_PM",
            "Radiometer_Retrieval_Data_AM",
            "Radiometer_Retrieval_Data_PM",
            "Radiometer_Retrieval_Data",
            "Soil_Moisture_Retrieval_Data",
        ]
        for grp in groups_to_try:
            try:
                with xr.open_dataset(smap_l3_path, engine="h5netcdf", phony_dims="sort", group=grp) as ds:
                    cand_vars = [v for v in ds.data_vars if ds[v].ndim == 2]
                    if cand_vars:
                        v0 = ds[cand_vars[0]]
                        ny, nx = v0.shape
                        print(f"[INFO] Template shape from group '{grp}': ny={ny}, nx={nx}")
                        break
            except Exception:
                continue

    if ny is None or nx is None:
        # Fallback to canonical EASE2 36 km global size if automatic detection fails
        ny, nx = 406, 964
        warnings.warn(
            "No 2D variables found in SMAP L3 to derive template shape. "
            "Using fallback EASE2 36 km global size ny=406, nx=964."
        )

    crs = CRS.from_epsg(6933)
    dx = dy = 36000.0
    transform = from_origin(-17367530.45, 7314540.83, dx, dy)

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "height": int(ny),
        "width": int(nx),
        "transform": transform,
        "crs": crs,
        "compress": "deflate",
        "predictor": 2,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "nodata": np.nan,
    }

    out_template = Path(out_template)
    out_template.parent.mkdir(parents=True, exist_ok=True)
    data = np.full((ny, nx), np.nan, dtype=np.float32)
    with rasterio.open(out_template, "w", **profile) as dst:
        dst.write(data, 1)

    print(f"[INFO] Created GLOBAL template from SMAP L3 radiometer: {out_template}")
    return str(out_template)


def make_template_from_bbox(
    out_template: str,
    bbox_lonlat: tuple[float, float, float, float],
    pixel_size_m: float = 36000.0,
    crs_out: CRS = CRS.from_epsg(6933),
) -> str:
    """Create a GeoTIFF template covering a provided lon/lat bbox (EPSG:4326).

    The bounding box is transformed to the output CRS and a template with
    the requested pixel size is created (width/height are ceil'd to ensure
    coverage). This is useful when the user wants a local subset rather
    than the global grid.
    """
    lon_min, lat_min, lon_max, lat_max = bbox_lonlat

    left, bottom, right, top = transform_bounds(
        CRS.from_epsg(4326), crs_out,
        lon_min, lat_min, lon_max, lat_max,
        densify_pts=21
    )

    dx = dy = float(pixel_size_m)
    width = int(math.ceil((right - left) / dx))
    height = int(math.ceil((top - bottom) / dy))

    if width <= 0 or height <= 0:
        raise ValueError(f"BBox produced non-positive template size (h={height}, w={width}). Check bbox.")

    transform = from_origin(left, top, dx, dy)

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "height": height,
        "width": width,
        "transform": transform,
        "crs": crs_out,
        "compress": "deflate",
        "predictor": 2,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "nodata": np.nan,
    }

    out_template = Path(out_template)
    out_template.parent.mkdir(parents=True, exist_ok=True)
    data = np.full((height, width), np.nan, dtype=np.float32)
    with rasterio.open(out_template, "w", **profile) as dst:
        dst.write(data, 1)

    print(f"[INFO] Created BBOX template: {out_template} (h={height}, w={width}, px={pixel_size_m} m)")
    return str(out_template)


# -------------------------------------------------------------------------
# SMAP radiometer (L3 TB / L3 SM with TB)
# -------------------------------------------------------------------------


def _guess_tb_var_from_l3(ds: xr.Dataset) -> str | None:
    """Try heuristics to find a brightness temperature variable in a SMAP L3 group.

    Looks at variable names and attributes like `long_name` / `standard_name` to
    identify a TB-like field even when names vary across product versions.
    Returns the variable name or None if not found.
    """
    patt_names = [
        r".*tb.*corrected.*", r".*TB.*corrected.*",
        r".*tb_[hv]_.*", r".*TB_[HV]_.*",
        r".*TB.*36.*", r".*tb.*36.*",
        r".*TB.*", r".*tb.*",
        r".*Brightness.*", r".*brightness.*",
    ]
    for v in ds.data_vars:
        da = ds[v]
        if da.ndim != 2:
            continue
        ln = str(da.attrs.get("long_name", ""))
        sn = str(da.attrs.get("standard_name", ""))
        if any(re.fullmatch(p, v) for p in patt_names) or "brightness" in ln.lower() or "brightness" in sn.lower():
            return v
    return None


def load_smap_l3_tb_2d(path_l3: str, pol: str = "V", date: str | None = None, tb_group: str = "AUTO") -> np.ndarray:
    """Load a 2D brightness temperature array from a SMAP L3 product.

    The function tries multiple group names and variable name patterns; it
    also supports selecting AM/PM groups when the user prefers one pass.
    If `date` is present in a time dimension it attempts to select that time.
    """
    pol = pol.upper()
    if pol not in ("H", "V"):
        raise ValueError("pol must be 'H' or 'V'.")

    tb_group = (tb_group or "AUTO").upper()
    base_groups = [
        None,
        "Soil_Moisture_Retrieval_Data_AM",
        "Soil_Moisture_Retrieval_Data_PM",
        "Radiometer_Retrieval_Data_AM",
        "Radiometer_Retrieval_Data_PM",
        "Radiometer_Retrieval_Data",
        "Soil_Moisture_Retrieval_Data",
    ]

    if tb_group == "AM":
        pref = [None, "Soil_Moisture_Retrieval_Data_AM", "Radiometer_Retrieval_Data_AM"]
        groups_to_try = pref + [g for g in base_groups if g not in pref]
    elif tb_group == "PM":
        pref = [None, "Soil_Moisture_Retrieval_Data_PM", "Radiometer_Retrieval_Data_PM"]
        groups_to_try = pref + [g for g in base_groups if g not in pref]
    else:
        groups_to_try = base_groups

    for grp in groups_to_try:
        try:
            with xr.open_dataset(path_l3, engine="h5netcdf", phony_dims="sort", group=grp) as ds:
                if "time" in ds.dims and date is not None:
                    try:
                        ds = ds.sel(time=np.datetime64(date))
                    except Exception:
                        # If time selection fails, we continue to try other groups
                        pass

                cand_names = [
                    f"tb_{pol.lower()}_corrected",
                    f"tb{pol}_corrected", f"TB{pol}_corrected",
                    f"tb_{pol.lower()}", f"tb{pol}", f"TB{pol}",
                    f"TB_{pol}", f"TB_{pol.lower()}",
                    f"TB{pol}_36km", f"tb{pol}_36km",
                    f"brightness_temperature_{'vertical' if pol == 'V' else 'horizontal'}",
                ]
                for nm in cand_names:
                    if nm in ds:
                        return np.asarray(ds[nm].values, dtype=np.float32)

                tb_name = _guess_tb_var_from_l3(ds)
                if tb_name is not None:
                    da = ds[tb_name]
                    # Handle different dimension orders and pol dimension names
                    if "pol" in da.dims:
                        pol_vals = [str(p).upper() for p in da.coords["pol"].values]
                        if pol in pol_vals:
                            return np.asarray(da.isel(pol=pol_vals.index(pol)).values, dtype=np.float32)
                    if da.ndim == 3 and da.shape[-1] in (2, 4):
                        idx = 1 if pol == "V" else 0
                        return np.asarray(da.values[..., idx], dtype=np.float32)
                    if da.ndim == 2:
                        return np.asarray(da.values, dtype=np.float32)
        except Exception:
            continue

    raise ValueError("Could not find TB in SMAP L3 file (try SPL3TB* or SPL3SMP* V009).")


def get_TBc_2d(
    pol: str,
    date: str | None,
    template_shape: tuple[int, int],
    template_transform: Affine,
    template_crs: CRS,
    path_l3: str,
    qa: bool = False,
    tb_group: str = "AUTO",
) -> np.ndarray:
    """Return TB reprojected to a template grid.

    Attempts to read CRS/transform from the SMAP L3 file. If the original
    file contains regular x/y coordinates we build an Affine transform from
    them; otherwise we conservatively fall back to the provided template
    CRS and transform (so that reprojecting remains predictable).
    """
    TB = load_smap_l3_tb_2d(path_l3, pol=pol, date=date, tb_group=tb_group)

    src_crs = None
    src_transform = None
    try:
        with xr.open_dataset(path_l3, engine="h5netcdf", phony_dims="sort") as ds:
            grid_mapping = None
            # Find a variable that points to a grid_mapping object (CF style)
            for v in ds.data_vars:
                gm = ds[v].attrs.get("grid_mapping")
                if gm and gm in ds.variables:
                    grid_mapping = ds[gm]
                    break

            if grid_mapping is not None:
                try:
                    src_crs = CRS.from_wkt(grid_mapping.attrs.get("spatial_ref"))
                except Exception:
                    src_crs = None

            if src_crs is not None:
                vname = _guess_tb_var_from_l3(ds) or next(v for v in ds.data_vars if ds[v].ndim == 2)
                da = ds[vname]
                x_name = next((c for c in da.coords if c.lower() in ("x", "cols", "col")), None)
                y_name = next((c for c in da.coords if c.lower() in ("y", "rows", "row")), None)
                if x_name and y_name:
                    x = da.coords[x_name].values
                    y = da.coords[y_name].values
                    if is_mono_eq(x) and is_mono_eq(y):
                        dx = float(np.mean(np.diff(x)))
                        dy = float(np.mean(np.diff(y)))
                        x0 = float(x[0]) - dx / 2.0
                        y0 = float(y[0]) - dy / 2.0
                        src_transform = Affine(dx, 0.0, x0, 0.0, dy, y0)
    except Exception:
        src_crs = None
        src_transform = None

    # If we couldn't infer CRS/transform from the product, fall back to template
    if src_crs is None:
        src_crs = template_crs
    if src_transform is None:
        src_transform = template_transform

    src_arr = np.asarray(TB, dtype=np.float32)
    dst_arr = np.full(template_shape, np.nan, dtype=np.float32)

    reproject(
        source=src_arr,
        destination=dst_arr,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=template_transform,
        dst_crs=template_crs,
        resampling=ResampEnum.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )

    if qa:
        _write_debug_tif(INTERIM / "debug_TB_boulder.tif", dst_arr, template_transform, template_crs)

    return dst_arr


# -------------------------------------------------------------------------
# SMAP radar gridded σ0 (SPL3SMA etc.)
# -------------------------------------------------------------------------


def _select_radar_var(ds: xr.Dataset, pol: str) -> xr.DataArray:
    """Heuristics to pick a radar σ0 / backscatter variable from the dataset.

    Checks common name patterns first, then inspects attributes like
    `long_name`, `standard_name`, and `units` to spot backscatter-like
    variables. If a polarization dimension is found it will return the
    slice corresponding to the requested pol (VV/HH).
    """
    pol = pol.upper()
    cand_names = [
        f"sigma0_{pol.lower()}",
        f"sigma0_{pol}",
        f"Sigma0_{pol}",
        f"sigma0{pol.lower()}",
        f"Sigma0{pol}",
        f"sigma0_{pol.lower()}_db",
        f"sigma0_{pol}_db",
        f"Sigma0_{pol}_db",
    ]
    for nm in cand_names:
        if nm in ds:
            da = ds[nm]
            if da.ndim >= 2:
                return da

    for v in ds.data_vars:
        da = ds[v]
        if da.ndim < 2:
            continue
        name_lower = v.lower()
        attrs = " ".join(str(da.attrs.get(k, "")).lower() for k in ("long_name", "standard_name", "units", "description"))
        if "sigma0" in name_lower or "backscatter" in name_lower or "sigma0" in attrs:
            # Respect explicit polarization dimensions if present
            for dim in ("pol", "polarization", "polarisation"):
                if dim in da.dims:
                    pol_vals = [str(p).upper() for p in da.coords[dim].values]
                    if pol in pol_vals:
                        return da.isel({dim: pol_vals.index(pol)})
            return da

    raise ValueError("Could not find a suitable σ0/backscatter variable in SMAP radar file.")


def _get_crs_transform_from_da(da: xr.DataArray, ds: xr.Dataset) -> tuple[CRS | None, Affine | None]:
    """Attempt to derive CRS and Affine transform from an xarray DataArray.

    - Looks for a CF-style `grid_mapping` variable with `spatial_ref` or `crs_wkt`.
    - If regular x/y or easting/northing coordinates exist and are equally
      spaced, builds an Affine from them.

    Returns (crs_or_None, transform_or_None).
    """
    src_crs = None
    src_transform = None

    gm_name = da.attrs.get("grid_mapping")
    if gm_name and gm_name in ds.variables:
        gm = ds[gm_name]
        wkt = gm.attrs.get("spatial_ref") or gm.attrs.get("crs_wkt")
        if isinstance(wkt, bytes):
            wkt = wkt.decode()
        if isinstance(wkt, str):
            try:
                src_crs = CRS.from_wkt(wkt)
            except Exception:
                src_crs = None

    x_name = next((c for c in da.coords if c.lower() in ("x", "cols", "col", "easting")), None)
    y_name = next((c for c in da.coords if c.lower() in ("y", "rows", "row", "northing")), None)

    if x_name and y_name:
        x = da.coords[x_name].values
        y = da.coords[y_name].values
        if is_mono_eq(x) and is_mono_eq(y):
            dx = float(np.mean(np.diff(x)))
            dy = float(np.mean(np.diff(y)))
            x0 = float(x[0]) - dx / 2.0
            y0 = float(y[0]) - dy / 2.0
            src_transform = Affine(dx, 0.0, x0, 0.0, dy, y0)

    return src_crs, src_transform


def load_smap_radar_sigma0_2d(radar_path: str, pol: str = "VV") -> tuple[np.ndarray, Affine, CRS]:
    """Load an approximately 2D radar σ0 array (in dB) together with CRS/transform.

    The function prefers an xarray-based approach that preserves CRS and
    transform information if present; otherwise, it falls back to a HDF5
    scan heuristic (searching for datasets whose names contain 'sigma0').

    When geolocation cannot be inferred it uses a conservative EASE2 3 km
    world transform (not identity) to avoid creating invalid geo-references.
    """
    pol = pol.upper()

    try:
        with xr.open_dataset(radar_path, engine="h5netcdf", phony_dims="sort") as ds:
            da = _select_radar_var(ds, pol=pol)
            arr = np.asarray(da.values, dtype=np.float32)

            # Convert to dB if it looks linear
            if np.nanmax(arr) > 50.0:
                arr_db = to_db(arr)
            else:
                arr_db = arr

            src_crs, src_transform = _get_crs_transform_from_da(da, ds)

            if src_crs is None:
                warnings.warn("Could not infer CRS from radar root group. Assuming EPSG:6933 (EASE2).")
                src_crs = CRS.from_epsg(6933)

            if src_transform is None:
                warnings.warn("Could not infer transform from radar. Fallback to EASE2 3km global transform.")
                dx = dy = 3000.0
                src_transform = from_origin(-17367530.45, 7314540.83, dx, dy)

            return arr_db, src_transform, src_crs

    except Exception:
        pass

    # Fallback HDF5 scan: robust to products that do not follow CF / xarray patterns
    print("[WARN] Xarray-based radar σ0 detection failed. Falling back to raw HDF5 scan.")
    with open_h5(radar_path) as h5f:
        best = None  # (name, dset, score)

        def visitor(name, obj):
            nonlocal best
            if not isinstance(obj, h5py.Dataset) or obj.ndim < 2:
                return
            lname = name.lower()
            if "sigma0" not in lname:
                return
            if "flag" in lname or "quality" in lname or "qc" in lname:
                return

            score = 0
            if pol == "VV" and "vv" in lname:
                score += 5
            if pol == "HH" and "hh" in lname:
                score += 5
            if "mean" in lname:
                score += 2
            if "std" in lname or "stdev" in lname:
                score -= 1
            if best is None or score > best[2]:
                best = (name, obj, score)

        h5f.visititems(visitor)

        if best is None:
            raise ValueError("Could not find a suitable σ0/backscatter dataset in SMAP radar file.")

        name, dset, score = best
        print(f"[INFO] Using radar σ0 dataset '{name}' with score={score}")
        arr = dset[...].astype(np.float32)

        if np.nanmax(arr) > 50.0:
            arr_db = to_db(arr)
        else:
            arr_db = arr

        if arr_db.ndim != 2:
            arr_db = arr_db.reshape(arr_db.shape[-2], arr_db.shape[-1])

        crs = CRS.from_epsg(6933)
        dx = dy = 3000.0
        transform = from_origin(-17367530.45, 7314540.83, dx, dy)
        return arr_db, transform, crs


# -------------------------------------------------------------------------
# SMAP L1C XPOL SWATH
# -------------------------------------------------------------------------


def load_smap_l1c_xpol_swath(
    radar_l1c_path: str,
    which: str = "aft",
    apply_qual_flags: bool = True,
    convert_to_db: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load XPOL swath arrays and their cell lon/lat coordinates from an L1C file.

    Returns (sigma, lon, lat). If `apply_qual_flags` is True and a
    `cell_sigma0_qual_flag_xpol` dataset exists, non-zero flags are set to
    NaN in the returned sigma array. Optionally converts linear sigma to dB.
    """
    which = (which or "aft").lower()
    if which not in ("aft", "fore", "mean"):
        raise ValueError("which must be 'aft', 'fore', or 'mean'")

    with open_h5(radar_l1c_path) as f:
        grp = f["Sigma0_Data"]

        sig_aft = np.asarray(grp["cell_sigma0_xpol_aft"][...], dtype=np.float32)
        sig_fore = np.asarray(grp["cell_sigma0_xpol_fore"][...], dtype=np.float32)

        if which == "aft":
            sigma = sig_aft
        elif which == "fore":
            sigma = sig_fore
        else:
            sigma = np.nanmean(np.stack([sig_aft, sig_fore], axis=0), axis=0).astype(np.float32)

        lon = np.asarray(grp["cell_lon"][...], dtype=np.float64)
        lat = np.asarray(grp["cell_lat"][...], dtype=np.float64)

        if apply_qual_flags and "cell_sigma0_qual_flag_xpol" in grp:
            qual = np.asarray(grp["cell_sigma0_qual_flag_xpol"][...], dtype=np.uint16)
            sigma = sigma.astype(np.float32, copy=False)
            sigma[qual != 0] = np.nan

    # Clean common fill values and enforce sensible limits
    for fv in (0, -9999, -999, -32768, 65535):
        sigma[sigma == fv] = np.nan

    if convert_to_db:
        sigma = to_db(sigma, eps=1e-8)

    sigma = sigma.astype(np.float32, copy=False)
    sigma[~np.isfinite(sigma)] = np.nan
    sigma[sigma < -80.0] = np.nan
    sigma[sigma > 20.0] = np.nan

    return sigma, lon, lat


# -------------------------------------------------------------------------
# Reprojection helpers
# -------------------------------------------------------------------------


def reproject_to_template(
    src_arr: np.ndarray,
    src_transform: Affine,
    src_crs: CRS,
    tpl_transform: Affine,
    tpl_crs: CRS,
    tpl_shape: tuple[int, int],
    resampling: ResampEnum = ResampEnum.bilinear,
) -> np.ndarray:
    """Reproject `src_arr` to `tpl_crs`/`tpl_transform` with requested resampling.

    Returns an array of shape `tpl_shape` with NaNs where data is not present.
    """
    dst = np.full(tpl_shape, np.nan, dtype=np.float32)
    reproject(
        source=src_arr,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=tpl_transform,
        dst_crs=tpl_crs,
        resampling=resampling,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    return dst


def swath_average_to_template(
    swath_val: np.ndarray,
    swath_lon: np.ndarray,
    swath_lat: np.ndarray,
    tpl_transform: Affine,
    tpl_crs: CRS,
    tpl_shape: tuple[int, int],
    src_crs: CRS = CRS.from_epsg(4326),
) -> np.ndarray:
    """Bin a swath of point samples into template pixels using simple averaging.

    This is a crude but effective way to convert swath cell-level products to
    a regular grid: we transform cell lon/lat to the template CRS, compute
    which pixel each sample falls into, and average values per-pixel.
    """
    h, w = tpl_shape
    dst_sum = np.zeros((h, w), dtype=np.float64)
    dst_cnt = np.zeros((h, w), dtype=np.int32)

    m = np.isfinite(swath_val) & np.isfinite(swath_lon) & np.isfinite(swath_lat)
    if not np.any(m):
        return np.full((h, w), np.nan, dtype=np.float32)

    v = swath_val[m].astype(np.float64)
    lon = swath_lon[m].astype(np.float64)
    lat = swath_lat[m].astype(np.float64)

    xs, ys = rio_transform(src_crs, tpl_crs, lon.tolist(), lat.tolist())
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)

    inv = ~tpl_transform
    cols, rows = inv * (xs, ys)
    cols = np.floor(cols).astype(np.int64)
    rows = np.floor(rows).astype(np.int64)

    inside = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    if not np.any(inside):
        return np.full((h, w), np.nan, dtype=np.float32)

    rows = rows[inside]
    cols = cols[inside]
    v = v[inside]

    # Efficient per-pixel accumulation using numpy.add.at
    np.add.at(dst_sum, (rows, cols), v)
    np.add.at(dst_cnt, (rows, cols), 1)

    out = np.full((h, w), np.nan, dtype=np.float32)
    ok = dst_cnt > 0
    out[ok] = (dst_sum[ok] / dst_cnt[ok]).astype(np.float32)
    return out


# -------------------------------------------------------------------------
# Pipeline
# -------------------------------------------------------------------------


def prepare_smap_radiometer_radar(args) -> tuple[np.ndarray, np.ndarray]:
    """Main pipeline to prepare aligned SMAP TB and radar σ0 on a template.

    Returns (TBc_2d, S_pp_db) where both arrays are on the same grid. Side
    effects: writes an NPZ to `data/interim` with arrays and metadata.
    The function implements safe fallbacks so it can be used in tests where
    full geolocation metadata might not be available.
    """
    S_xpol_db = None

    # 0) TEMPLATE: create from bbox if provided, otherwise use or build global template
    if args.bbox is not None:
        # If user provided --template, we create it if missing; else use a default bbox template name
        tpl_path = Path(args.template) if args.template else (PROCESSED / "template_bbox_ease2.tif")
        if not tpl_path.exists():
            tpl_path = Path(make_template_from_bbox(
                out_template=str(tpl_path),
                bbox_lonlat=tuple(args.bbox),
                pixel_size_m=float(args.pixel_size),
                crs_out=CRS.from_epsg(6933),
            ))
    else:
        tpl_path = Path(args.template) if args.template else (PROCESSED / "SMAP_L3_template_36km_from_rad.tif")
        if not tpl_path.exists():
            print(f"[INFO] Template not found → creating GLOBAL 36 km template from SMAP L3 radiometer: {tpl_path}")
            tpl_path = Path(make_template_from_smap_l3(args.smap_l3_rad, str(tpl_path)))

    tpl_crs, tpl_transform, tpl_h, tpl_w, _ = load_template(str(tpl_path))
    tpl_shape = (tpl_h, tpl_w)
    args.template = str(tpl_path)

    # Optional: warn if pixel size is unexpected
    dx = float(tpl_transform.a)
    dy = float(-tpl_transform.e)
    print(f"[INFO] Template px size: dx={dx:.1f} m, dy={dy:.1f} m, shape={tpl_shape}")

    # 1) TB
    TBc_2d = get_TBc_2d(
        pol=args.pol,
        date=args.date,
        template_shape=tpl_shape,
        template_transform=tpl_transform,
        template_crs=tpl_crs,
        path_l3=args.smap_l3_rad,
        qa=args.qa,
        tb_group=args.tb_group,
    ).astype(np.float32)
    TBc_2d[~np.isfinite(TBc_2d)] = np.nan

    # 2) Radar gridded σ0
    S_pp_db_src, Spp_tr, Spp_crs = load_smap_radar_sigma0_2d(args.smap_radar, pol=args.radar_pol)
    S_pp_db = reproject_to_template(
        S_pp_db_src,
        Spp_tr,
        Spp_crs,
        tpl_transform,
        tpl_crs,
        tpl_shape,
        resampling=ResampEnum.average,
    ).astype(np.float32)

    # Mask non-physical values
    S_pp_db[~np.isfinite(S_pp_db)] = np.nan
    S_pp_db[S_pp_db < -60.0] = np.nan
    S_pp_db[S_pp_db > 5.0] = np.nan

    if args.qa:
        _write_debug_tif(INTERIM / "debug_radar_boulder.tif", S_pp_db, tpl_transform, tpl_crs)

    # 2b) L1C XPOL swath -> template (binning avg)
    if args.smap_l1c_xpol is not None:
        print(f"[INFO] Cargando cross-pol (XPOL) SWATH desde L1C: {args.smap_l1c_xpol}")

        xpol_db_swath, lon2d, lat2d = load_smap_l1c_xpol_swath(
            args.smap_l1c_xpol,
            which="aft",
            apply_qual_flags=True,
            convert_to_db=True,
        )

        S_xpol_db = swath_average_to_template(
            xpol_db_swath, lon2d, lat2d,
            tpl_transform, tpl_crs, tpl_shape,
            src_crs=CRS.from_epsg(4326),
        ).astype(np.float32)

        S_xpol_db[~np.isfinite(S_xpol_db)] = np.nan
        S_xpol_db[S_xpol_db < -80.0] = np.nan
        S_xpol_db[S_xpol_db > 20.0] = np.nan

        if args.qa:
            _write_debug_tif(INTERIM / "debug_radar_xpol_boulder.tif", S_xpol_db, tpl_transform, tpl_crs)

    # 3) Save NPZ
    date_str = (args.date or "nodate").replace("-", "")
    out_npz = INTERIM / f"aligned-smap-{date_str}.npz"

    meta_list = [
        f"smap_l3_rad={Path(args.smap_l3_rad).name}",
        f"smap_radar={Path(args.smap_radar).name}",
        f"pol_tb={args.pol}",
        f"pol_radar={args.radar_pol}",
        f"date={args.date}",
        f"template={Path(tpl_path).name}",
    ]
    if args.bbox is not None:
        meta_list.append(f"bbox={args.bbox}")
        meta_list.append(f"pixel_size_m={args.pixel_size}")
    if S_xpol_db is not None and args.smap_l1c_xpol is not None:
        meta_list.append(f"smap_l1c_xpol={Path(args.smap_l1c_xpol).name}")

    save_kwargs = dict(
        TBc_2d=TBc_2d,
        S_pp_dB=S_pp_db,
        crs_wkt=tpl_crs.to_wkt(),
        transform=np.array(tpl_transform, dtype=np.float64),
        height=np.int32(tpl_h),
        width=np.int32(tpl_w),
        meta=np.array(meta_list, dtype=object),
    )
    if S_xpol_db is not None:
        save_kwargs["S_xpol_dB"] = S_xpol_db

    np.savez_compressed(out_npz, **save_kwargs)
    print(f"[OK] Saved {out_npz}")

    # Quick sanity print
    print(f"[STAT] TB finite: {np.isfinite(TBc_2d).sum()} / {TBc_2d.size}")
    print(f"[STAT] Spp finite: {np.isfinite(S_pp_db).sum()} / {S_pp_db.size}")
    if S_xpol_db is not None:
        print(f"[STAT] Xpol finite: {np.isfinite(S_xpol_db).sum()} / {S_xpol_db.size}")

    return TBc_2d, S_pp_db


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Build argument parser for this script.

    Provides options to select date, polarization, input paths, and whether
    to generate an explicit bbox template. Also supports debug QA output.
    """
    p = argparse.ArgumentParser(description="Align SMAP radiometer TB and SMAP radar σ0 on a common grid.")
    p.add_argument("--date", type=str, required=False, help="Date YYYY-MM-DD (optional).")
    p.add_argument("--pol", type=str, default="V", help="SMAP TB polarisation: H or V (default: V).")

    p.add_argument("--smap-l3-rad", type=str, required=True,
                   help="Path to SMAP L3 radiometer file (SPL3TB* or SPL3SMP* with TB).")

    p.add_argument("--smap-radar", type=str, required=True,
                   help="Path to SMAP radar gridded backscatter file (σ0, e.g. SPL3SMA).")

    p.add_argument("--radar-pol", type=str, default="VV", help="SMAP radar co-pol: VV or HH (default: VV).")

    p.add_argument("--smap-l1c-xpol", type=str, required=False,
                   help="SMAP L1C S0 HiRes file to extract XPOL swath and bin to template grid.")

    p.add_argument("--tb-group", type=str, choices=["AUTO", "AM", "PM"], default="AUTO",
                   help="TB group preference in L3 SM: AM, PM, or AUTO (default AUTO).")

    p.add_argument("--template", type=str, required=False, default=None,
                   help="GeoTIFF template path. If missing, created automatically.")

    # ✅ NEW: bbox template
    p.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
        help="Bounding box in lon/lat (EPSG:4326). If provided, template covers ONLY this area.",
    )
    p.add_argument(
        "--pixel-size",
        type=float,
        default=36000.0,
        help="Template pixel size in meters (default 36000). Used when --bbox is provided.",
    )

    p.add_argument("--qa", action="store_true", help="Save debug GeoTIFFs for TB and radar.")
    args = p.parse_args()

    args.pol = args.pol.upper()
    args.radar_pol = args.radar_pol.upper()
    return args


def main() -> None:
    args = parse_args()
    try:
        prepare_smap_radiometer_radar(args)
    except Exception as e:
        warnings.warn(f"[ERROR] {e}")
        raise


if __name__ == "__main__":
    main()
