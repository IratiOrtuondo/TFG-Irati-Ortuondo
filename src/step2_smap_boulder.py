#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step2_smap_boulder.py — Prepare SMAP radiometer TB and SMAP radar backscatter
aligned on a common grid (e.g. for a small region like Boulder).

This script mirrors the logic of step1.py (SMAP + NISAR), but uses:
    - SMAP radiometer (L3 TB or L3 SM with embedded TB), and
    - SMAP radar (gridded backscatter, e.g. σ0) instead of NISAR GCOV.

Assumed folder structure (project root):
    TFG-NISAR/
        data/
            raw/         ← inputs (.h5/.nc)
            interim/     ← temporary/QA outputs
            processed/   ← templates, etc.
        src/
            step1.py
            step2_smap_boulder.py  ← this file

Typical inputs:
    - SMAP L3 TB/SM file with TB (radiometer, e.g. SPL3SMP)
    - SMAP radar gridded backscatter file (σ0, same date/overpass, e.g. SPL3SMA)

Template handling:
    - If --template exists → it is used directly.
    - If --template is missing or does not exist → a global EASE2 36 km template
      is automatically created from the SMAP radiometer file.

Outputs (by default):
    - data/interim/debug_TB_boulder.tif        (if --qa)
    - data/interim/debug_radar_boulder.tif     (if --qa)
    - data/interim/aligned_step2_smap.npz with:
        TBc_2d     : brightness temperature on template grid (K, float32, NaN=nodata)
        S_pp_dB    : backscatter σ0 in dB on template grid (float32, NaN=nodata)
        crs_wkt    : WKT of template CRS
        transform  : affine transform as 6-element array
        height     : grid height (int32)
        width      : grid width (int32)
        meta       : metadata strings (products, polarisations, date)

Example usage:
    python step2_smap_boulder.py \
        --date 2015-06-15 \
        --pol V \
        --smap-l3-rad ../data/raw/SMAP_L3_SM_P_20150615_R19240_001.h5 \
        --smap-radar  ../data/raw/SMAP_L3_SM_A_20150615_R13080_001.h5 \
        --radar-pol VV \
        --template ../data/processed/SMAP_L3_template_from_rad.tif \
        --qa
"""

from __future__ import annotations

import argparse
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

# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------

# If this file is inside TFG-NISAR/src/, ROOT is the project root (one level up).
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
INTERIM.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------

def to_db(arr: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Convert power array to decibels (dB) with clipping and float32 output."""
    return (10.0 * np.log10(np.clip(arr, eps, None))).astype(np.float32)


def is_mono_eq(v: np.ndarray, rtol: float = 1e-6, atol: float = 1e-6) -> bool:
    """Check if a 1D array is monotonic and equally spaced."""
    if v.ndim != 1 or v.size < 2:
        return False
    d = np.diff(v.astype(np.float64))
    return np.all(d > -atol) and np.allclose(d, d[0], rtol=rtol, atol=atol)


def open_h5(path: str) -> h5py.File:
    """Open HDF5 file, raising if missing."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File does not exist: {p}")
    return h5py.File(p, "r")


def _write_debug_tif(path: Path, arr: np.ndarray, transform: Affine, crs: CRS) -> None:
    """Write a single-band float32 GeoTIFF for QA."""
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


# -----------------------------------------------------------------------------
# Template handling
# -----------------------------------------------------------------------------

def load_template(template_path: str) -> tuple[CRS, Affine, int, int, dict]:
    """Load a GeoTIFF template (EASE2 or UTM)."""
    with rasterio.open(template_path) as ds:
        crs = ds.crs
        transform = ds.transform
        height, width = ds.height, ds.width
        profile = ds.profile
    return crs, transform, height, width, profile


def make_template_from_smap_l3(smap_l3_path: str, out_template: str) -> str:
    """
    Create a simple EASE2-like template GeoTIFF from a SMAP L3 radiometer file.

    - Intenta usar la forma (ny, nx) del primer campo 2D que encuentre
      en el root o en grupos típicos de SMAP L3 (Soil_Moisture_Retrieval_Data_*).
    - Si no encuentra nada, cae a una forma típica global EASE2 36 km (~406x964).
    - Asume EPSG:6933 (EASE-Grid 2.0) y resolución ~36 km.
    """
    ny = nx = None

    # 1) Buscar variables 2D en root
    try:
        with xr.open_dataset(smap_l3_path, engine="h5netcdf", phony_dims="sort") as ds_root:
            cand_vars = [v for v in ds_root.data_vars if ds_root[v].ndim == 2]
            if cand_vars:
                v0 = ds_root[cand_vars[0]]
                ny, nx = v0.shape
                print(f"[INFO] Template shape from root group: ny={ny}, nx={nx}")
    except Exception as e:
        print(f"[WARN] Could not read root group of SMAP L3: {e}")

    # 2) Si no hay nada en root, probar grupos típicos de SMAP L3
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
                with xr.open_dataset(
                    smap_l3_path, engine="h5netcdf", phony_dims="sort", group=grp
                ) as ds:
                    cand_vars = [v for v in ds.data_vars if ds[v].ndim == 2]
                    if cand_vars:
                        v0 = ds[cand_vars[0]]
                        ny, nx = v0.shape
                        print(f"[INFO] Template shape from group '{grp}': ny={ny}, nx={nx}")
                        break
            except Exception:
                continue

    # 3) Fallback duro: forma típica EASE2 36 km global
    if ny is None or nx is None:
        ny, nx = 406, 964
        warnings.warn(
            "No 2D variables found in SMAP L3 to derive template shape. "
            "Using fallback EASE2 36 km global size ny=406, nx=964."
        )
    crs = CRS.from_epsg(6933)  # EASE-Grid 2.0 global
    dx = dy = 36000.0          # 36 km
    # Extensión típica de EASE2 36 km global
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

    print(f"[INFO] Created template from SMAP L3 radiometer: {out_template}")
    return str(out_template)
    """
    Create a simple EASE2-like template GeoTIFF from a SMAP L3 radiometer file.

    - Uses the shape (ny, nx) of the first 2D variable found.
    - Assumes EASE2 global equal-area projection (EPSG:6933) with 36 km pixels.
      This is sufficient for TB + σ0 experiments on the SMAP 36 km grid.
    """
    with xr.open_dataset(smap_l3_path, engine="h5netcdf", phony_dims="sort") as ds:
        cand_vars = [v for v in ds.data_vars if ds[v].ndim == 2]
        if not cand_vars:
            raise RuntimeError(
                "No 2D variables found in the SMAP L3 file to derive the template shape."
            )
        v0 = ds[cand_vars[0]]
        ny, nx = v0.shape

    crs = CRS.from_epsg(6933)  # EASE-Grid 2.0 global
    dx = dy = 36000.0          # 36 km
    # Standard EASE2 36 km global extent (used as a reasonable default)
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

    print(f"[INFO] Created template from SMAP L3 radiometer: {out_template}")
    return str(out_template)


# -----------------------------------------------------------------------------
# SMAP radiometer (L3 TB / L3 SM with TB)
# -----------------------------------------------------------------------------

def _guess_tb_var_from_l3(ds: xr.Dataset) -> str | None:
    """Heuristic search for a TB variable in a SMAP L3 dataset."""
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
        name = v
        ln = str(da.attrs.get("long_name", ""))
        sn = str(da.attrs.get("standard_name", ""))
        hay = any(re.fullmatch(p.replace(".*", ".*"), name) for p in patt_names) \
              or "brightness" in ln.lower() or "brightness" in sn.lower()
        if hay:
            return v
    return None


def load_smap_l3_tb_2d(
    path_l3: str,
    pol: str = "V",
    date: str | None = None,
    tb_group: str = "AUTO",
) -> np.ndarray:
    """Load 2D TB (K) from a SMAP L3 file (TB or SM with embedded TB)."""
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
                        da = ds[nm]
                        arr = np.asarray(da.values, dtype=np.float32)
                        return arr

                tb_name = _guess_tb_var_from_l3(ds)
                if tb_name is not None:
                    da = ds[tb_name]
                    if "pol" in da.dims:
                        pol_vals = [str(p).upper() for p in da.coords["pol"].values]
                        if pol in pol_vals:
                            idx = pol_vals.index(pol)
                            arr = da.isel(pol=idx).values
                            return np.asarray(arr, dtype=np.float32)
                    if da.ndim == 3 and da.shape[-1] in (2, 4):
                        idx = 1 if pol == "V" else 0
                        arr = da.values[..., idx]
                        return np.asarray(arr, dtype=np.float32)
                    if da.ndim == 2:
                        return np.asarray(da.values, dtype=np.float32)
        except Exception:
            continue

    raise ValueError(
        "Could not find TB in the given SMAP L3 file. "
        "Check that it is SPL3TB* or SPL3SMP* V009 with tb_h_corrected/tb_v_corrected."
    )


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
    """Get TBc_2d (K) resampled to the template grid."""
    TB = load_smap_l3_tb_2d(path_l3, pol=pol, date=date, tb_group=tb_group)

    # Try to read CRS/transform from the L3 file; if not, assume same as template.
    src_crs = None
    src_transform = None
    try:
        with xr.open_dataset(path_l3, engine="h5netcdf", phony_dims="sort") as ds:
            grid_mapping = None
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
                vname = _guess_tb_var_from_l3(ds) or next(
                    v for v in ds.data_vars if ds[v].ndim == 2
                )
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


# -----------------------------------------------------------------------------
# SMAP radar backscatter (gridded σ0)
# -----------------------------------------------------------------------------

def _select_radar_var(ds: xr.Dataset, pol: str) -> xr.DataArray:
    """Heuristically select a radar backscatter variable for the given polarisation."""
    pol = pol.upper()

    # 1) Direct name patterns: sigma0_VV, Sigma0_VV, sigma0_vv, etc.
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

    # 2) Variables with a polarisation dimension
    for v in ds.data_vars:
        da = ds[v]
        if da.ndim < 2:
            continue
        name_lower = v.lower()
        attrs = " ".join(
            str(ds[v].attrs.get(k, "")).lower()
            for k in ("long_name", "standard_name", "units", "description")
        )
        if "sigma0" in name_lower or "backscatter" in name_lower:
            # Check for pol dim
            for dim in ("pol", "polarization", "polarisation"):
                if dim in da.dims:
                    pol_vals = [str(p).upper() for p in da.coords[dim].values]
                    if pol in pol_vals:
                        idx = pol_vals.index(pol)
                        return da.isel({dim: idx})
            # If no pol dim, just return this one
            return da

    raise ValueError("Could not find a suitable σ0/backscatter variable in SMAP radar file.")


def _get_crs_transform_from_da(da: xr.DataArray, ds: xr.Dataset) -> tuple[CRS | None, Affine | None]:
    """Extract CRS and affine transform from a CF-style DataArray and its Dataset."""
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

    # try x/y coords
    x_name = next(
        (c for c in da.coords if c.lower() in ("x", "cols", "col", "easting")),
        None,
    )
    y_name = next(
        (c for c in da.coords if c.lower() in ("y", "rows", "row", "northing")),
        None,
    )

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
def load_smap_radar_sigma0_2d(
    radar_path: str,
    pol: str = "VV",
) -> tuple[np.ndarray, Affine, CRS]:
    """Load SMAP radar σ0 (dB) and geolocation.

    Strategy:
      1) Primero intenta con xarray en el grupo raíz (por si las vars están allí).
      2) Si no funciona, recorre todo el HDF5 con h5py y busca cualquier dataset
         2D cuyo nombre contenga 'sigma0' y la polarización adecuada (vv/hh/xpol).
      3) Asume rejilla EASE2 3 km (EPSG:6933) para la geolocalización aproximada.
    """
    pol = pol.upper()

    # ------------------------------------------------------------------
    # 1) Intento "fino": xarray en el grupo raíz
    # ------------------------------------------------------------------
    try:
        with xr.open_dataset(radar_path, engine="h5netcdf", phony_dims="sort") as ds:
            try:
                da = _select_radar_var(ds, pol=pol)
                arr = np.asarray(da.values, dtype=np.float32)

                # Convertir a dB si parece lineal
                if np.nanmax(arr) > 50.0:
                    arr_db = to_db(arr)
                else:
                    arr_db = arr

                src_crs, src_transform = _get_crs_transform_from_da(da, ds)

                if src_crs is None:
                    warnings.warn(
                        "Could not infer CRS from SMAP radar root group. "
                        "Assuming EPSG:6933 (EASE2)."
                    )
                    src_crs = CRS.from_epsg(6933)
                if src_transform is None:
                    warnings.warn(
                        "Could not infer transform from SMAP radar root group. "
                        "Reprojection might be approximate."
                    )
                    src_transform = Affine.identity()

                return arr_db, src_transform, src_crs

            except Exception:
                # si _select_radar_var falla, pasamos al fallback
                pass
    except Exception:
        # si ni siquiera se puede abrir con xarray, también pasamos al fallback
        pass

    # ------------------------------------------------------------------
    # 2) Fallback robusto: recorrer todo el HDF5 con h5py
    # ------------------------------------------------------------------
    print("[WARN] Xarray-based radar σ0 detection failed. Falling back to raw HDF5 scan.")
    with open_h5(radar_path) as h5f:
        best = None  # (name, dset, score)

        def visitor(name, obj):
            nonlocal best
            if not isinstance(obj, h5py.Dataset):
                return
            if obj.ndim < 2:
                return

            lname = name.lower()

            # Debe contener sigma0 y no ser flags/QC claramente
            if "sigma0" not in lname:
                return
            if "flag" in lname or "quality" in lname or "qc" in lname:
                return

            score = 0
            # Polarización deseada
            if pol == "VV" and "vv" in lname:
                score += 5
            if pol == "HH" and "hh" in lname:
                score += 5
            if pol in ("HV", "VH") and ("xpol" in lname or "hv" in lname or "vh" in lname):
                score += 5

            # Prefiere medias a std dev
            if "mean" in lname:
                score += 2
            if "std" in lname or "stdev" in lname:
                score -= 1

            # Si no se menciona explícitamente pol, le damos algo de puntuación igual
            if "vv" not in lname and "hh" not in lname and "xpol" not in lname:
                score += 1

            if best is None or score > best[2]:
                best = (name, obj, score)

        h5f.visititems(visitor)

        if best is None or best[2] < 0:
            raise ValueError("Could not find a suitable σ0/backscatter dataset in SMAP radar file.")

        name, dset, score = best
        print(f"[INFO] Using radar σ0 dataset '{name}' with score={score}")
        arr = dset[...].astype(np.float32)

        # Convertir a dB si parece lineal
        if np.nanmax(arr) > 50.0:
            arr_db = to_db(arr)
        else:
            arr_db = arr

        # Suponemos grid 3 km EASE2 global (coherente con SPL3SMA aprox)
        if arr_db.ndim != 2:
            # Si por algún motivo viniera con dims extra, cogemos la última 2D
            arr_db = arr_db.reshape(arr_db.shape[-2], arr_db.shape[-1])

        ny, nx = arr_db.shape
        crs = CRS.from_epsg(6933)
        dx = dy = 3000.0  # 3 km
        transform = from_origin(-17367530.45, 7314540.83, dx, dy)

        return arr_db, transform, crs



# -----------------------------------------------------------------------------
# Reprojection helper (same idea as step1.py)
# -----------------------------------------------------------------------------

def reproject_to_template(
    src_arr: np.ndarray,
    src_transform: Affine,
    src_crs: CRS,
    tpl_transform: Affine,
    tpl_crs: CRS,
    tpl_shape: tuple[int, int],
    resampling: ResampEnum = ResampEnum.bilinear,
) -> np.ndarray:
    """Reproject source array to the template grid."""
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


# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------

def prepare_smap_radiometer_radar(args) -> tuple[np.ndarray, np.ndarray]:
    """Prepare and align SMAP TB and SMAP radar σ0 to a common grid."""
    # 0) Template: if missing, create it from radiometer L3
    if args.template and Path(args.template).exists():
        tpl_path = args.template
    else:
        default_tpl = PROCESSED / "SMAP_L3_template_from_rad.tif"
        out_tpl = args.template or str(default_tpl)
        print(f"[INFO] Template not found → creating from SMAP L3 radiometer: {out_tpl}")
        tpl_path = make_template_from_smap_l3(args.smap_l3_rad, out_tpl)
        args.template = tpl_path

    tpl_crs, tpl_transform, tpl_h, tpl_w, _ = load_template(tpl_path)
    tpl_shape = (tpl_h, tpl_w)

    # 1) SMAP radiometer TB
    TBc_2d = get_TBc_2d(
        pol=args.pol,
        date=args.date,
        template_shape=tpl_shape,
        template_transform=tpl_transform,
        template_crs=tpl_crs,
        path_l3=args.smap_l3_rad,
        qa=args.qa,
        tb_group=args.tb_group,
    )
    TBc_2d = TBc_2d.astype(np.float32)
    TBc_2d[~np.isfinite(TBc_2d)] = np.nan

    # 2) SMAP radar σ0
    S_pp_db_src, Spp_tr, Spp_crs = load_smap_radar_sigma0_2d(
        args.smap_radar, pol=args.radar_pol
    )
    S_pp_db = reproject_to_template(
        S_pp_db_src,
        Spp_tr,
        Spp_crs,
        tpl_transform,
        tpl_crs,
        tpl_shape,
        resampling=ResampEnum.bilinear,
    )



        # ---------------------------------------------------------
    # MASK RADAR SIGMA0 → eliminar valores no físicos
    # ---------------------------------------------------------
    S_pp_db = S_pp_db.astype(np.float32)
    S_pp_db[~np.isfinite(S_pp_db)] = np.nan

    # σ0 físico suele caer entre -30 dB y -5 dB
    # Todo lo que esté fuera suele ser nodata o error
    S_pp_db[S_pp_db < -60.0] = np.nan     # -100 dB y similares
    S_pp_db[S_pp_db > 5.0] = np.nan       # valores imposibles (>0 dB)

    if args.qa:
        _write_debug_tif(
            INTERIM / "debug_radar_boulder.tif",
            S_pp_db,
            tpl_transform,
            tpl_crs,
        )

    # 3) Save NPZ
    out_npz = INTERIM / "aligned_step2_smap.npz"
    np.savez_compressed(
        out_npz,
        TBc_2d=TBc_2d,
        S_pp_dB=S_pp_db,
        crs_wkt=tpl_crs.to_wkt(),
        transform=np.array(tpl_transform, dtype=np.float64),
        height=np.int32(tpl_h),
        width=np.int32(tpl_w),
        meta=np.array(
            [
                f"smap_l3_rad={Path(args.smap_l3_rad).name}",
                f"smap_radar={Path(args.smap_radar).name}",
                f"pol_tb={args.pol}",
                f"pol_radar={args.radar_pol}",
                f"date={args.date}",
                f"template={Path(tpl_path).name}",
            ],
            dtype=object,
        ),
    )
    print(f"[OK] Saved {out_npz}")
    return TBc_2d, S_pp_db


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for SMAP radiometer/radar alignment."""
    p = argparse.ArgumentParser(
        description=(
            "Align SMAP radiometer TB and SMAP radar σ0 on a common grid "
            "(e.g. for a small region like Boulder)."
        )
    )
    p.add_argument(
        "--date",
        type=str,
        required=False,
        help="Date YYYY-MM-DD (if applicable for multi-time products).",
    )
    p.add_argument(
        "--pol",
        type=str,
        default="V",
        help="SMAP TB polarisation: H or V (default: V).",
    )
    p.add_argument(
        "--smap-l3-rad",
        type=str,
        required=True,
        help="Path to SMAP L3 radiometer file (SPL3TB* or SPL3SMP* with TB).",
    )
    p.add_argument(
        "--tb-group",
        type=str,
        choices=["AUTO", "AM", "PM"],
        default="AUTO",
        help="TB group preference in L3 SM: AM, PM, or AUTO (default AUTO).",
    )
    p.add_argument(
        "--smap-radar",
        type=str,
        required=True,
        help="Path to SMAP radar gridded backscatter file (σ0, e.g. SPL3SMA).",
    )
    p.add_argument(
        "--radar-pol",
        type=str,
        default="VV",
        help="SMAP radar co-pol: VV or HH (default: VV).",
    )
    p.add_argument(
        "--template",
        type=str,
        required=False,
        default=None,
        help=(
            "GeoTIFF template for output grid. "
            "If missing or non-existent, it is automatically created from the SMAP L3 radiometer."
        ),
    )
    p.add_argument(
        "--qa",
        action="store_true",
        help="Save debug GeoTIFFs for TB and radar.",
    )

    args = p.parse_args()
    args.pol = args.pol.upper()
    args.radar_pol = args.radar_pol.upper()
    return args


def main() -> None:
    """Main entry point for SMAP-only experiment."""
    args = parse_args()
    try:
        prepare_smap_radiometer_radar(args)
    except Exception as e:
        warnings.warn(f"[ERROR] {e}")
        raise


if __name__ == "__main__":
    main()
