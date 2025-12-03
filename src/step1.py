#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step1.py — Prepares SMAP (TB @36 km) and NISAR L2 GCOV (σ0/γ0) aligned to EASE2 36 km grid.

Assumed folder structure:
    TFG-NISAR/
        data/
            raw/         ← inputs (.h5/.nc)
            interim/     ← temporary/QA outputs
            processed/   ← UTM 1km template (recommended), or EASE2 36km template (legacy)
        step1.py

Typical inputs:
    - SMAP L3 TB (e.g., SPL3TB*) or SMAP L3 SM (SPL3SMP*) if it contains embedded TB
    - NISAR L2 PR GCOV *.h5 with backscatter (σ0 or γ0)
        - UTM 1km template (GeoTIFF, recommended). Generate with make_utm_template.py from NISAR GCOV.
        - EASE2_36km template (GeoTIFF, legacy). If missing, can be created from SMAP L3.

Outputs:
    - data/interim/debug_TB.tif, debug_VV.tif, debug_VH.tif (if --qa)
    - data/interim/aligned_step1.npz with:
            TBc_2d  : corrected/collocated TB (K) in EASE2 36km (float32, NaN=nodata)
            S_pp_dB : co-pol backscatter (VV or HH) in dB (float32, NaN=nodata)
            S_pq_dB : cross-pol backscatter (VH or HV) in dB (float32, NaN=nodata)
            crs_wkt, transform, height, width (grid metadata)

Example usage:
    python step1.py --date 2021-06-01 --pol V \
            --smap-l3 data/raw/SMAP_L3_SM_P_20210601_R19240_001.h5 \
            --nisar   data/raw/NISAR_L2_PR_GCOV.h5 \
            --template data/processed/UTM_1km_template.tif \
            --nisar-pol VV --nisar-sigma0 --qa
"""

import argparse
import glob
import os
import re
import warnings
from pathlib import Path

import h5py
import numpy as np
import rasterio
import xarray as xr
from affine import Affine
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import reproject
from rasterio.enums import Resampling as ResampEnum
from pyproj import CRS as PJCRS


# -----------------------------
# Rutas proyecto
# -----------------------------
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
INTERIM.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Utilidades básicas
# -----------------------------
def to_db(arr: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Convert power array to decibels (dB) with clipping and float32 output.

    Args:
        arr: Input array of power values.
        eps: Minimum value to clip to (prevents log(0)).

    Returns:
        np.ndarray: Array in dB, dtype float32.
    """
    return (10.0 * np.log10(np.clip(arr, eps, None))).astype(np.float32)

def is_mono_eq(v: np.ndarray, rtol: float = 1e-6, atol: float = 1e-6) -> bool:
    """Check if a 1D array is monotonic and equally spaced.

    Args:
        v: Input 1D array.
        rtol: Relative tolerance for spacing equality.
        atol: Absolute tolerance for monotonicity.

    Returns:
        bool: True if monotonic and equally spaced, False otherwise.
    """
    if v.ndim != 1 or v.size < 2:
        return False
    d = np.diff(v.astype(np.float64))
    return np.all(d > -atol) and np.allclose(d, d[0], rtol=rtol, atol=atol)

def open_h5(path: str) -> h5py.File:
    """Open an HDF5 file for reading, raising FileNotFoundError if missing.

    Args:
        path: Path to the HDF5 file.

    Returns:
        h5py.File: Opened file object.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    return h5py.File(path, "r")


# -----------------------------
# Plantilla EASE2 36 km
# -----------------------------
def load_template(template_path: str) -> tuple[CRS, Affine, int, int, dict]:
    """Load a GeoTIFF template (UTM 1km recommended, EASE2 36km legacy).

    Args:
        template_path: Path to the GeoTIFF template file.

    Returns:
        Tuple containing (crs, transform, height, width, profile).
    """
    with rasterio.open(template_path) as ds:
        crs = ds.crs
        transform = ds.transform
        height, width = ds.height, ds.width
        profile = ds.profile
    return crs, transform, height, width, profile

def make_template_from_smap_l3(smap_l3_path: str, out_template: str) -> str:
    """(Legacy) Create an EASE2 36 km template GeoTIFF from a SMAP L3 file.

    Args:
        smap_l3_path: Path to SMAP L3 file (TB or SM).
        out_template: Output path for the template GeoTIFF.

    Returns:
        str: Path to the created template GeoTIFF.
    """
    # Intentamos con xarray (NetCDF/HDF que expone dims lat/lon o y/x)
    with xr.open_dataset(smap_l3_path, engine="h5netcdf", phony_dims="sort") as ds:
        # Variables típicas de rejilla: lat/lon o y/x
        # Preferimos 'EASE2' si viene como attrs.
        # Buscamos shape/grids de algún campo 2D (TB u otro).
        cand_vars = [v for v in ds.data_vars if ds[v].ndim == 2]
        v0 = None
        if cand_vars:
            v0 = ds[cand_vars[0]]
        else:
            # Fallback: abrir grupos conocidos y buscar 2D
            for grp in (
                "Soil_Moisture_Retrieval_Data_AM",
                "Soil_Moisture_Retrieval_Data_PM",
                "Radiometer_Retrieval_Data_AM",
                "Radiometer_Retrieval_Data_PM",
            ):
                try:
                    with xr.open_dataset(smap_l3_path, engine="h5netcdf", phony_dims="sort", group=grp) as dsg:
                        cands = [v for v in dsg.data_vars if dsg[v].ndim == 2]
                        if cands:
                            v0 = dsg[cands[0]]
                            break
                except Exception:
                    continue
        if v0 is None:
            # Último recurso: usar tamaño global típico EASE2 36km
            nx, ny = 964, 406
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
            return str(out_template)
        ny, nx = v0.shape

        # Espaciado y transform: algunos L3 traen x/y en km o m como coords
        # Intentamos atributos estándar
        x_name = next((c for c in v0.coords if c.lower() in ("x", "cols", "col")), None)
        y_name = next((c for c in v0.coords if c.lower() in ("y", "rows", "row")), None)

        if x_name and y_name:
            x = v0.coords[x_name].values
            y = v0.coords[y_name].values
            # Asumimos proyección EASE2 en metros si hay attrs 'grid_mapping'
            grid_mapping = None
            for v in ds.data_vars:
                gm = ds[v].attrs.get("grid_mapping")
                if gm and gm in ds.variables:
                    grid_mapping = ds[gm]
                    break

            if grid_mapping is not None:
                try:
                    crs = CRS.from_wkt(grid_mapping.attrs.get("spatial_ref"))
                except Exception:
                    # fallback a EPSG:6933 (EASE2 global equal area)
                    crs = CRS.from_epsg(6933)
            else:
                crs = CRS.from_epsg(6933)  # razonable para EASE2 global

            # Transform suponiendo píxel centrado (usamos borde superior-izq)
            if is_mono_eq(x) and is_mono_eq(y):
                dx = float(np.mean(np.diff(x)))
                dy = float(np.mean(np.diff(y)))
                x0 = float(x[0]) - dx / 2.0
                y0 = float(y[0]) - dy / 2.0
                transform = Affine(dx, 0.0, x0, 0.0, dy, y0)
            else:
                # fallback: tamaño típico EASE2 36 km global ~ 406x964 (varía por recorte)
                # asumimos paso fijo de 36 km en metros
                dx = dy = 36000.0
                transform = from_origin(-17367530.45, 7314540.83, dx, dy)  # valores estándar EASE2
        else:
            # fallback duro si no hay coords explícitas
            crs = CRS.from_epsg(6933)
            dx = dy = 36000.0
            # Extensión típica (ajustable). Esto sirve como plantilla consistente.
            transform = from_origin(-17367530.45, 7314540.83, dx, dy)
            nx = int(nx)
            ny = int(ny)

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
    # Escribimos un raster vacío solo para ser plantilla
    data = np.full((ny, nx), np.nan, dtype=np.float32)
    with rasterio.open(out_template, "w", **profile) as dst:
        dst.write(data, 1)
    return str(out_template)

# -----------------------------
# SMAP TB (desde L3 TB o L3 SM si trae TB)
# -----------------------------
def _guess_tb_var_from_l3(ds: xr.Dataset) -> str | None:
    """Attempt to find a TB variable in a SMAP L3 dataset.

    Args:
        ds: xarray.Dataset from SMAP L3 (TB or SM).

    Returns:
        str | None: Name of the TB variable if found, else None.
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
    """Load 2D TB (K) from a SMAP L3 file (TB or SM with embedded TB).

    Args:
        path_l3: Path to SMAP L3 file.
        pol: Polarization, 'H' or 'V'.
        date: Date string 'YYYY-MM-DD' for multi-date datasets (usually not needed for daily L3).
        tb_group: Group preference for TB in L3 SM: 'AM', 'PM', or 'AUTO'.

    Returns:
        np.ndarray: 2D array of TB values (float32).

    Raises:
        ValueError: If TB variable cannot be found in the file.
    """
    pol = pol.upper()
    if pol not in ("H", "V"):
        raise ValueError("pol debe ser 'H' o 'V'")

    # Intentar abrir grupos específicos de SMAP L3 SM (Soil_Moisture_Retrieval_Data_AM/PM)
    # y también el grupo raíz para SPL3TB; además probar grupos Radiometer*
    tb_group = (tb_group or "AUTO").upper()
    base_groups = [
        None,  # grupo raíz (SPL3TB)
        "Soil_Moisture_Retrieval_Data_AM",
        "Soil_Moisture_Retrieval_Data_PM",
        "Radiometer_Retrieval_Data_AM",
        "Radiometer_Retrieval_Data_PM",
        "Radiometer_Retrieval_Data",  # por si existe sin AM/PM
        "Soil_Moisture_Retrieval_Data",  # idem
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
                # Si trae dimensión time, seleccionamos por fecha si se dio.
                if "time" in ds.dims and date is not None:
                    try:
                        ds = ds.sel(time=np.datetime64(date))
                    except Exception:
                        pass

                # 1) Intento directo: variables separadas por pol
                cand_names = [
                    f"tb_{pol.lower()}_corrected",     # SPL3SMP típico
                    f"tb{pol}_corrected",
                    f"TB{pol}_corrected",
                    f"tb_{pol.lower()}",
                    f"tb{pol}", f"TB{pol}", f"TB_{pol}", f"TB_{pol.lower()}",
                    f"TB{pol}_36km", f"tb{pol}_36km",
                    f"t b{pol}".replace(" ", ""),  # tbh/tbv variantes sin guion bajo
                    f"brightness_temperature_{'vertical' if pol=='V' else 'horizontal'}",
                ]
                for nm in cand_names:
                    if nm in ds:
                        da = ds[nm]
                        arr = np.asarray(da.values, dtype=np.float32)
                        print(f"[INFO] Encontré TB en grupo '{grp}' variable '{nm}' shape {arr.shape}")
                        return arr

                # 2) Heurística: buscar una TB genérica
                tb_name = _guess_tb_var_from_l3(ds)
                if tb_name is not None:
                    da = ds[tb_name]
                    if "pol" in da.dims:
                        if "pol" in da.coords:
                            pol_vals = [str(p).upper() for p in da.coords["pol"].values]
                            if pol in pol_vals:
                                idx = pol_vals.index(pol)
                                arr = da.isel(pol=idx).values
                                return np.asarray(arr, dtype=np.float32)
                    if da.ndim == 3 and da.shape[-1] in (2, 4):
                        idx = 1 if pol == "V" else 0
                        arr = da.values[..., idx]
                        return np.asarray(arr, dtype=np.float32)
                    # Si es 2D directamente
                    if da.ndim == 2:
                        return np.asarray(da.values, dtype=np.float32)
        except Exception:
            continue
    
    # Fallback: búsqueda exhaustiva con h5py por si el nombre varía
    try:
        with h5py.File(path_l3, "r") as h5f:
            patt = re.compile(rf".*/(tb[_]?{pol.lower()}(_corrected)?|TB{pol}|TB_{pol}|brightness.*)")
            hit_path = None
            def visitor(name, obj):
                nonlocal hit_path
                if hit_path is not None:
                    return
                if isinstance(obj, h5py.Dataset):
                    if obj.ndim == 2 and patt.fullmatch("/" + name):
                        hit_path = "/" + name
            h5f.visititems(visitor)
            if hit_path:
                arr = h5f[hit_path][...].astype(np.float32)
                print(f"[INFO] Encontré TB por escaneo HDF5 en '{hit_path}' shape {arr.shape}")
                return arr
    except Exception:
        pass

    raise ValueError("No pude localizar TB en el L3 suministrado. "
                     "Verifica que el archivo sea SPL3TB* o SPL3SMP* V009 con variables tb_h_corrected/tb_v_corrected.")

def get_TBc_2d(
    pol: str,
    date: str | None,
    template_shape: tuple[int, int],
    template_transform: Affine,
    template_crs: CRS,
    path_l3: str | None = None,
    path_l1c: str | None = None,
    qa: bool = False,
    tb_group: str = "AUTO",
) -> np.ndarray | None:
    """Get TBc_2d (K) resampled to the template grid.

    Args:
        pol: Polarization, 'H' or 'V'.
        date: Date string 'YYYY-MM-DD' (optional).
        template_shape: Output shape (height, width).
        template_transform: Affine transform for output grid.
        template_crs: CRS for output grid.
        path_l3: Path to SMAP L3 file.
        path_l1c: Path to SMAP L1C file (not implemented).
        qa: If True, write debug GeoTIFF.
        tb_group: Group preference for TB in L3 SM.

    Returns:
        np.ndarray | None: 2D TB array resampled to template, or None if not available.
    """
    if not path_l3:
        return None

    TB = load_smap_l3_tb_2d(path_l3, pol=pol, date=date, tb_group=tb_group)
    if TB is None:
        return None

    # Geo-referenciación origen (usamos misma que plantilla si el L3 es EASE2 36km)
    # Intentamos leer spatial ref vía xarray otra vez para el L3 para ser precisos.
    with xr.open_dataset(path_l3, engine="h5netcdf", phony_dims="sort") as ds:
        # Si hay grid_mapping usable, tratamos de construir CRS/transform; si no,
        # asumimos que ya está en la misma rejilla EASE2 y transform = plantilla.
        src_crs = None
        src_transform = None
        for v in ds.data_vars:
            gm = ds[v].attrs.get("grid_mapping")
            if gm and gm in ds.variables:
                try:
                    src_crs = CRS.from_wkt(ds[gm].attrs.get("spatial_ref"))
                except Exception:
                    src_crs = None
                break

    if src_crs is None:
        # Fallback: asumir ya en EASE2 36 km con la misma transform que la plantilla.
        src_crs = template_crs
        src_transform = template_transform
    else:
        # Si el dataset trae coords x/y monótonas podemos construir transform:
        with xr.open_dataset(path_l3, engine="h5netcdf", phony_dims="sort") as ds2:
            vname = _guess_tb_var_from_l3(ds2) or next(v for v in ds2.data_vars if ds2[v].ndim == 2)
            da = ds2[vname]
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
            if src_transform is None:
                src_transform = template_transform  # mejor que nada

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
        _write_debug_tif(INTERIM / "debug_TB.tif", dst_arr, template_transform, template_crs)
    return dst_arr

# -----------------------------
# NISAR GCOV backscatter (σ0/γ0)
# -----------------------------
_PATT_BACK = [
    r"/science/LSAR/GCOV/.*/Sigma0.*",
    r"/science/LSAR/GCOV/.*/sigma0.*",
    r"/science/LSAR/GCOV/.*/Gamma0.*",
    r"/science/LSAR/GCOV/.*/gamma0.*",
    r"/science/LSAR/GCOV/.*/Backscatter.*",
    r"/science/LSAR/GCOV/.*/backscatter.*",
    r"/science/LSAR/GCOV/.*/VV$",
    r"/science/LSAR/GCOV/.*/HH$",
    r"/science/LSAR/GCOV/.*/VH$",
    r"/science/LSAR/GCOV/.*/HV$",
    r"/science/LSAR/GCOV/.*/(VVVV|HHHH|HVHV|VHVH)$",
]

def _score_path(path: str) -> int:
    """Score a dataset path for backscatter likelihood.

    Args:
        path: HDF5 dataset path.

    Returns:
        int: Score (higher is better match).
    """
    s = 0
    for p in _PATT_BACK:
        if re.fullmatch(p.replace(".*", ".*"), path):
            s += 5
    return s

def _walk_h5_for_backscatter(
    h5f: h5py.File,
    want_sigma: bool = True,
    pol: str = "VV",
) -> tuple[h5py.Dataset | None, str]:
    """Search HDF5 for a backscatter dataset matching sigma/gamma and polarization.

    Args:
        h5f: Open HDF5 file.
        want_sigma: If True, prefer sigma0; else gamma0.
        pol: Polarization string (e.g., 'VV', 'HH').

    Returns:
        Tuple of (dataset, path) for best match, or (None, '').
    """
    pol = pol.upper()
    goal = "sigma" if want_sigma else "gamma"

    best = (None, "", -1)
    def visit(name, obj):
        nonlocal best
        if isinstance(obj, h5py.Dataset):
            lower = name.lower()
            if "gcov" not in lower:
                return
            if goal == "sigma" and "sigma0" not in lower and "sigma" not in lower:
                return
            if goal == "gamma" and "gamma0" not in lower and "gamma" not in lower:
                return
            # pol debe aparecer en la ruta o en attrs
            if pol.lower() not in lower and not any(pol in str(v) for v in obj.attrs.values()):
                return
            sc = _score_path("/" + name)
            if sc > best[2]:
                best = (obj, name, sc)

    h5f.visititems(visit)
    return best[0], best[1]

def _read_geo_from_gcov(
    h5f: h5py.File,
    dset_path: str,
    crs_override: str | None = None,
) -> tuple[CRS, Affine]:
    """Extract CRS and transform for a GCOV dataset.

    Args:
        h5f: Open HDF5 file.
        dset_path: Path to dataset in HDF5.
        crs_override: Optional CRS override string (e.g., 'EPSG:32630').

    Returns:
        Tuple of (CRS, Affine transform).
    """
    crs = None
    transform = None

    dset = h5f[dset_path]

    # 1) grid_mapping → spatial_ref (WKT)
    gm_name = dset.attrs.get("grid_mapping")
    if isinstance(gm_name, (bytes, str)):
        gm_name = gm_name.decode() if isinstance(gm_name, bytes) else gm_name
        node = h5f.get(gm_name)
        if node is not None:
            wkt = node.attrs.get("spatial_ref")
            if isinstance(wkt, (bytes, str)):
                wkt = wkt.decode() if isinstance(wkt, bytes) else wkt
                try:
                    crs = CRS.from_wkt(wkt)
                except Exception:
                    crs = None

    # grupo padre para projection/x/y
    try:
        grp = h5f["/" + "/".join(dset_path.split("/")[:-1])]
    except Exception:
        grp = getattr(dset, "parent", None)

    # 2) dataset 'projection'
    if crs is None and grp is not None:
        proj_node = grp.get("projection", None)
        if isinstance(proj_node, h5py.Dataset):
            prj = proj_node[()]
            if isinstance(prj, (bytes, bytearray)):
                prj = prj.decode()
            prj_str = str(prj)
            try:
                crs = CRS.from_wkt(prj_str)
            except Exception:
                try:
                    pj = PJCRS.from_user_input(prj_str)   # WKT/JSON/PROJ4/EPSG
                    crs = CRS.from_wkt(pj.to_wkt())
                except Exception:
                    crs = None

    # 3) x/y → transform (north-up)
    x = grp.get("xCoordinates", None) if grp is not None else None
    if x is None and grp is not None:
        x = grp.get("x", None)
    y = grp.get("yCoordinates", None) if grp is not None else None
    if y is None and grp is not None:
        y = grp.get("y", None)

    if isinstance(x, h5py.Dataset) and isinstance(y, h5py.Dataset):
        xv = x[...]; yv = y[...]
        if xv.ndim == 1 and yv.ndim == 1 and xv.size > 1 and yv.size > 1:
            dx = float((xv[-1] - xv[0]) / (xv.size - 1))
            dy = float((yv[-1] - yv[0]) / (yv.size - 1))
            x0 = float(xv[0]) - dx/2.0
            if dy > 0:
                y0 = float(yv[-1]) + dy/2.0
                dy = -dy
            else:
                y0 = float(yv[0]) - dy/2.0
            transform = Affine(dx, 0.0, x0, 0.0, dy, y0)

    # 4) override explícito si se pasa --nisar-crs
    if crs_override:
        try:
            pj = PJCRS.from_user_input(crs_override)
            crs = CRS.from_wkt(pj.to_wkt())
        except Exception as _e:
            warnings.warn(f"No pude aplicar --nisar-crs='{crs_override}': {_e}")

    # fallbacks
    if crs is None:
        warnings.warn("No se pudo leer el CRS de GCOV; considera usar --nisar-crs.")
        crs = CRS.from_epsg(6933)
    if transform is None:
        transform = Affine.identity()

    return crs, transform    
    crs = None
    transform = None

    # Dataset objetivo
    dset = h5f[dset_path]

    # -------- 1) grid_mapping → spatial_ref (WKT)
    gm_name = dset.attrs.get("grid_mapping")
    if isinstance(gm_name, (bytes, str)):
        gm_name = gm_name.decode() if isinstance(gm_name, bytes) else gm_name
        node = h5f.get(gm_name)
        if node is not None:
            wkt = node.attrs.get("spatial_ref")
            if isinstance(wkt, (bytes, str)):
                wkt = wkt.decode() if isinstance(wkt, bytes) else wkt
                try:
                    crs = CRS.from_wkt(wkt)
                except Exception:
                    crs = None

    # Grupo contenedor (para projection/x/y)
    try:
        grp = h5f["/" + "/".join(dset_path.split("/")[:-1])]
    except Exception:
        grp = getattr(dset, "parent", None)

    # -------- 2) Dataset 'projection' (WKT / PROJJSON / PROJ4 / auth)
    if crs is None and grp is not None:
        proj_node = grp.get("projection", None)
        if isinstance(proj_node, h5py.Dataset):
            prj = proj_node[()]
            if isinstance(prj, (bytes, bytearray)):
                prj = prj.decode()
            prj_str = str(prj)

            # 2a) intentar como WKT con rasterio
            try:
                crs = CRS.from_wkt(prj_str)
            except Exception:
                crs = None

            # 2b) si no es WKT, usar pyproj (acepta WKT/JSON/PROJ4/EPSG)
            if crs is None:
                try:
                    pj = PJCRS.from_user_input(prj_str)
                    crs = CRS.from_wkt(pj.to_wkt())
                except Exception:
                    crs = None

    # -------- 3) x/y → transform (north-up)
    x = grp.get("xCoordinates", None) if grp is not None else None
    if x is None and grp is not None:
        x = grp.get("x", None)
    y = grp.get("yCoordinates", None) if grp is not None else None
    if y is None and grp is not None:
        y = grp.get("y", None)

    if isinstance(x, h5py.Dataset) and isinstance(y, h5py.Dataset):
        xv = x[...]
        yv = y[...]
        if xv.ndim == 1 and yv.ndim == 1 and xv.size > 1 and yv.size > 1:
            dx = float((xv[-1] - xv[0]) / (xv.size - 1))
            dy = float((yv[-1] - yv[0]) / (yv.size - 1))
            x0 = float(xv[0]) - dx / 2.0
            # Forzar north-up (dy negativo). Tomamos borde superior real.
            if dy > 0:
                y0 = float(yv[-1]) + dy / 2.0
                dy = -dy
            else:
                y0 = float(yv[0]) - dy / 2.0
            transform = Affine(dx, 0.0, x0, 0.0, dy, y0)

    # -------- 4) Fallbacks seguros
    if crs is None:
        warnings.warn("No se pudo leer el CRS de GCOV; considera fijarlo con --nisar-crs.")
        crs = CRS.from_epsg(6933)  # fallback neutro (EASE2 global)
    if transform is None:
        transform = Affine.identity()

    return crs, transform
    """
    Recupera CRS y transform para GCOV.
    - Intenta grid_mapping → spatial_ref (WKT)
    - Si no, usa dataset 'projection' que puede ser WKT/PROJJSON/PROJ4
    - Construye transform a partir de xCoordinates/yCoordinates en modo north-up
    """
    crs = None
    transform = None

    dset = h5f[dset_path]

    # 1) grid_mapping → spatial_ref (WKT)
    gm_name = dset.attrs.get("grid_mapping")
    if isinstance(gm_name, (bytes, str)):
        gm_name = gm_name.decode() if isinstance(gm_name, bytes) else gm_name
        node = h5f.get(gm_name)
        if node is not None:
            wkt = node.attrs.get("spatial_ref")
            if isinstance(wkt, (bytes, str)):
                wkt = wkt.decode() if isinstance(wkt, bytes) else wkt
                try:
                    crs = CRS.from_wkt(wkt)
                except Exception:
                    crs = None

    # 2) Dataset 'projection' (WKT / PROJJSON / PROJ4)
        if crs is None and isinstance(proj_node, h5py.Dataset):
            prj = proj_node[()]
        if isinstance(prj, (bytes, bytearray)):
            prj = prj.decode()
        prj_str = str(prj)

        # 1) intentar como WKT directo (rasterio)
        try:
            crs = CRS.from_wkt(prj_str)
        except Exception:
            crs = None

        # 2) si no es WKT, usar pyproj (PROJJSON / PROJ4 / auth string)
        if crs is None:
            try:
                pj = PJCRS.from_user_input(prj_str)  # acepta WKT/JSON/PROJ4/EPSG:xxxx
                crs = CRS.from_wkt(pj.to_wkt())      # convertir a rasterio.CRS vía WKT
            except Exception:
                crs = None


    # 3) x/y → transform (north-up)
    # Usamos el grupo contenedor del dataset para buscar x/y coordinates.
    if grp is None:
        try:
            grp = dset.parent
        except Exception:
            grp = None

    x = grp.get("xCoordinates", None) if grp is not None else None
    if x is None and grp is not None:
        x = grp.get("x", None)

    y = grp.get("yCoordinates", None) if grp is not None else None
    if y is None and grp is not None:
        y = grp.get("y", None)

    if isinstance(x, h5py.Dataset) and isinstance(y, h5py.Dataset):
        xv = x[...]; yv = y[...]
        if xv.ndim == 1 and yv.ndim == 1 and xv.size > 1 and yv.size > 1:
            dx = float((xv[-1] - xv[0]) / (xv.size - 1))
            dy = float((yv[-1] - yv[0]) / (yv.size - 1))
            x0 = float(xv[0]) - dx / 2.0
            # Forzar north-up (dy negativo) usando el borde superior real
            if dy > 0:
                y0 = float(yv[-1]) + dy / 2.0
                dy = -dy
            else:
                y0 = float(yv[0]) - dy / 2.0
            transform = Affine(dx, 0.0, x0, 0.0, dy, y0)

    # 4) Fallbacks seguros
    if crs is None:
        # Último recurso; evita 4326 (haría imposible reprojectar metros correctamente)
        # Si ves que tu proyección es UTM zona 30/31N, puedes fijar 32630/32631.
        warnings.warn("No se pudo leer el CRS de GCOV; considera fijarlo explícitamente.")
        crs = CRS.from_epsg(6933)  # EASE2 global como fallback neutral
    if transform is None:
        transform = Affine.identity()

    return crs, transform


def _open_dset_if_exists(h5f: h5py.File, path: str) -> h5py.Dataset | None:
    """Return HDF5 dataset if it exists, else None.

    Args:
        h5f: Open HDF5 file.
        path: Dataset path.

    Returns:
        h5py.Dataset or None.
    """
    try:
        return h5f[path]
    except Exception:
        return None

def load_nisar_gcov_backscatter(
    nisar_path: str,
    pol_key: str = "VV",
    want_sigma: bool = True,
    crs_override: str | None = None,
) -> tuple[np.ndarray | None, Affine | None, CRS | None]:
    """Load NISAR GCOV backscatter (sigma0/gamma0) for a given polarization.

    Args:
        nisar_path: Path to NISAR GCOV HDF5 file.
        pol_key: Polarization key (e.g., 'VV', 'HH').
        want_sigma: If True, prefer sigma0; else gamma0.
        crs_override: Optional CRS override string.

    Returns:
        Tuple of (arr_db, src_transform, src_crs), or (None, None, None) if not found.
    """
    pol_key = pol_key.upper()
    with open_h5(nisar_path) as h5f:
        # 1) intento normal
        dset, path = _walk_h5_for_backscatter(h5f, want_sigma=want_sigma, pol=pol_key)

        # 2) fallback compacto
        if dset is None:
            bases = ["/science/LSAR/GCOV/grids/frequencyA",
                     "/science/LSAR/GCOV/grids/frequencyB"]
            code = {"HH": "HHHH", "VV": "VVVV", "HV": "HVHV", "VH": "VHVH"}.get(pol_key)
            if code:
                for b in bases:
                    cand = f"{b}/{code}"
                    d = _open_dset_if_exists(h5f, cand)
                    if d is not None:
                        dset, path = d, cand
                        break

        if dset is None:
            return None, None, None

        # datos
        arr = dset[...].astype(np.float32)
        if np.nanmax(arr) > 50.0:
            arr_db = to_db(arr); is_linear = True
        else:
            arr_db = arr;         is_linear = False

        # γ→σ si procede
        if want_sigma:
            freq_grp = "/".join(path.split("/")[:-1])
            fac_d = _open_dset_if_exists(h5f, f"{freq_grp}/rtcGammaToSigmaFactor")
            if fac_d is not None:
                fac = fac_d[...].astype(np.float32)
                lin = arr if is_linear else np.power(10.0, arr_db / 10.0)
                arr_db = to_db(lin * fac)

        # georreferenciación (con defaults seguros)
        src_crs = CRS.from_epsg(4326)
        src_transform = Affine.identity()
        try:
            crs2, tr2 = _read_geo_from_gcov(h5f, path, crs_override=crs_override)
            if crs2 is not None:
                src_crs = crs2
            if tr2 is not None:
                src_transform = tr2
        except Exception as ee:
            warnings.warn(f"No se pudo leer CRS/transform de {path}: {ee}")

        return arr_db, src_transform, src_crs

def reproject_to_template(
    src_arr: np.ndarray,
    src_transform: Affine,
    src_crs: CRS,
    tpl_transform: Affine,
    tpl_crs: CRS,
    tpl_shape: tuple[int, int],
    resampling: ResampEnum = ResampEnum.bilinear,
) -> np.ndarray:
    """Reproject a source array to a template grid.

    Args:
        src_arr: Source array.
        src_transform: Source affine transform.
        src_crs: Source CRS.
        tpl_transform: Template affine transform.
        tpl_crs: Template CRS.
        tpl_shape: Output shape (height, width).
        resampling: Resampling method (default: bilinear).

    Returns:
        np.ndarray: Reprojected array.
    """
    dst = np.full(tpl_shape, np.nan, dtype=np.float32)
    valid_src = np.sum(np.isfinite(src_arr))
    print(f"[REPRO] src CRS: {src_crs} Shape: {src_arr.shape} Valid: {valid_src}/{src_arr.size}")
    print(f"[REPRO] dst CRS: {tpl_crs} Shape: {tpl_shape}")
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
    valid_dst = np.sum(np.isfinite(dst))
    nan_pct = 100.0 * (1.0 - valid_dst / dst.size)
    print(f"[REPRO] Result: Valid: {valid_dst}/{dst.size} NaN%: {nan_pct:.2f}")
    if nan_pct > 95.0:
        warnings.warn(
            f"Reprojection produced {nan_pct:.1f}% NaNs. "
            "Possible CRS mismatch (e.g., UTM local → EASE2 global). "
            "Consider using --nisar-crs to override source CRS or create a regional template."
        )
    return dst

def _write_debug_tif(path, arr, transform, crs):
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
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr, 1)

# -----------------------------
# Pipeline
# -----------------------------
def prepare_smap_nisar(args) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    """Prepare and align SMAP TB and NISAR GCOV to a common grid.

    Args:
        args: Parsed CLI arguments (argparse.Namespace).

    Returns:
        Tuple of (TBc_2d, S_pp_db, S_pq_db).
    """
    # 0) Template
    if not args.template or not Path(args.template).exists():
        if args.smap_l3:
            out_tpl = args.template or (PROCESSED / "EASE2_36km_template.tif")
            print(f"[INFO] Template does not exist. Creating from {args.smap_l3} → {out_tpl}")
            make_template_from_smap_l3(args.smap_l3, out_tpl)
            args.template = str(out_tpl)
        else:
            raise FileNotFoundError("No template found and --smap-l3 not provided to create one.")
    tb_crs, tb_transform, tb_h, tb_w, _ = load_template(args.template)

    # 1) SMAP TB @36 km (optional)
    TBc_2d = get_TBc_2d(
        pol=args.pol,
        date=args.date,
        template_shape=(tb_h, tb_w),
        template_transform=tb_transform,
        template_crs=tb_crs,
        path_l3=args.smap_l3,
        path_l1c=args.smap_l1c,
        qa=args.qa,
        tb_group=args.tb_group,
    )
    if TBc_2d is not None:
        TBc_2d = TBc_2d.astype("float32")
        TBc_2d[~np.isfinite(TBc_2d)] = np.nan
    #    Si --nisar-pol VV → co=VV, cross=VH ; si HH → co=HH, cross=HV
    # Forzar HH para este GCOV (solo tiene HHHH, no VV)
    co_pol = "HH"
    cross_pol = "HV"  # No existe, se rellenará con NaN
    want_sigma = bool(args.nisar_sigma0)

    print(f"[INFO] Cargando NISAR {co_pol} ({'σ0' if want_sigma else 'γ0'}) desde {args.nisar}")
    S_pp_db_src, Spp_tr, Spp_crs = load_nisar_gcov_backscatter(
        args.nisar, pol_key=co_pol, want_sigma=want_sigma, crs_override=args.nisar_crs
    )
    print(f"[INFO] Cargando NISAR {cross_pol} ({'σ0' if want_sigma else 'γ0'}) desde {args.nisar}")
    S_pq_db_src, Spq_tr, Spq_crs = None, None, None  # No hay HV en este producto


    # Reproyección co–pol (requerida)
    if S_pp_db_src is None:
        raise ValueError(
            f"No encontré banda {('σ0' if want_sigma else 'γ0')} para {co_pol} en {args.nisar}. Este GCOV solo tiene HHHH."
        )
    S_pp_db = reproject_to_template(
        S_pp_db_src, Spp_tr, Spp_crs, tb_transform, tb_crs, (tb_h, tb_w),
        resampling=ResampEnum.nearest   
    )

    # Reproyección cross–pol (opcional → NaN si no existe)
    warnings.warn(f"No encontré cross-pol {cross_pol}; se rellenará con NaN.")
    S_pq_db = np.full((tb_h, tb_w), np.nan, dtype=np.float32)

    # 3) QA
    if args.qa:
        _write_debug_tif(INTERIM / f"debug_{co_pol}.tif", S_pp_db, tb_transform, tb_crs)
        _write_debug_tif(INTERIM / f"debug_{cross_pol}.tif", S_pq_db, tb_transform, tb_crs)
        if TBc_2d is not None:
            _write_debug_tif(INTERIM / "debug_TB.tif", TBc_2d, tb_transform, tb_crs)

    # 4) Guardado
    out_npz = INTERIM / "aligned_step1.npz"
    np.savez_compressed(
        out_npz,
        TBc_2d=TBc_2d if TBc_2d is not None else np.array([]),
        S_pp_dB=S_pp_db,
        S_pq_dB=S_pq_db,
        crs_wkt=tb_crs.to_wkt(),
        transform=np.array(tb_transform, dtype=np.float64),
        height=np.int32(tb_h),
        width=np.int32(tb_w),
        meta=np.array([
            f"nisar={'sigma0' if want_sigma else 'gamma0'} {co_pol}/{cross_pol}",
            f"pol_smap={args.pol}",
            f"date={args.date}"
        ], dtype=object),
    )
    print(f"[OK] Guardado {out_npz}")
    return TBc_2d, S_pp_db, S_pq_db


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for SMAP-NISAR alignment script.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    p = argparse.ArgumentParser(description="Align SMAP TB and NISAR GCOV to EASE2 36 km grid.")
    p.add_argument("--date", type=str, required=False, help="Date YYYY-MM-DD (if applicable).")
    p.add_argument("--pol", type=str, default="V", help="SMAP TB polarization: H or V (default V).")
    p.add_argument("--smap-l3", type=str, help="Path to SMAP L3 (SPL3TB* or SPL3SMP* with embedded TB).")
    p.add_argument("--tb-group", type=str, choices=["AUTO","AM","PM"], default="AUTO",
                   help="TB group preference in L3 SM: AM, PM, or AUTO (default AUTO).")
    p.add_argument("--smap-l1c", type=str, default=None, help="(Future option) SMAP L1C.")
    p.add_argument("--nisar", type=str, required=True, help="Path to NISAR L2 PR GCOV .h5")
    p.add_argument("--template", type=str, default=str(PROCESSED / "UTM_1km_template.tif"),
                   help="GeoTIFF template for output grid (UTM 1km recommended, use make_utm_template.py). If missing and --smap-l3 is given, will create EASE2 36km legacy template.")
    p.add_argument("--nisar-pol", type=str, default="VV", help="Desired co-pol: VV or HH (default VV).")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--nisar-sigma0", action="store_true", help="Use σ0 (default).")
    grp.add_argument("--nisar-gamma0", action="store_true", help="Use γ0.")
    p.add_argument("--qa", action="store_true", help="Save debug GeoTIFFs.")
    p.add_argument("--nisar-crs", type=str, default=None,
                   help="CRS override for NISAR (e.g., 'EPSG:32630').")

    args = p.parse_args()
    if not args.nisar_sigma0 and not args.nisar_gamma0:
        args.nisar_sigma0 = True  # Default to σ0
    args.pol = args.pol.upper()
    args.nisar_pol = args.nisar_pol.upper()
    return args


def main() -> None:
    """Main entry point for SMAP-NISAR alignment script."""
    args = parse_args()
    try:
        prepare_smap_nisar(args)
    except Exception as e:
        warnings.warn(f"[ERROR] {e}")
        raise

if __name__ == "__main__":
    main()
