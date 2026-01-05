#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reprojects SMAP ATBD parameter grids to the native fine grid and saves NPZ files.

For each date, this script:
  1) Loads ancillary (coarse) ATBD parameters on the 36 km grid.
  2) Loads the target (native) fine-grid geometry from an aligned file.
  3) Reprojects each ancillary parameter to the native grid.
  4) Writes one NPZ per parameter to the processed directory.
  5) Calls step4_final.py to generate soil moisture for that date.

Notes:
- This script overwrites the parameter NPZ outputs for each date (same filenames).
  If you want to keep per-date parameter files, include the date in the output names.
- Reprojection uses bilinear resampling and NaN as nodata.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Tuple

import numpy as np
from affine import Affine
from rasterio.crs import CRS
from rasterio.warp import Resampling, reproject


DATES = ["20150610", "20150615", "20150618", "20150620"]

INTERIM_DIR = Path(r"c:\Users\ortuo\tfgirati\tfg-nisar\data\interim")
PROCESSED_DIR = Path(r"c:\Users\ortuo\tfgirati\tfg-nisar\data\processed")

PYTHON_EXE = Path(r"C:/Users/ortuo/tfgirati/.venv/Scripts/python.exe")
STEP4_SCRIPT = Path(r"C:\Users\ortuo\tfgirati\tfg-nisar\src\step4_final.py")


def _affine_from_npz(transform_arr: np.ndarray) -> Affine:
    """Builds an Affine transform from a 6-element array (GDAL-style)."""
    return Affine(
        float(transform_arr[0]),
        float(transform_arr[1]),
        float(transform_arr[2]),
        float(transform_arr[3]),
        float(transform_arr[4]),
        float(transform_arr[5]),
    )


def reproject_to_target(
    src: np.ndarray,
    src_transform: Affine,
    src_crs: CRS,
    dst_transform: Affine,
    dst_crs: CRS,
    dst_shape: Tuple[int, int],
    *,
    resampling: Resampling = Resampling.bilinear,
) -> np.ndarray:
    """Reprojects a 2D array to a target grid.

    Args:
        src: Source 2D array.
        src_transform: Source affine transform.
        src_crs: Source CRS.
        dst_transform: Destination affine transform.
        dst_crs: Destination CRS.
        dst_shape: (height, width) of destination.
        resampling: Rasterio resampling method.

    Returns:
        Reprojected 2D array (float32) with NaNs as nodata.
    """
    dst = np.full(dst_shape, np.nan, dtype=np.float32)
    reproject(
        source=src,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=resampling,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    return dst


def _save_npz(
    out_path: Path,
    data_key: str,
    data: np.ndarray,
    transform: Affine,
    crs: CRS,
    shape: Tuple[int, int],
    date: str,
) -> None:
    """Saves a single-parameter NPZ with common georeferencing metadata."""
    transform_vec = np.array(
        [transform.a, transform.b, transform.c, transform.d, transform.e, transform.f],
        dtype=np.float64,
    )
    np.savez_compressed(
        out_path,
        **{
            data_key: data.astype(np.float32),
            "transform": transform_vec,
            "crs_wkt": crs.to_wkt(),
            "height": int(shape[0]),
            "width": int(shape[1]),
            "date": date,
        },
    )


def _run_step4_final(date: str) -> None:
    """Runs step4_final.py for a given date."""
    cmd = [
        str(PYTHON_EXE),
        str(STEP4_SCRIPT),
        "--tb",
        str(PROCESSED_DIR / f"TB_fine_{date}_VV_native.npz"),
        "--tb-pol",
        "V",
        "--Teff-npz",
        str(PROCESSED_DIR / "L3_teff_native.npz"),
        "--tau-npz",
        str(PROCESSED_DIR / "tau_native.npz"),
        "--omega-npz",
        str(PROCESSED_DIR / "omega_native.npz"),
        "--theta-npz",
        str(PROCESSED_DIR / "theta_native.npz"),
        "--h-npz",
        str(PROCESSED_DIR / "h_native.npz"),
        "--sm-min",
        "0.05",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode == 0:
        print(f"  ✓ Soil moisture generated for {date}")
        return

    stderr = result.stderr.strip() or "(no stderr captured)"
    print(f"  ✗ step4_final failed for {date}:\n{stderr}")


def main() -> None:
    print("\n" + "=" * 70)
    print(f"Generating ATBD parameter NPZ files for {len(DATES)} dates")
    print("=" * 70 + "\n")

    for date in DATES:
        print("\n" + "=" * 70)
        print(f"Processing date: {date}")
        print("=" * 70 + "\n")

        # ---------------------------------------------------------------------
        # 1) Load coarse ancillary data (36 km).
        # ---------------------------------------------------------------------
        print("[1/5] Loading ancillary data (coarse grid)...")
        anc_path = INTERIM_DIR / f"smap-ancillary-{date}.npz"
        if not anc_path.exists():
            print(f"  [WARN] Missing {anc_path.name}; skipping date.")
            continue

        anc = np.load(anc_path)

        teff_coarse = anc["surface_temperature"]  # Kelvin
        tau_coarse = anc["vegetation_opacity"]  # unitless
        omega_coarse = anc["albedo"]  # unitless (single scattering albedo)
        theta_coarse = anc["boresight_incidence"]  # degrees
        h_coarse = anc["roughness_coefficient"]  # roughness parameter

        anc_transform = _affine_from_npz(anc["transform"])
        anc_crs = CRS.from_wkt(str(anc["crs_wkt"]))

        print(f"  Teff:  {teff_coarse.shape}, mean={np.nanmean(teff_coarse):.2f} K")
        print(f"  tau:   {tau_coarse.shape}, mean={np.nanmean(tau_coarse):.3f}")
        print(f"  omega: {omega_coarse.shape}, mean={np.nanmean(omega_coarse):.3f}")
        print(f"  theta: {theta_coarse.shape}, mean={np.nanmean(theta_coarse):.2f}°")
        print(f"  h:     {h_coarse.shape}, mean={np.nanmean(h_coarse):.3f}")

        # ---------------------------------------------------------------------
        # 2) Load native target grid geometry from aligned file.
        # ---------------------------------------------------------------------
        print("\n[2/5] Loading native target grid...")
        native_path = INTERIM_DIR / f"aligned-smap-copol-{date}-vv-native.npz"
        if not native_path.exists():
            print(f"  [WARN] Missing {native_path.name}; skipping date.")
            continue

        native = np.load(native_path)
        native_transform = _affine_from_npz(native["transform_native"])
        native_crs = CRS.from_wkt(str(native["crs_wkt"]))
        native_shape = (int(native["height_native"]), int(native["width_native"]))

        print(f"  Native grid shape: {native_shape}")

        # ---------------------------------------------------------------------
        # 3) Reproject each parameter to the native grid.
        # ---------------------------------------------------------------------
        print("\n[3/5] Reprojecting parameters to native grid...")

        print("  Reprojecting Teff...")
        teff_native = reproject_to_target(
            teff_coarse,
            anc_transform,
            anc_crs,
            native_transform,
            native_crs,
            native_shape,
        )

        print("  Reprojecting tau...")
        tau_native = reproject_to_target(
            tau_coarse,
            anc_transform,
            anc_crs,
            native_transform,
            native_crs,
            native_shape,
        )

        print("  Reprojecting omega...")
        omega_native = reproject_to_target(
            omega_coarse,
            anc_transform,
            anc_crs,
            native_transform,
            native_crs,
            native_shape,
        )

        print("  Reprojecting theta...")
        theta_native = reproject_to_target(
            theta_coarse,
            anc_transform,
            anc_crs,
            native_transform,
            native_crs,
            native_shape,
        )

        print("  Reprojecting h...")
        h_native = reproject_to_target(
            h_coarse,
            anc_transform,
            anc_crs,
            native_transform,
            native_crs,
            native_shape,
        )

        # ---------------------------------------------------------------------
        # 4) Save NPZ outputs (overwrites each date).
        # ---------------------------------------------------------------------
        print(f"\n[4/5] Writing parameter NPZ files for {date}...")

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        out_teff = PROCESSED_DIR / "L3_teff_native.npz"
        _save_npz(
            out_teff,
            data_key="Teff_K",
            data=teff_native,
            transform=native_transform,
            crs=native_crs,
            shape=native_shape,
            date=date,
        )
        print(f"  ✓ {out_teff.name}")

        out_tau = PROCESSED_DIR / "tau_native.npz"
        _save_npz(
            out_tau,
            data_key="tau",
            data=tau_native,
            transform=native_transform,
            crs=native_crs,
            shape=native_shape,
            date=date,
        )
        print(f"  ✓ {out_tau.name}")

        out_omega = PROCESSED_DIR / "omega_native.npz"
        _save_npz(
            out_omega,
            data_key="omega",
            data=omega_native,
            transform=native_transform,
            crs=native_crs,
            shape=native_shape,
            date=date,
        )
        print(f"  ✓ {out_omega.name}")

        out_theta = PROCESSED_DIR / "theta_native.npz"
        _save_npz(
            out_theta,
            data_key="theta_deg",
            data=theta_native,
            transform=native_transform,
            crs=native_crs,
            shape=native_shape,
            date=date,
        )
        print(f"  ✓ {out_theta.name}")

        out_h = PROCESSED_DIR / "h_native.npz"
        _save_npz(
            out_h,
            data_key="h",
            data=h_native,
            transform=native_transform,
            crs=native_crs,
            shape=native_shape,
            date=date,
        )
        print(f"  ✓ {out_h.name}")

        # ---------------------------------------------------------------------
        # 5) Run step4_final for this date.
        # ---------------------------------------------------------------------
        print(f"\n[5/5] Running step4_final for {date}...")
        _run_step4_final(date)

        print("\n" + "=" * 70)
        print(f"✓ Completed: {date}")
        print("=" * 70)

    print("\n" + "=" * 70)
    print("✓ ALL DONE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
