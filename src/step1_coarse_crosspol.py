#!/usr/bin/env python3
"""smap_xpol_to_grid.py

Read SMAP L1C HiRes cross-polarization (xpol) arrays from HDF5 and bin them
to a destination grid (an EASE2 or bbox grid). Writes per-day compressed NPZ
files with the gridded xpol in dB and a sample-count map.

This is a refactor of the previous `crosspol.py` rewritten in English and
formatted following Google Python style guidelines.

Usage:
  python smap_xpol_to_grid.py --data-dir data/raw --start 20150501 --end 20150704 \
      --out-dir data/interim

The script will look for SMAP L1C HDF5 files in `--data-dir` matching the
date and extract one of the available xpol datasets (AFT preferred). The
resulting NPZ files are created as `aligned-smap-xpol-<YYYYMMDD>.npz`.
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
from affine import Affine
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import transform as rio_transform, transform_bounds
from rasterio.transform import from_origin as rio_from_origin


# Paths inside the SMAP L1C HiRes HDF5 files where datasets are expected.
LAT_PATH = "Sigma0_Data/cell_lat"
LON_PATH = "Sigma0_Data/cell_lon"
XPOL_AFT = "Sigma0_Data/cell_sigma0_xpol_aft"
XPOL_FORE = "Sigma0_Data/cell_sigma0_xpol_fore"

# Destination CRS: EASE-Grid 2.0 Global in meters (EPSG:6933)
DST_CRS = CRS.from_epsg(6933)
# Source CRS: geographic (lat/lon)
SRC_CRS = CRS.from_epsg(4326)


def make_template_from_bbox_lonlat(
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
    pixel_size_m: float = 36000.0,
    dst_crs: CRS = DST_CRS,
) -> Tuple[Affine, int, int, CRS, dict]:
    """Create an aligned destination grid from a lon/lat bbox.

    The bbox is transformed to the destination CRS and snapped to whole
    pixel steps of ``pixel_size_m``. The returned transform follows
    rasterio's Affine convention and the height/width are integers.

    Returns:
        transform: Affine transform for the destination grid.
        height: Number of rows.
        width: Number of columns.
        dst_crs: Destination CRS object.
        meta: Small metadata dict with bbox info.
    """

    left, bottom, right, top = transform_bounds(
        SRC_CRS, dst_crs, lon_min, lat_min, lon_max, lat_max, densify_pts=21
    )

    def snap(value: float, step: float, mode: str) -> float:
        return math.floor(value / step) * step if mode == "floor" else math.ceil(value / step) * step

    left_s = snap(left, pixel_size_m, "floor")
    right_s = snap(right, pixel_size_m, "ceil")
    bottom_s = snap(bottom, pixel_size_m, "floor")
    top_s = snap(top, pixel_size_m, "ceil")

    width = int(round((right_s - left_s) / pixel_size_m))
    height = int(round((top_s - bottom_s) / pixel_size_m))

    transform = from_origin(left_s, top_s, pixel_size_m, pixel_size_m)
    meta = {
        "bbox_lonlat": [lon_min, lat_min, lon_max, lat_max],
        "bbox_dst": [left_s, bottom_s, right_s, top_s],
        "pixel_size_m": float(pixel_size_m),
        "dst_epsg": int(dst_crs.to_epsg() or 6933),
    }

    return transform, height, width, dst_crs, meta


def read_dataset(h5: h5py.File, path: str) -> np.ndarray:
    """Read a dataset from ``h5`` and return it as a NumPy array.

    Raises a KeyError when the dataset is missing.
    """

    if path not in h5:
        raise KeyError(f"Dataset not found: {path}")
    return np.array(h5[path][...], dtype=np.float64)


def apply_fill_values(h5: h5py.File, path: str, arr: np.ndarray) -> np.ndarray:
    """Apply fill values and common sentinel values to an array.

    This replaces common SMAP fill values (e.g. _FillValue, 0, -9999) with
    ``np.nan``.
    """

    ds = h5[path]
    fv = ds.attrs.get("_FillValue", None)
    if fv is not None:
        try:
            fv_val = float(np.array(fv).ravel()[0])
            arr[arr == fv_val] = np.nan
        except Exception:
            pass

    for sentinel in [0, -9999, -999, -32768, 65535]:
        arr[arr == sentinel] = np.nan

    return arr


def sigma_to_db(arr: np.ndarray) -> np.ndarray:
    """Convert sigma0 linear values to decibels when appropriate.

    The heuristic checks percentiles of the finite values. Typical linear
    sigma0 values are in the range (1e-6 .. 1) while dB values are often
    negative or small positive numbers (e.g. -50 .. +20). If the 5th
    percentile is >= 0 and the 95th percentile <= 2.0 we assume the
    array is linear and convert with 10*log10; otherwise we assume it's
    already in dB.
    """

    a = np.asarray(arr, dtype=np.float64)
    mask = np.isfinite(a)
    if mask.sum() == 0:
        return a.astype(np.float32)

    q95 = float(np.nanpercentile(a[mask], 95))
    q05 = float(np.nanpercentile(a[mask], 5))

    likely_linear = (q05 >= 0.0) and (q95 <= 2.0)
    if likely_linear:
        a = np.where(a > 0, 10.0 * np.log10(a), np.nan)
        return a.astype(np.float32)

    return a.astype(np.float32)


def swath_to_grid_mean(
    lon: np.ndarray,
    lat: np.ndarray,
    val_db: np.ndarray,
    transform: Affine,
    height: int,
    width: int,
    dst_crs: CRS = DST_CRS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bin swath points to a regular grid computing the mean per pixel.

    Returns a tuple (mean_grid, counts_grid). If no valid points fall into
    the destination grid, mean_grid is filled with ``np.nan`` and counts
    grid is zeros.
    """

    lonf = lon.reshape(-1)
    latf = lat.reshape(-1)
    vf = val_db.reshape(-1)

    m = np.isfinite(lonf) & np.isfinite(latf) & np.isfinite(vf)
    if m.sum() == 0:
        return np.full((height, width), np.nan, np.float32), np.zeros((height, width), np.int32)

    xs, ys = rio_transform(SRC_CRS, dst_crs, lonf[m].tolist(), latf[m].tolist())
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    vf = vf[m].astype(np.float64)

    a = transform.a
    e = transform.e
    c = transform.c
    f = transform.f

    cols = np.floor((xs - c) / a).astype(np.int64)
    rows = np.floor((ys - f) / e).astype(np.int64)

    ok = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    if ok.sum() == 0:
        return np.full((height, width), np.nan, np.float32), np.zeros((height, width), np.int32)

    rows = rows[ok]
    cols = cols[ok]
    vf = vf[ok]

    idx = rows * width + cols
    order = np.argsort(idx)
    idx_s = idx[order]
    vf_s = vf[order]

    uniq, start = np.unique(idx_s, return_index=True)
    sums = np.add.reduceat(vf_s, start)
    counts = np.diff(np.append(start, vf_s.size))

    out = np.full(height * width, np.nan, dtype=np.float64)
    out[uniq] = sums / counts
    out = out.reshape((height, width)).astype(np.float32)

    n = np.zeros(height * width, dtype=np.int32)
    n[uniq] = counts.astype(np.int32)
    n = n.reshape((height, width))

    return out, n


def pick_xpol_path(h5: h5py.File) -> str:
    """Select which xpol dataset to use from the HDF5 file.

    Preference is given to the AFT dataset matching earlier behavior.
    """

    if XPOL_AFT in h5:
        return XPOL_AFT
    if XPOL_FORE in h5:
        return XPOL_FORE
    raise KeyError("No xpol dataset found in Sigma0_Data (AFT/FORE)")


def find_file_for_date(data_dir: Path, ymd: str) -> Path | None:
    """Find a matching HDF5 file in ``data_dir`` for the given date string.

    The function first looks for files containing an explicit timestamp
    pattern '*_<ymd>T*.h5', else falls back to any file containing ``ymd``.
    """

    hits = sorted(data_dir.glob(f"*_{ymd}T*.h5"))
    if hits:
        return hits[0]
    hits = sorted(data_dir.glob(f"*{ymd}*.h5"))
    return hits[0] if hits else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Bin SMAP L1C xpol to a grid and save per-day NPZ files.")
    parser.add_argument("--data-dir", required=True, help="Directory containing SMAP L1C HDF5 (.h5) files")
    parser.add_argument("--start", required=True, help="Start date YYYYMMDD")
    parser.add_argument("--end", required=True, help="End date YYYYMMDD")
    parser.add_argument("--out-dir", default="data/interim", help="Output directory for NPZ files")
    parser.add_argument("--pixel-size", type=float, default=36000.0, help="Pixel size in meters for destination grid")
    parser.add_argument("--lon-min", type=float, default=-104.8884912)
    parser.add_argument("--lat-min", type=float, default=39.8008444)
    parser.add_argument("--lon-max", type=float, default=-103.7115088)
    parser.add_argument("--lat-max", type=float, default=40.6991556)

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    transform, height, width, dst_crs, meta = make_template_from_bbox_lonlat(
        args.lon_min, args.lat_min, args.lon_max, args.lat_max, pixel_size_m=args.pixel_size
    )

    start = datetime.strptime(args.start, "%Y%m%d")
    end = datetime.strptime(args.end, "%Y%m%d")

    current = start
    while current <= end:
        ymd = current.strftime("%Y%m%d")
        print(f"\n[DATE] {ymd}")

        h5_path = find_file_for_date(data_dir, ymd)
        if h5_path is None:
            print("  [SKIP] no HDF5 for that date")
            current += timedelta(days=1)
            continue

        print("  [FILE]", h5_path.name)

        try:
            with h5py.File(h5_path, "r") as h5:
                if LAT_PATH not in h5 or LON_PATH not in h5:
                    raise KeyError("Latitude/longitude arrays missing in HDF5")

                lat = read_dataset(h5, LAT_PATH)
                lon = read_dataset(h5, LON_PATH)

                xpol_path = pick_xpol_path(h5)
                xpol = read_dataset(h5, xpol_path)

                lat = apply_fill_values(h5, LAT_PATH, lat)
                lon = apply_fill_values(h5, LON_PATH, lon)
                xpol = apply_fill_values(h5, xpol_path, xpol)

            if lat.shape != lon.shape or xpol.shape != lat.shape:
                raise RuntimeError(f"Shape mismatch: lat{lat.shape} lon{lon.shape} xpol{xpol.shape}")

            xpol_db = sigma_to_db(xpol)

                # --- Save native-resolution NPZ for this date (leave original 36km NPZ behavior) ---
                # Save a cropped native NPZ restricted to the user's lon/lat bbox when possible.
            try:
                nat_out_path = out_dir / f"aligned-smap-xpol-{ymd}-native.npz"

                # Base payload
                native_save = dict(
                    S_xpol_dB_native=xpol_db.astype(np.float32),
                    lat_native=lat.astype(np.float64),
                    lon_native=lon.astype(np.float64),
                    source_h5=str(h5_path),
                    xpol_path=xpol_path,
                    crs_wkt=SRC_CRS.to_wkt(),
                    meta=meta,  # include user's bbox and template info so native NPZ knows the bbox
                )

                # Compute mask of points inside the user's lon/lat bbox (inclusive)
                lon_min_u, lat_min_u, lon_max_u, lat_max_u = (
                    float(args.lon_min),
                    float(args.lat_min),
                    float(args.lon_max),
                    float(args.lat_max),
                )

                try:
                    inside = (
                        (lon >= lon_min_u) & (lon <= lon_max_u) & (lat >= lat_min_u) & (lat <= lat_max_u)
                    )
                    if np.any(inside):
                        rows, cols = np.where(inside)
                        r0, r1 = int(rows.min()), int(rows.max())
                        c0, c1 = int(cols.min()), int(cols.max())

                        # Attempt to slice contiguous subarray covering the bbox footprint
                        s_xpol = xpol_db[r0 : r1 + 1, c0 : c1 + 1].astype(np.float32)
                        s_lat = lat[r0 : r1 + 1, c0 : c1 + 1].astype(np.float64)
                        s_lon = lon[r0 : r1 + 1, c0 : c1 + 1].astype(np.float64)

                        native_save["S_xpol_dB_native"] = s_xpol
                        native_save["lat_native"] = s_lat
                        native_save["lon_native"] = s_lon

                        # If a regular-grid transform can be inferred, compute and adjust it
                        try:
                            lon_row = s_lon[0, :]
                            lat_col = s_lat[:, 0]
                            dxs = np.diff(lon_row)
                            dys = np.diff(lat_col)
                            if lon_row.size >= 2 and lat_col.size >= 2 and np.allclose(dxs, dxs[0], rtol=1e-6, atol=1e-6) and np.allclose(dys, dys[0], rtol=1e-6, atol=1e-6):
                                dx = float(np.mean(dxs))
                                dy = float(np.mean(dys))
                                left = float(np.min(lon_row)) - dx / 2.0
                                top = float(np.max(lat_col)) + dy / 2.0
                                trans = rio_from_origin(left, top, abs(dx), abs(dy))
                                # If we had previously stored a full-native transform, we need to offset it
                                native_save["transform_native"] = np.array(
                                    [trans.a, trans.b, trans.c, trans.d, trans.e, trans.f], dtype=np.float64
                                )
                                native_save["height_native"] = np.int32(s_xpol.shape[0])
                                native_save["width_native"] = np.int32(s_xpol.shape[1])
                        except Exception:
                            pass

                        native_save["crop_index"] = np.array([r0, r1, c0, c1], dtype=np.int32)
                        native_save["cropped_to_bbox"] = True
                        np.savez_compressed(nat_out_path, **native_save)
                        print(
                            f"  [OK] native NPZ saved (cropped): {nat_out_path.name} | shape: {s_xpol.shape} | finite: {int(np.isfinite(s_xpol).sum())}"
                        )
                    else:
                        # no points inside bbox: save full native arrays but note the condition
                        native_save["cropped_to_bbox"] = False
                        np.savez_compressed(nat_out_path, **native_save)
                        print(
                            f"  [OK] native NPZ saved (no points inside bbox): {nat_out_path.name} | shape: {xpol_db.shape} | finite: {int(np.isfinite(xpol_db).sum())}"
                        )
                except Exception:
                    # If anything goes wrong in cropping, fallback to saving the full native arrays
                    native_save["cropped_to_bbox"] = False
                    np.savez_compressed(nat_out_path, **native_save)
                    print(f"  [OK] native NPZ saved (fallback): {nat_out_path.name} | shape: {xpol_db.shape} | finite: {int(np.isfinite(xpol_db).sum())}")
            except Exception as _e:
                print("  [WARN] could not save native NPZ:", _e)

            S_xpol_dB, n_samples = swath_to_grid_mean(
                lon=lon, lat=lat, val_db=xpol_db, transform=transform, height=height, width=width, dst_crs=dst_crs
            )

            out_path = out_dir / f"aligned-smap-xpol-{ymd}.npz"
            np.savez_compressed(
                out_path,
                S_xpol_dB=S_xpol_dB,
                n_samples=n_samples,
                crs_wkt=dst_crs.to_wkt(),
                transform=np.array([transform.a, transform.b, transform.c, transform.d, transform.e, transform.f], dtype=np.float64),
                height=np.int32(height),
                width=np.int32(width),
                meta=meta,
                source_h5=str(h5_path),
                xpol_path=xpol_path,
            )

            print(f"  [OK] {out_path.name} | valid pixels: {int(np.isfinite(S_xpol_dB).sum())} | n_samples max: {int(n_samples.max())}")

        except Exception as exc:  # pylint: disable=broad-except
            print("  [ERROR]", exc)

        current += timedelta(days=1)


if __name__ == "__main__":
    main()
