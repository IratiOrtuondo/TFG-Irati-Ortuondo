r"""SMAP Granule Downloader (earthaccess CLI helper)
===============================================

This script provides a flexible command-line interface to search and
optionally download SMAP granules using the `earthaccess` package. It
supports different temporal selections (single date or date range),
spatial selections (bounding box, cell-center, or a fixed Boulder tile),
and can optionally download radar products alongside radiometer files.

Usage examples:
  # Download SPL3SMP (L3 radiometer, 36 km) for a single date (global)
  python src/smap.py --date 2015-06-15

  # Download SPL3SMP and SPL3SMA (radar) for the same date
  python src/smap.py --date 2015-06-15 --also-radar

  # Download a single ~36 km cell centered at (-10, 40)
  python src/smap.py --date 2015-06-15 --cell-center -10,40 --also-radar

  # Provide an explicit bounding box
  python src/smap.py --date 2015-06-15 --bbox -10,35,10,45 --also-radar

  # Use a built-in Boulder 36km helper bbox
  python src\smap.py --date 2015-06-15 --boulder-36km --also-radar

Credentials:
  You can set EARTHDATA_USERNAME / EARTHDATA_PASSWORD environment variables
  or use a ~/.netrc file with your Earthdata login.

  Example .netrc (Windows: %USERPROFILE%\.netrc):
  machine urs.earthdata.nasa.gov
    login YOUR_USERNAME
    password YOUR_PASSWORD
"""

from __future__ import annotations

import os
import sys
import argparse
from datetime import datetime, timedelta
from typing import Tuple, Optional, Sequence

import math
import earthaccess as ea


# -----------------------------------------------------------------------------
# Login
# -----------------------------------------------------------------------------

def try_login() -> None:
    """
    Attempt authentication with Earthdata.

    Tries multiple strategies to obtain an authenticated earthaccess
    session. First attempts environment variables (EARTHDATA_USERNAME /
    EARTHDATA_PASSWORD) and falls back to ~/.netrc if environment
    variables are not set. If no authentication is available, the script
    continues, but searches or downloads may fail or return no results.
    """
    strategies = ["environment", "netrc"]
    for strat in strategies:
        try:
            # ea.login will try the given strategy and return a session object
            sess = ea.login(strategy=strat)
            if sess:
                print(f"[INFO] Authenticated using strategy '{strat}'.")
                return
        except Exception as e:  # pragma: no cover - defensive
            # Keep a clear English warning message for users
            print(f"[WARN] Login failed using '{strat}': {e}")
    # If we reach here, no authentication succeeded
    print("[WARN] Not authenticated. Public searches may return no results or downloads may be blocked.")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Provides flexible temporal and spatial selection options and supports
    downloading radar products in addition to the primary radiometer
    collection. Returns the populated argparse Namespace.
    """
    p = argparse.ArgumentParser(description="Search and download SMAP granules using earthaccess")

    # Temporal selection: either a single date or a start date (with --end)
    gdate = p.add_mutually_exclusive_group(required=True)
    gdate.add_argument("--date", help="Single date (YYYY-MM-DD)")
    gdate.add_argument("--start", help="Start date (YYYY-MM-DD) for a range")
    p.add_argument("--end", help="Exclusive end date (YYYY-MM-DD) when using --start")

    # Primary collection selection (default: SPL3SMP - L3 Radiometer 36 km)
    p.add_argument(
        "--short-name",
        default="SPL3SMP",
        help=(
            "Primary collection short_name (default: SPL3SMP - L3 Radiometer Global Daily 36 km Soil Moisture)"
        ),
    )
    p.add_argument("--version", help="Optional collection version (e.g., 009) to narrow search")
    p.add_argument("--concept-id", help="Optional CMR concept-id (overrides short_name/version if provided)")

    # Spatial selection options
    p.add_argument(
        "--bbox",
        nargs="+",
        help=(
            "Bounding box given as minLon,minLat,maxLon,maxLat. Accepts a single comma-separated string "
            "('-104.8,39.8,-103.7,40.7') or four space-separated values. Ignored if --cell-center or --boulder-36km is used."
        ),
    )
    p.add_argument(
        "--boulder-36km",
        action="store_true",
        help="Use a fixed approximate 36 km Boulder bbox.",
    )
    p.add_argument(
        "--cell-center",
        help="Small cell center specified as 'lon,lat'. If provided, it overrides --bbox and --boulder-36km.",
    )
    p.add_argument(
        "--cell-size-km",
        type=float,
        default=36.0,
        help="Approximate side length of the cell in km (default: 36). Only used with --cell-center.",
    )

    # Runtime options
    p.add_argument("--threads", type=int, default=4, help="Threads for parallel downloads")
    p.add_argument("--dry-run", action="store_true", help="Only search and list matches; do not download")
    p.add_argument(
        "--list-datasets",
        help="List datasets by keyword and exit (ignores short_name/granule selection)",
    )

    # Optional radar collection selection when --also-radar is provided
    p.add_argument(
        "--also-radar",
        action="store_true",
        help=(
            "Also download a radar product (backscatter / σ0) for the same period and bbox. "
            "Default radar collection: SPL3SMA (L3 Radar Soil Moisture 3 km)."
        ),
    )
    p.add_argument(
        "--radar-short-name",
        default="SPL3SMA",
        help="Radar collection short_name to use when --also-radar (default: SPL3SMA).",
    )
    p.add_argument(
        "--radar-version",
        help="Optional radar collection version (e.g., 003). Note: SPL3SMA only exists approx 2015-04-13 to 2015-07-07.",
    )
    p.add_argument(
        "--radar-concept-id",
        help="Optional radar CMR concept-id (overrides radar-short-name/version if provided).",
    )

    return p.parse_args()


# -----------------------------------------------------------------------------
# Tiempo y bbox
# -----------------------------------------------------------------------------

def build_temporal(args: argparse.Namespace) -> Tuple[str, str]:
    """
    Build an exclusive temporal window from CLI args.

    Returns a tuple (start_date, end_date) in 'YYYY-MM-DD' format where
    end_date is exclusive. Accepts either a single --date (returns that day
    to the following day) or a --start/--end range.
    """
    fmt = "%Y-%m-%d"
    if args.date:
        d = datetime.strptime(args.date, fmt)
        # single date -> return [date, date+1 day)
        return d.strftime(fmt), (d + timedelta(days=1)).strftime(fmt)
    if not args.end:
        raise SystemExit("--end is required when using --start")
    start_dt = datetime.strptime(args.start, fmt)
    end_dt = datetime.strptime(args.end, fmt)
    if end_dt <= start_dt:
        raise SystemExit("--end must be later than --start")
    return start_dt.strftime(fmt), end_dt.strftime(fmt)


def parse_bbox(bbox_arg):
    """
    Parse the --bbox argument into a numeric (minLon, minLat, maxLon, maxLat) tuple.

    Accepts three styles of input:
    - None -> returns None (no bounding box provided)
    - Single comma-separated string: '-104.8,39.8,-103.7,40.7'
    - Four separate tokens: -104.8 39.8 -103.7 40.7

    Returns:
        Tuple of floats (minLon, minLat, maxLon, maxLat) or None.
    """
    if not bbox_arg:
        return None

    # If argparse used nargs, bbox_arg may be a list/tuple
    if isinstance(bbox_arg, (list, tuple)):
        if len(bbox_arg) == 1:
            # Single comma-separated string inside a list
            s = bbox_arg[0]
            parts = s.split(",")
        else:
            # Expect exactly four space-separated tokens
            parts = list(bbox_arg)
    else:
        # Single string provided directly
        parts = str(bbox_arg).split(",")

    if len(parts) != 4:
        raise SystemExit("--bbox invalid format. Use minLon,minLat,maxLon,maxLat or 4 space-separated values")
    try:
        return tuple(float(p) for p in parts)
    except ValueError:
        raise SystemExit("--bbox values must be numeric")


def boulder_bbox_36km() -> Tuple[float, float, float, float]:
    """
    Return a hard-coded approximate 36 km SMAP pixel bounding box centered near Boulder, CO.

    This helper is convenient for quick tests and is not intended as a precise
    scientific footprint; use explicit lat/lon bounding boxes for production.
    """
    min_lon = -105.6
    max_lon = -104.9
    min_lat = 39.8
    max_lat = 40.2
    return (min_lon, min_lat, max_lon, max_lat)


def bbox_from_cell_center(center_str: str, size_km: float) -> Tuple[float, float, float, float]:
    """
    Construct an approximate square bounding box (minLon,minLat,maxLon,maxLat)
    from a center point string 'lon,lat' and an approximate side length in km.

    This function uses a simple equirectangular approximation where 1 degree
    of latitude ≈ 111.32 km and 1 degree of longitude ≈ 111.32 * cos(lat).
    The approximation is suitable for small (~36 km) boxes away from the poles.
    """
    try:
        lon_str, lat_str = center_str.split(",")
        lon0 = float(lon_str)
        lat0 = float(lat_str)
    except Exception:
        raise SystemExit("--cell-center must be 'lon,lat' with numeric values")

    half_km = size_km / 2.0
    # 1° latitude approximately 111.32 km; longitude scaled by cos(latitude)
    dlat = half_km / 111.32
    cos_lat = math.cos(math.radians(lat0))
    if abs(cos_lat) < 1e-6:
        # Near poles, avoid division by zero; fallback to crude approx
        dlon = half_km / 111.32
    else:
        dlon = half_km / (111.32 * cos_lat)

    min_lon = lon0 - dlon
    max_lon = lon0 + dlon
    min_lat = lat0 - dlat
    max_lat = lat0 + dlat
    return (min_lon, min_lat, max_lon, max_lat)


# -----------------------------------------------------------------------------
# Utilidades varias
# -----------------------------------------------------------------------------

def safe_short_name(granule) -> str:
    """
    Return a best-effort short name or title for a collection/granule object.

    The EarthAccess search results can have different attribute naming
    conventions. This helper attempts a small list of possible attributes
    and also inspects a 'properties' dictionary fallback.
    """
    for attr in ("short_name", "shortName", "name", "title"):
        if hasattr(granule, attr):
            return getattr(granule, attr)
    props = getattr(granule, "properties", None)
    if isinstance(props, dict):
        for key in ("short_name", "shortName", "shortname", "title", "Name"):
            if key in props:
                return props[key]
    return "<unknown>"


def search_and_download_one_collection(
    label: str,
    temporal: Tuple[str, str],
    bbox: Optional[Tuple[float, float, float, float]],
    short_name: Optional[str] = None,
    version: Optional[str] = None,
    concept_id: Optional[str] = None,
    threads: int = 4,
    dry_run: bool = False,
) -> Sequence[str]:
    """
    Search a SMAP collection via Earthdata and optionally download matches.

    This helper wraps `ea.search_data(...)` to find granules for the given
    temporal window and spatial region. If `dry_run` is False, it will
    download the discovered granules in parallel using `ea.download`.

    Returns a list of saved file paths (empty list when no files or in dry-run).
    """
    print()
    print(f"[INFO] === Collection: {label} ===")

    # Build search kwargs using either a concept-id or short_name (+ optional version)
    if concept_id:
        search_kwargs = dict(concept_id=concept_id, temporal=temporal)
        desc = f"concept_id='{concept_id}'"
    else:
        if not short_name:
            raise SystemExit("short_name is required if concept_id is not provided.")
        search_kwargs = dict(short_name=short_name, temporal=temporal)
        desc = f"short_name='{short_name}'"
        if version:
            search_kwargs["version"] = version
            desc += f" version={version}"

    if bbox:
        # Provide bounding_box to the CMR search if available
        search_kwargs["bounding_box"] = bbox
        bbox_desc = f"bbox={bbox}"
    else:
        bbox_desc = "<global>"

    print(f"[INFO] Searching {desc} temporal={temporal} {bbox_desc}")
    print(f"[DEBUG] search_kwargs={search_kwargs}")

    try:
        files = ea.search_data(**search_kwargs)
    except Exception as e:  # pragma: no cover
        print(f"[ERROR] search_data failed: {e}")
        sys.exit(1)

    print("[INFO] Found granules:", len(files))
    if not files:
        print("[WARN] No results for this collection.")
        return []

    # Print a short preview for up to 5 granules to give the user context
    for i, f in enumerate(files[:5]):
        try:
            size_mb = getattr(f, "size", None)
            if size_mb is not None:
                try:
                    size_mb = float(size_mb) / (1024 * 1024)
                except Exception:
                    size_mb = None
            if size_mb:
                print(
                    f"  [{i}] {safe_short_name(f)} | "
                    f"id={getattr(f,'granule_id', '<no-id>')} | sizeMB={size_mb:.2f}"
                )
            else:
                print(f"  [{i}] {safe_short_name(f)}")
        except Exception as e:
            print(f"  [{i}] <error reading granule>: {e}")

    # If user asked for dry-run, do not download; just return empty list
    if dry_run:
        print("[INFO] Dry-run: files will not be downloaded for this collection.")
        return []

    # Otherwise, download to './data/raw' with requested thread count
    out_dir = os.path.join(".", "data", "raw")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Downloading to '{out_dir}' with threads={threads}...")
    try:
        saved = ea.download(files, out_dir, threads=threads)
    except Exception as e:  # pragma: no cover
        print(f"[ERROR] Download failed: {e}")
        sys.exit(3)
    print("[INFO] Download complete.")
    for p in saved:
        print("  -", p)
    return saved


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main():
    """
    High-level CLI workflow:
      1. Parse CLI arguments
      2. Attempt authentication
      3. Optionally list datasets by keyword and exit
      4. Build temporal window and spatial bounding box
      5. Search and optionally download the primary collection
      6. Optionally search/download a radar collection
    """
    args = parse_args()
    try_login()

    # If user asked to list datasets by keyword, print and exit
    if args.list_datasets:
        print(f"[INFO] Listing datasets with keyword='{args.list_datasets}'...")
        try:
            ds = ea.search_datasets(keyword=args.list_datasets)
            print(f"[INFO] Found {len(ds)} datasets")
            for i, d in enumerate(ds):
                title = getattr(d, "title", "<no-title>")
                meta = getattr(d, "meta", {}) or {}
                cid = meta.get("concept-id", meta.get("concept_id", "<no-concept-id>"))
                sn = getattr(d, "short_name", getattr(d, "shortName", "<no-short-name>"))
                ver = getattr(d, "version_id", getattr(d, "version", "?"))
                print(f"  [{i}] short_name={sn} version={ver} concept_id={cid} | {title}")
        except Exception as e:
            print(f"[ERROR] Could not list datasets: {e}")
        return

    # Build the temporal window for searches
    temporal = build_temporal(args)

    # Spatial selection priority: explicit cell center -> Boulder helper -> explicit bbox -> global
    if args.cell_center:
        bbox = bbox_from_cell_center(args.cell_center, args.cell_size_km)
        print(f"[INFO] Using bbox derived from cell-center={args.cell_center}, cell-size-km={args.cell_size_km}: {bbox}")
    elif args.boulder_36km:
        bbox = boulder_bbox_36km()
        print(f"[INFO] Using Boulder bbox (~36 km): {bbox}")
    else:
        bbox = parse_bbox(args.bbox)
        print(f"[INFO] bbox={bbox or '<global>'}")

    # Search (and optionally download) primary collection
    saved_main = search_and_download_one_collection(
        label="primary (e.g. SPL3SMP)",
        temporal=temporal,
        bbox=bbox,
        short_name=args.short_name,
        version=args.version,
        concept_id=args.concept_id,
        threads=args.threads,
        dry_run=args.dry_run,
    )

    # Optionally handle radar collection
    if args.also_radar:
        saved_rad = search_and_download_one_collection(
            label="radar (e.g. SPL3SMA)",
            temporal=temporal,
            bbox=bbox,
            short_name=args.radar_short_name,
            version=args.radar_version,
            concept_id=args.radar_concept_id,
            threads=args.threads,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            print("[INFO] Dry-run: radar NOT downloaded (only listed).")
        else:
            print(f"[INFO] Radar downloaded ({len(saved_rad)} files).")
            if len(saved_rad) == 0:
                print("[TIP] Remember radar products (SPL3SMA/SPL3SMAP) only exist approx. 2015-04-13 to 2015-07-07.")
    else:
        print("[INFO] --also-radar not set: no additional radar product will be downloaded.")

    print("[INFO] End of smap.py script.")


if __name__ == "__main__":
    main()
