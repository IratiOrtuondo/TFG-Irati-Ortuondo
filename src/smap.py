"""Descarga granulos SMAP vía earthaccess con CLI flexible.

Ejemplos:
  # Descargar SPL3SMP (radiometro L3 36 km) de un día (global)
  python src\smap.py --date 2015-06-15

  # Descargar SPL3SMP + SPL3SMA (radar) misma fecha, global
  python src\smap.py --date 2015-06-15 --also-radar

  # Descargar sólo en una celda de ~36 km centrada en (lon,lat)=(-10,40)
  python src\smap.py --date 2015-06-15 --cell-center -10,40 --also-radar

  # Usar bbox genérico
  python src\smap.py --date 2015-06-15 --bbox -10,35,10,45 --also-radar

  # Usar helper Boulder (~36 km alrededor)
  python src\smap.py --date 2015-06-15 --boulder-36km --also-radar

Variables de entorno si no tienes ~/.netrc:
  EARTHDATA_USERNAME, EARTHDATA_PASSWORD

Archivo ~/.netrc (Windows: %USERPROFILE%\.netrc):
machine urs.earthdata.nasa.gov
  login TU_USUARIO
  password TU_PASSWORD
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
    """Intenta login con variables de entorno y si falla, con netrc."""
    strategies = ["environment", "netrc"]
    for strat in strategies:
        try:
            sess = ea.login(strategy=strat)
            if sess:
                print(f"[INFO] Autenticado con estrategia '{strat}'.")
                return
        except Exception as e:  # pragma: no cover - defensivo
            print(f"[WARN] Falló login '{strat}': {e}")
    print("[WARN] No autenticado. La búsqueda pública puede devolver 0 resultados o impedir la descarga.")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Buscar y descargar granulos SMAP con earthaccess")

    # Fecha única o rango
    gdate = p.add_mutually_exclusive_group(required=True)
    gdate.add_argument("--date", help="Fecha única (YYYY-MM-DD)")
    gdate.add_argument("--start", help="Fecha inicio (YYYY-MM-DD) para rango")
    p.add_argument("--end", help="Fecha fin (YYYY-MM-DD) exclusiva si se usa --start")

    # Colección principal (por defecto SPL3SMP = radiometer L3 36 km)
    p.add_argument(
        "--short-name",
        default="SPL3SMP",
        help=(
            "short_name de la colección principal "
            "(default: SPL3SMP - L3 Radiometer Global Daily 36 km Soil Moisture)"
        ),
    )
    p.add_argument("--version", help="Número de versión (ej: 009) opcional para afinar búsqueda")
    p.add_argument("--concept-id", help="concept-id CMR (si se proporciona se ignora short_name/version)")

    # ---- Selección espacial
    p.add_argument(
        "--bbox",
        help="minLon,minLat,maxLon,maxLat (opcional). Se ignora si usas --cell-center o --boulder-36km.",
    )
    p.add_argument(
        "--boulder-36km",
        action="store_true",
        help="Usar un bbox fijo aproximado de un píxel (~36 km) centrado en Boulder.",
    )
    p.add_argument(
        "--cell-center",
        help="Centro de la celda pequeña como 'lon,lat'. Si se proporciona, "
             "se ignoran --bbox y --boulder-36km.",
    )
    p.add_argument(
        "--cell-size-km",
        type=float,
        default=36.0,
        help="Tamaño aproximado del lado de la celda en km (default: 36). "
             "Sólo se usa si se da --cell-center.",
    )

    p.add_argument("--threads", type=int, default=4, help="Hilos para descarga paralela")
    p.add_argument("--dry-run", action="store_true", help="Solo buscar, no descargar")
    p.add_argument(
        "--list-datasets",
        help="Buscar datasets por palabra clave y salir (ignora short_name/granulos)",
    )

    # ----- Radar adicional (SPL3SMA por defecto)
    p.add_argument(
        "--also-radar",
        action="store_true",
        help=(
            "Descargar también un producto de radar (backscatter / σ0) para el mismo rango "
            "temporal y bbox. Por defecto usa SPL3SMA (L3 Radar Soil Moisture 3 km)."
        ),
    )
    p.add_argument(
        "--radar-short-name",
        default="SPL3SMA",
        help="short_name de la colección radar usada cuando --also-radar. "
             "Default: SPL3SMA (L3 Radar Global Daily 3 km Soil Moisture / σ0).",
    )
    p.add_argument(
        "--radar-version",
        help="Versión de la colección radar (opcional, ej: 003). "
             "Ten en cuenta que SPL3SMA sólo existe aprox. 2015-04-13 a 2015-07-07.",
    )
    p.add_argument(
        "--radar-concept-id",
        help="concept-id CMR para radar (si se proporciona se ignora radar-short-name/version).",
    )

    return p.parse_args()


# -----------------------------------------------------------------------------
# Tiempo y bbox
# -----------------------------------------------------------------------------

def build_temporal(args: argparse.Namespace) -> Tuple[str, str]:
    fmt = "%Y-%m-%d"
    if args.date:
        d = datetime.strptime(args.date, fmt)
        return d.strftime(fmt), (d + timedelta(days=1)).strftime(fmt)
    if not args.end:
        raise SystemExit("--end requerido cuando usas --start")
    start_dt = datetime.strptime(args.start, fmt)
    end_dt = datetime.strptime(args.end, fmt)
    if end_dt <= start_dt:
        raise SystemExit("--end debe ser posterior a --start")
    return start_dt.strftime(fmt), end_dt.strftime(fmt)


def parse_bbox(bbox_str: Optional[str]):
    if not bbox_str:
        return None
    parts = bbox_str.split(",")
    if len(parts) != 4:
        raise SystemExit("--bbox formato invalido. Usa minLon,minLat,maxLon,maxLat")
    try:
        return tuple(float(p) for p in parts)
    except ValueError:
        raise SystemExit("--bbox valores deben ser numéricos")


def boulder_bbox_36km() -> Tuple[float, float, float, float]:
    """BBox aproximado de un píxel SMAP 36 km centrado en Boulder."""
    min_lon = -105.6
    max_lon = -104.9
    min_lat = 39.8
    max_lat = 40.2
    return (min_lon, min_lat, max_lon, max_lat)


def bbox_from_cell_center(center_str: str, size_km: float) -> Tuple[float, float, float, float]:
    """Construye un bbox aproximado cuadrado a partir de lon,lat y un tamaño en km."""
    try:
        lon_str, lat_str = center_str.split(",")
        lon0 = float(lon_str)
        lat0 = float(lat_str)
    except Exception:
        raise SystemExit("--cell-center debe tener formato 'lon,lat' con valores numéricos")

    half_km = size_km / 2.0
    # 1° lat ~ 111.32 km; 1° lon ~ 111.32 * cos(lat)
    dlat = half_km / 111.32
    cos_lat = math.cos(math.radians(lat0))
    if abs(cos_lat) < 1e-6:
        dlon = half_km / 111.32  # cerca de polos usamos aproximación tosca
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
    """Busca y (opcionalmente) descarga una colección SMAP."""
    print()
    print(f"[INFO] === Colección: {label} ===")

    if concept_id:
        search_kwargs = dict(concept_id=concept_id, temporal=temporal)
        desc = f"concept_id='{concept_id}'"
    else:
        if not short_name:
            raise SystemExit("short_name requerido si no se proporciona concept_id.")
        search_kwargs = dict(short_name=short_name, temporal=temporal)
        desc = f"short_name='{short_name}'"
        if version:
            search_kwargs["version"] = version
            desc += f" version={version}"

    if bbox:
        search_kwargs["bounding_box"] = bbox
        bbox_desc = f"bbox={bbox}"
    else:
        bbox_desc = "<global>"

    print(f"[INFO] Buscando {desc} temporal={temporal} {bbox_desc}")
    print(f"[DEBUG] search_kwargs={search_kwargs}")

    try:
        files = ea.search_data(**search_kwargs)
    except Exception as e:  # pragma: no cover
        print(f"[ERROR] Falla en search_data: {e}")
        sys.exit(1)

    print("[INFO] Granulos encontrados:", len(files))
    if not files:
        print("[WARN] Sin resultados para esta colección.")
        return []

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
            print(f"  [{i}] <error leyendo granulo>: {e}")

    if dry_run:
        print("[INFO] Dry-run: no se descargan archivos para esta colección.")
        return []

    out_dir = os.path.join(".", "data", "raw")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Descargando a '{out_dir}' con threads={threads}...")
    try:
        saved = ea.download(files, out_dir, threads=threads)
    except Exception as e:  # pragma: no cover
        print(f"[ERROR] Falla en descarga: {e}")
        sys.exit(3)
    print("[INFO] Descarga completada.")
    for p in saved:
        print("  -", p)
    return saved


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    try_login()

    # Modo listar datasets por keyword
    if args.list_datasets:
        print(f"[INFO] Listando datasets con keyword='{args.list_datasets}'...")
        try:
            ds = ea.search_datasets(keyword=args.list_datasets)
            print(f"[INFO] Encontrados {len(ds)} datasets")
            for i, d in enumerate(ds):
                title = getattr(d, "title", "<sin-title>")
                meta = getattr(d, "meta", {}) or {}
                cid = meta.get("concept-id", meta.get("concept_id", "<sin-concept-id>"))
                sn = getattr(d, "short_name", getattr(d, "shortName", "<sin-short-name>"))
                ver = getattr(d, "version_id", getattr(d, "version", "?"))
                print(f"  [{i}] short_name={sn} version={ver} concept_id={cid} | {title}")
        except Exception as e:
            print(f"[ERROR] No se pudo listar datasets: {e}")
        return

    temporal = build_temporal(args)

    # ---- Selección espacial con prioridades
    if args.cell_center:
        bbox = bbox_from_cell_center(args.cell_center, args.cell_size_km)
        print(f"[INFO] Usando bbox derivado de cell-center={args.cell_center}, "
              f"cell-size-km={args.cell_size_km}: {bbox}")
    elif args.boulder_36km:
        bbox = boulder_bbox_36km()
        print(f"[INFO] Usando bbox Boulder (~36 km): {bbox}")
    else:
        bbox = parse_bbox(args.bbox)
        print(f"[INFO] bbox={bbox or '<global>'}")

    # ---- Colección principal (default SPL3SMP)
    saved_main = search_and_download_one_collection(
        label="principal (p.ej. SPL3SMP)",
        temporal=temporal,
        bbox=bbox,
        short_name=args.short_name,
        version=args.version,
        concept_id=args.concept_id,
        threads=args.threads,
        dry_run=args.dry_run,
    )

    # ---- Radar adicional (SPL3SMA)
    if args.also_radar:
        saved_rad = search_and_download_one_collection(
            label="radar (p.ej. SPL3SMA)",
            temporal=temporal,
            bbox=bbox,
            short_name=args.radar_short_name,
            version=args.radar_version,
            concept_id=args.radar_concept_id,
            threads=args.threads,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            print("[INFO] Dry-run: radar NO descargado (solo listado).")
        else:
            print(f"[INFO] Radar descargado ({len(saved_rad)} archivos).")
            if len(saved_rad) == 0:
                print("[TIP] Recuerda que los productos radar (SPL3SMA/SPL3SMAP) "
                      "solo existen entre aprox. 2015-04-13 y 2015-07-07.")
    else:
        print("[INFO] --also-radar no activado: no se descarga producto radar adicional.")

    print("[INFO] Fin del script smap.py.")


if __name__ == "__main__":
    main()
