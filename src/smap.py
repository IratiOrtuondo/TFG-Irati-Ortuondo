"""Descarga granulos SMAP vía earthaccess con CLI flexible.

Ejemplos PowerShell:
  # Descargar TB (SPL3TB_E) de un día
  python src\smap.py --date 2021-06-01

  # Rango de fechas (inicio inclusive, fin exclusiva) => 3 días
  python src\smap.py --start 2021-06-01 --end 2021-06-04

  # Cambiar colección a Soil Moisture (ej: SPL3SMP)
  python src\smap.py --date 2021-06-01 --short-name SPL3SMP

  # Usar bbox (aunque TB global no lo necesita)
  python src\smap.py --date 2021-06-01 --bbox -118.46,34.23,-117.48,35.36

Variables de entorno necesarias si no tienes ~/.netrc:
  $env:EARTHDATA_USERNAME="tu_usuario"
  $env:EARTHDATA_PASSWORD="tu_password"

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
from typing import Tuple, Optional
import earthaccess as ea


def try_login() -> None:
    """Intenta login con variables de entorno y si falla, con netrc.

    Imprime advertencias si no se pudo autenticar.
    """
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Buscar y descargar granulos SMAP con earthaccess")
    gdate = p.add_mutually_exclusive_group(required=True)
    gdate.add_argument("--date", help="Fecha única (YYYY-MM-DD)")
    gdate.add_argument("--start", help="Fecha inicio (YYYY-MM-DD) para rango")
    p.add_argument("--end", help="Fecha fin (YYYY-MM-DD) exclusiva si se usa --start")
    p.add_argument("--short-name", default="SPL3SMP", help="short_name de la colección (default: SPL3SMP - L3 Soil Moisture 36 km que incluye TB corregido)")
    p.add_argument("--version", help="Número de versión (ej: 009) opcional para afinar búsqueda")
    p.add_argument("--concept-id", help="concept-id CMR (si se proporciona se ignora short_name/version)")
    p.add_argument("--bbox", help="minLon,minLat,maxLon,maxLat (opcional)")
    p.add_argument("--threads", type=int, default=4, help="Hilos para descarga paralela")
    p.add_argument("--dry-run", action="store_true", help="Solo buscar, no descargar")
    p.add_argument("--list-datasets", help="Buscar datasets por palabra clave y salir (ignora short_name/granulos)")
    return p.parse_args()


def build_temporal(args: argparse.Namespace) -> Tuple[str, str]:
    fmt = "%Y-%m-%d"
    if args.date:
        d = datetime.strptime(args.date, fmt)
        return d.strftime(fmt), (d + timedelta(days=1)).strftime(fmt)
    # rango
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
    bbox = parse_bbox(args.bbox)
    print(f"[INFO] Buscando short_name='{args.short_name}' temporal={temporal} bbox={bbox or '<global>'}")

    if args.concept_id:
        search_kwargs = dict(concept_id=args.concept_id, temporal=temporal)
    else:
        search_kwargs = dict(short_name=args.short_name, temporal=temporal)
        if args.version:
            search_kwargs["version"] = args.version
    print(f"[DEBUG] search_kwargs={search_kwargs}")
    if bbox:
        search_kwargs["bounding_box"] = bbox
    try:
        files = ea.search_data(**search_kwargs)
    except Exception as e:  # pragma: no cover
        print(f"[ERROR] Falla en search_data: {e}")
        sys.exit(1)

    print("[INFO] Granulos encontrados:", len(files))
    if not files:
        print("[WARN] Sin resultados. Intentando listar datasets que coincidan con el short_name para diagnosticar...")
        try:
            ds = ea.search_datasets(short_name=args.short_name)
            if ds:
                print(f"[INFO] Datasets encontrados ({len(ds)}):")
                for i, d in enumerate(ds):
                    cid = getattr(d, "meta", {}).get("concept-id", "<sin-concept-id>")
                    title = getattr(d, "title", "<sin-title>")
                    # versiones posibles
                    versions = getattr(d, "version_id", getattr(d, "version", "?"))
                    print(f"  [{i}] title={title} concept_id={cid} version={versions}")
                print("[TIP] Puede que necesites especificar version/concept_id o usar un rango temporal distinto.")
            else:
                print("[WARN] No se encontraron datasets con ese short_name.")
        except Exception as e2:  # pragma: no cover
            print(f"[WARN] search_datasets falló: {e2}")
        sys.exit(2)

    # Mostrar resumen de los primeros
    for i, f in enumerate(files[:5]):
        try:
            size_mb = getattr(f, "size", None)
            if size_mb is not None:
                try:
                    size_mb = float(size_mb) / (1024 * 1024)
                except Exception:
                    size_mb = None
            print(f"  [{i}] {safe_short_name(f)} | id={getattr(f,'granule_id', '<no-id>')} | sizeMB={size_mb:.2f}" if size_mb else f"  [{i}] {safe_short_name(f)}")
        except Exception as e:
            print(f"  [{i}] <error leyendo granulo>: {e}")

    if args.dry_run:
        print("[INFO] Dry-run: no se descargan archivos.")
        return

    out_dir = os.path.join(".", "data", "raw")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Descargando a '{out_dir}' con threads={args.threads}...")
    try:
        saved = ea.download(files, out_dir, threads=args.threads)
    except Exception as e:  # pragma: no cover
        print(f"[ERROR] Falla en descarga: {e}")
        sys.exit(3)
    print("[INFO] Descarga completada.")
    for p in saved:
        print("  -", p)


if __name__ == "__main__":
    main()
