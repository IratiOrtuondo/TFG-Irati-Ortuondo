
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gcov_to_tiff_fixed.py
---------------------
Versión mejorada del script para convertir productos NISAR L2 GCOV (.h5) a GeoTIFF.

Principales mejoras respecto al original:
  - Corrige la georreferenciación: si x/y son CENTROS de píxel, desplaza medio píxel
    para ubicar la esquina superior izquierda (from_origin exige esquina, no centro).
  - Permite exportar múltiples bandas (p. ej. HHHH, HVHV, VVVV) via --bands.
  - Opción para derivar sigma0 a partir de gamma0 usando rtcGammaToSigmaFactor (--sigma0).
  - Exporta capas auxiliares (numberOfLooks y rtcGammaToSigmaFactor) con --ancillary.
  - Compresión DEFLATE, tiling 512x512 y BIGTIFF IF_SAFER para archivos grandes.
  - Búsqueda más robusta del grupo GCOV y opción --freq para escoger frequencyA o frequencyB.
  - Comentarios línea a línea para entender cada paso.
"""

# --------------------------- IMPORTS ---------------------------
import argparse          # parseo de argumentos CLI
import os                # utilidades de sistema de archivos
import sys               # acceso a stderr para logs
from typing import List, Optional

import numpy as np       # manejo de arrays y cálculos numéricos
import h5py              # lectura de archivos HDF5
import rasterio          # escritura de GeoTIFF
from rasterio.transform import from_origin   # crea la transformada afín
from rasterio.crs import CRS                # interpreta CRS en WKT
from loguru import logger                    # logging legible

# ---------------------- UTILIDADES BÁSICAS ---------------------
def to_db(arr: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Convierte valores lineales a dB de forma segura: 10*log10(max(arr, eps))."""
    # Clampeamos para evitar log10(0) y retornamos float32 (tamaño moderado)
    return (10.0 * np.log10(np.clip(arr, eps, None))).astype(np.float32)

def is_monotonic_and_equispaced(v: np.ndarray, rtol: float = 1e-6, atol: float = 1e-6) -> bool:
    """Comprueba si el vector v es monótono y equiespaciado dentro de una tolerancia."""
    if v.ndim != 1 or v.size < 2:
        return False
    dif = np.diff(v.astype(np.float64))
    # monótono creciente o decreciente
    mono = np.all(dif > 0) or np.all(dif < 0)
    if not mono:
        return False
    # equiespaciado (todas las diferencias ~ iguales)
    return np.allclose(dif, dif[0], rtol=rtol, atol=atol)

def build_transform(x: np.ndarray, y: np.ndarray) -> rasterio.Affine:
    """
    Construye la transformada de georreferenciación para un raster REGULAR.
    Supone que x,y representan los CENTROS de píxel por columna/fila.

    Pasos:
      - Calcula el tamaño de píxel en X e Y (absoluto).
      - Calcula la coordenada de la ESQUINA superior izquierda desplazando medio píxel
        desde el centro superior-izquierdo.
      - Usa from_origin(x_ul, y_ul, pix_w, pix_h).
    """
    # Validaciones básicas
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("xCoordinates/yCoordinates deben ser vectores 1D")
    if x.size < 2 or y.size < 2:
        raise ValueError("Se requieren al menos dos puntos en x/y para derivar resolución")
    if not is_monotonic_and_equispaced(x) or not is_monotonic_and_equispaced(y):
        raise ValueError("x/y no son monótonos y equiespaciados: un Affine único no es válido")

    # Resoluciones (pueden ser negativas si y decrece)
    resx = float(x[1] - x[0])
    resy = float(y[1] - y[0])

    # Tamaño de píxel positivo
    pix_w = abs(resx)
    pix_h = abs(resy)

    # ESQUINA superior izquierda (ojo al medio píxel)
    x_ul = float(x.min()) - pix_w / 2.0   # medio píxel a la IZQUIERDA del centro más a la izq.
    y_ul = float(y.max()) + pix_h / 2.0   # medio píxel por ENCIMA del centro superior

    # Transformada afín para raster regular (avanza +x a derecha y -y hacia abajo)
    return from_origin(x_ul, y_ul, pix_w, pix_h)

def save_tiff(path: str, arr: np.ndarray, transform, crs_wkt: Optional[str] = None):
    """
    Guarda un array 2D float32 como GeoTIFF con compresión y tiling.
    - Usa NaN como nodata (compatible con float32).
    - Añade CRS si se pudo interpretar el WKT correctamente.
    """
    profile = {
        "driver": "GTiff",
        "height": int(arr.shape[0]),
        "width": int(arr.shape[1]),
        "count": 1,
        "dtype": rasterio.float32,
        "transform": transform,
        "nodata": np.nan,
        # Mejoras de rendimiento/tamaño:
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "compress": "DEFLATE",
        "BIGTIFF": "IF_SAFER",
    }
    # Si viene un WKT, intentamos convertirlo a objeto CRS
    if crs_wkt:
        try:
            profile["crs"] = CRS.from_wkt(crs_wkt if isinstance(crs_wkt, str) else crs_wkt.decode("utf-8", "ignore"))
        except Exception as e:
            logger.warning(f"No se pudo interpretar CRS WKT: {e}")

    # Creamos carpeta de salida si no existe
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Escribimos el GeoTIFF
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(np.float32), 1)

def find_gcov_group(h5: h5py.File, freq_hint: Optional[str] = None) -> h5py.Group:
    """
    Localiza el grupo GCOV con heurística:
      1) /science/LSAR/GCOV/grids/<freq_hint> si se indicó --freq
      2) /science/LSAR/GCOV/grids/frequencyA o frequencyB
      3) Búsqueda recursiva de cualquier grupo que contenga 'GCOV' y x/yCoordinates
    """
    candidates = []
    if freq_hint:
        candidates.append(f"/science/LSAR/GCOV/grids/{freq_hint}")
    candidates += [
        "/science/LSAR/GCOV/grids/frequencyA",
        "/science/LSAR/GCOV/grids/frequencyB",
        "/science/LSAR/GCOV/grids",
    ]
    # 1) y 2)
    for p in candidates:
        if p in h5 and isinstance(h5[p], h5py.Group):
            grp = h5[p]
            # Si es 'grids' (sin frequency), bajamos al primer subgroup 'frequency*'
            if p.endswith("/grids"):
                for k in grp.keys():
                    if k.lower().startswith("frequency") and isinstance(grp[k], h5py.Group):
                        return grp[k]
            else:
                return grp

    # 3) búsqueda en profundidad
    def dfs(g: h5py.Group):
        for _, v in g.items():
            if isinstance(v, h5py.Group):
                keys = set(v.keys())
                if "GCOV" in v.name and {"xCoordinates", "yCoordinates"} <= keys:
                    return v
                hit = dfs(v)
                if hit is not None:
                    return hit
        return None

    found = dfs(h5)
    if found is None:
        raise KeyError("No pude localizar el grupo GCOV. Ejecuta primero inspect_h5.py para ver la estructura.")
    return found

# --------------------------- NÚCLEO ----------------------------
def export(
    h5path: str,
    outdir: str,
    bands: List[str],
    freq: Optional[str],
    linear: bool,
    sigma0: bool,
    ancillary: bool,
) -> None:
    """
    Abre el HDF5, localiza GCOV, y exporta las bandas solicitadas y capas auxiliares.
    - linear=False (por defecto) exporta en dB (10*log10).
    - sigma0=True aplica rtcGammaToSigmaFactor si existe; si no existe, avisa y exporta gamma0.
    - ancillary=True exporta numberOfLooks y rtcGammaToSigmaFactor si existen.
    """
    logger.info(f"Abriendo HDF5: {h5path}")
    with h5py.File(h5path, "r") as f:
        grp = find_gcov_group(f, freq)
        logger.info(f"Grupo GCOV: {grp.name}")

        # Enumeramos datasets disponibles para diagnosticar
        keys = sorted(grp.keys())
        logger.info(f"Datasets en GCOV: {keys}")

        # Leemos coordenadas y construimos transformada
        if not {"xCoordinates", "yCoordinates"} <= set(keys):
            raise RuntimeError("Faltan xCoordinates/yCoordinates en el grupo GCOV")
        x = grp["xCoordinates"][...]
        y = grp["yCoordinates"][...]
        transform = build_transform(x, y)

        # CRS en atributos del grupo (puede ser bytes o str)
        crs_wkt = grp.attrs.get("crsWKT", None)

        # Máscara (si existe). Convención común: 0 = válido
        mask = grp["mask"][...] if "mask" in keys else None

        # Factor RTC para convertir gamma0 -> sigma0 (si existe)
        rtc = grp["rtcGammaToSigmaFactor"][...] if "rtcGammaToSigmaFactor" in keys else None

        # Helper para aplicar máscara a un array 2D
        def mask_it(a: np.ndarray) -> np.ndarray:
            out = a.astype(np.float32, copy=True)
            if mask is not None and mask.shape == out.shape:
                out[mask != 0] = np.nan
            return out

        stem = os.path.splitext(os.path.basename(h5path))[0]
        os.makedirs(outdir, exist_ok=True)
        generated = []

        # Exportamos cada banda solicitada
        for b in bands:
            if b not in keys:
                logger.warning(f"Banda '{b}' no existe en GCOV; se omite.")
                continue
            # Leemos banda (gamma0 típicamente) y aplicamos máscara
            arr = mask_it(grp[b][...])

            # Si se pide sigma0 y hay rtc, multiplicamos en lineal
            label_sigma = ""
            if sigma0:
                if rtc is not None and rtc.shape == arr.shape:
                    arr = arr * rtc
                    label_sigma = "_sigma0"
                else:
                    logger.warning("Pediste sigma0 pero no existe rtcGammaToSigmaFactor (o forma distinta). Exporto gamma0.")

            # lineal vs dB
            if linear:
                out = arr
                suffix = f"{label_sigma}.tif" if label_sigma else ".tif"
            else:
                out = to_db(arr)
                suffix = f"{label_sigma}_dB.tif" if label_sigma else "_dB.tif"

            out_path = os.path.join(outdir, f"{stem}_{b}{suffix}")
            save_tiff(out_path, out, transform, crs_wkt)
            generated.append(out_path)
            logger.success(f"Generado: {out_path}")

        # Capas auxiliares opcionales
        if ancillary:
            if "numberOfLooks" in keys:
                nlooks = mask_it(grp["numberOfLooks"][...])
                p = os.path.join(outdir, f"{stem}_numberOfLooks.tif")
                save_tiff(p, nlooks, transform, crs_wkt)
                generated.append(p)
                logger.success(f"Generado: {p}")
            if "rtcGammaToSigmaFactor" in keys:
                p = os.path.join(outdir, f"{stem}_rtcGammaToSigmaFactor.tif")
                save_tiff(p, grp["rtcGammaToSigmaFactor"][...].astype(np.float32), transform, crs_wkt)
                generated.append(p)
                logger.success(f"Generado: {p}")

        # Resumen rápido (si exportamos al menos una banda principal)
        if generated:
            logger.info(f"Total archivos generados: {len(generated)}")
        else:
            logger.warning("No se generó ningún archivo. ¿Coinciden las bandas pedidas con las disponibles?")

# ----------------------------- CLI -----------------------------
def parse_args(argv=None):
    """Define y parsea argumentos de línea de comandos."""
    ap = argparse.ArgumentParser(description="Exporta bandas de NISAR L2 GCOV a GeoTIFF (corregido)")
    ap.add_argument("h5", help="Ruta al archivo NISAR L2 GCOV (.h5)")
    ap.add_argument("--out", default="data/interim", help="Carpeta de salida (default: data/interim)")
    ap.add_argument("--bands", default="HHHH", help="Lista separada por comas: HHHH,HVHV,VVVV (default: HHHH)")
    ap.add_argument("--freq", default=None, help="frequencyA o frequencyB (opcional)")
    ap.add_argument("--linear", action="store_true", help="Exporta en lineal (por defecto exporta en dB)")
    ap.add_argument("--sigma0", action="store_true", help="Aplica rtcGammaToSigmaFactor si existe (gamma0→sigma0)")
    ap.add_argument("--ancillary", action="store_true", help="Exporta numberOfLooks y rtcGammaToSigmaFactor si existen")
    return ap.parse_args(argv)

def main(argv=None):
    """Punto de entrada: parsea args y llama a export()."""
    args = parse_args(argv)

    # Prepara logging legible a stderr
    logger.remove()
    logger.add(sys.stderr, level="INFO", colorize=True,
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}")

    # Lista de bandas a exportar normalizada (sin espacios)
    bands = [b.strip() for b in args.bands.split(",") if b.strip()]

    export(
        h5path=args.h5,
        outdir=args.out,
        bands=bands,
        freq=args.freq,
        linear=args.linear,
        sigma0=args.sigma0,
        ancillary=args.ancillary,
    )

if __name__ == "__main__":
    main()
