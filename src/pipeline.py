#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paso 1 (NISAR) – Preparación sin resampling:
- Lista freqs/pols disponibles
- Exporta SLC complejo (radar grid) a ENVI
- Calcula potencia y aplica LUTs de calibración (beta0, gamma0, sigma0) por interpolación
- Interpola incidence angle al tamaño completo (radar grid)
- Escribe GCPs derivados del geolocationGrid (sin re-muestrear la imagen)
NOTA: No se hace filtrado speckle ni normalización angular. No se re-muestrea.

Cómo probarlo: python nisar_step1.py /ruta/a/tu/NISAR_SLC.h5 --freq frequencyA --pol VV --outdir out_nisar --chunk 512

"""

import argparse
import h5py
import numpy as np
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
import rasterio
from rasterio.transform import Affine
from rasterio.control import GCP
from rasterio.io import MemoryFile
from tqdm import tqdm

# -------------------------
# Utilidades de lectura HDF5
# -------------------------

def h5_exists(h5: h5py.File, path: str) -> bool:
    try:
        h5[path]
        return True
    except KeyError:
        return False

def read_h5(h5: h5py.File, path: str):
    return h5[path][()]

def list_pols(h5: h5py.File):
    base = "/science/LSAR/SLC/swaths"
    out = []
    if not h5_exists(h5, base):
        raise RuntimeError(f"No existe {base} en el HDF5. ¿Es un SLC NISAR?")
    for freq in h5[base].keys():  # frequencyA, frequencyB...
        group = f"{base}/{freq}"
        # pols dentro de frequencyX son datasets complejos (HH,HV,VH,VV) si existen
        pols = [k for k, v in h5[group].items() if isinstance(v, h5py.Dataset)]
        if pols:
            out.append((freq, sorted(pols)))
    return out  # [(frequencyA, [HH,VH,...]), ...]

# -------------------------
# Interpoladores LUT (392x225)
# -------------------------

def build_lut_interpolator(h5: h5py.File, lut_name: str):
    """
    Crea un interpolador 2D para una LUT en:
    /science/LSAR/SLC/metadata/calibrationInformation/geometry/<lut_name>
    ejes:
    /science/LSAR/SLC/metadata/calibrationInformation/zeroDopplerTime  (Nz=392)
    /science/LSAR/SLC/metadata/calibrationInformation/slantRange       (Nr=225)
    """
    base = "/science/LSAR/SLC/metadata/calibrationInformation"
    lut_path = f"{base}/geometry/{lut_name}"
    zdt_path = f"{base}/zeroDopplerTime"
    srg_path = f"{base}/slantRange"
    if not (h5_exists(h5, lut_path) and h5_exists(h5, zdt_path) and h5_exists(h5, srg_path)):
        raise RuntimeError(f"Faltan LUT/axes para {lut_name}")

    lut = read_h5(h5, lut_path)            # shape (Nz, Nr)
    zdt = read_h5(h5, zdt_path).astype(np.float64)  # (Nz,)
    srg = read_h5(h5, srg_path).astype(np.float64)  # (Nr,)

    # Asegura monotonicidad
    assert zdt.ndim == 1 and srg.ndim == 1 and lut.shape == (zdt.size, srg.size)

    # Interpolador bilineal 2D sobre coordenadas físicas (tiempo, rango)
    interp = RegularGridInterpolator(
        (zdt, srg), lut, bounds_error=False, fill_value=np.nan
    )
    return interp, zdt, srg

def build_incidence_interpolator(h5: h5py.File):
    """
    Incidence angle grid:
    /science/LSAR/SLC/metadata/geolocationGrid/incidenceAngle   shape (6, Nz, Nr)
    Con mismos ejes coarse:
    /science/LSAR/SLC/metadata/geolocationGrid/zeroDopplerTime  (Nz,)
    /science/LSAR/SLC/metadata/geolocationGrid/slantRange       (Nr,)
    """
    base = "/science/LSAR/SLC/metadata/geolocationGrid"
    ic_path = f"{base}/incidenceAngle"
    zdt_path = f"{base}/zeroDopplerTime"
    srg_path = f"{base}/slantRange"
    if not (h5_exists(h5, ic_path) and h5_exists(h5, zdt_path) and h5_exists(h5, srg_path)):
        raise RuntimeError("Faltan grids de geolocalización para incidenceAngle")

    ic = read_h5(h5, ic_path).astype(np.float32)      # (6, Nz, Nr)
    zdt = read_h5(h5, zdt_path).astype(np.float64)    # (Nz,)
    srg = read_h5(h5, srg_path).astype(np.float64)    # (Nr,)

    # Colapsamos la dimensión 0 (6 capas) con una media robusta
    ic_mean = np.nanmean(ic, axis=0)  # (Nz, Nr)

    interp = RegularGridInterpolator(
        (zdt, srg), ic_mean, bounds_error=False, fill_value=np.nan
    )
    return interp, zdt, srg

# -------------------------
# Ejes de la imagen a coordenadas físicas (tiempo/rango)
# -------------------------

def read_full_axes(h5: h5py.File, freq: str):
    """
    Intenta leer los ejes de línea (zeroDopplerTime por línea) y rango (slantRange por columna).
    Si no existen de forma explícita, aproxima con inicio+espaciado.
    """
    meta = "/science/LSAR/SLC/metadata"
    # Líneas:
    zdt_full = None
    for cand in [
        f"{meta}/zeroDopplerTime",  # algunos simulados lo traen así
        f"{meta}/swaths/{freq}/zeroDopplerTime",
        f"/science/LSAR/SLC/swaths/{freq}/zeroDopplerTime",
    ]:
        if h5_exists(h5, cand):
            zdt_full = read_h5(h5, cand).astype(np.float64)
            break

    if zdt_full is None:
        # fallback: usar start + spacing, si están
        zdt_start = None
        zdt_spacing = None
        for c in [f"{meta}/zeroDopplerStartTime", f"{meta}/swaths/{freq}/zeroDopplerStartTime"]:
            if h5_exists(h5, c):
                # suele ser string ISO; aquí lo dejamos como "índice de línea" si no se puede parsear
                pass
        if h5_exists(h5, f"{meta}/zeroDopplerTimeSpacing"):
            zdt_spacing = float(read_h5(h5, f"{meta}/zeroDopplerTimeSpacing"))
        # Si no hay nada, devolvemos None y usaremos índice normalizado
    # Columnas:
    srg_full = None
    for cand in [
        f"{meta}/slantRange",               # a veces existe con longitud = width
        f"{meta}/swaths/{freq}/slantRange",
        f"/science/LSAR/SLC/swaths/{freq}/slantRange",
    ]:
        if h5_exists(h5, cand):
            srg_full = read_h5(h5, cand).astype(np.float64)
            break

    srg_spacing = None
    if srg_full is None:
        for cand in [
            f"{meta}/slantRangeSpacing",
            f"{meta}/swaths/{freq}/slantRangeSpacing",
            f"/science/LSAR/SLC/swaths/{freq}/slantRangeSpacing",
        ]:
            if h5_exists(h5, cand):
                srg_spacing = float(read_h5(h5, cand))
                break

    return zdt_full, srg_full, srg_spacing

# -------------------------
# Export SLC, Sigma0, IncAngle y GCPs
# -------------------------

def write_complex_envi(out_path: Path, data: np.ndarray):
    """
    Escribe un array complejo (lines, cols) en ENVI sin geotransform.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "ENVI",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": "complex64",
        "transform": Affine.identity(),
        "crs": None
    }
    with rasterio.open(out_path.as_posix(), "w", **profile) as dst:
        dst.write(data[np.newaxis, ...])

def write_float_geotiff(out_path: Path, data: np.ndarray, gcps=None, epsg=None):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": "float32",
        "transform": Affine.identity(),
        "crs": None,
        "tiled": True,
        "compress": "DEFLATE",
        "predictor": 2
    }
    with rasterio.open(out_path.as_posix(), "w", **profile) as dst:
        if gcps is not None:
            dst.gcps = (gcps, None if epsg is None else rasterio.crs.CRS.from_epsg(epsg))
        dst.write(data[np.newaxis, ...])

def build_gcps_from_geogrid(h5: h5py.File, nlines: int, ncols: int, step_lines: int = 64, step_cols: int = 64):
    """
    Crea GCPs dispersos (sin resampling) usando geolocationGrid (coordenadas X/Y y EPSG).
    Para no inflar el archivo, muestreamos cada 'step_lines/cols' píxeles.
    """
    base = "/science/LSAR/SLC/metadata/geolocationGrid"
    x_path = f"{base}/coordinateX"
    y_path = f"{base}/coordinateY"
    epsg_path = f"{base}/epsg"

    if not (h5_exists(h5, x_path) and h5_exists(h5, y_path) and h5_exists(h5, epsg_path)):
        return [], None

    X = read_h5(h5, x_path).astype(np.float64)  # (6, Nz, Nr)
    Y = read_h5(h5, y_path).astype(np.float64)
    epsg = int(read_h5(h5, epsg_path))

    # Media sobre la dimensión 0 (6) para un único grid
    Xmean = np.nanmean(X, axis=0)  # (Nz, Nr)
    Ymean = np.nanmean(Y, axis=0)

    Nz, Nr = Xmean.shape
    # malla en coordenadas de imagen (línea/columna) para esos Nz×Nr nodos
    row_coarse = np.linspace(0, nlines - 1, Nz)
    col_coarse = np.linspace(0, ncols - 1, Nr)

    gcps = []
    for i in range(0, Nz, max(1, Nz // (nlines // step_lines + 1))):
        for j in range(0, Nr, max(1, Nr // (ncols // step_cols + 1))):
            r = float(row_coarse[i])
            c = float(col_coarse[j])
            x = float(Xmean[i, j])
            y = float(Ymean[i, j])
            # GCP(row, col, x, y)
            gcps.append(GCP(col=c, row=r, x=x, y=y))
    return gcps, epsg

# -------------------------
# Flujo principal
# -------------------------

def run(input_h5: Path, outdir: Path, freq: str, pol: str, chunk_lines: int = 1024):
    input_h5 = Path(input_h5)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_h5, "r") as h5:

        # 1) Descubrir pols/frecuencias disponibles
        freqs_pols = list_pols(h5)
        print("Frec/Pols disponibles:")
        for f, ps in freqs_pols:
            print(f"  {f}: {', '.join(ps)}")

        # Validar selección
        dataset_path = f"/science/LSAR/SLC/swaths/{freq}/{pol}"
        if not h5_exists(h5, dataset_path):
            raise RuntimeError(f"No existe {dataset_path}. Elige otra freq/pol.")

        dset = h5[dataset_path]
        nlines, ncols = dset.shape
        print(f"SLC seleccionado: {freq}/{pol} -> tamaño {nlines}x{ncols}")

        # 2) Construir interpoladores de LUTs (beta0, gamma0, sigma0) y incidence angle
        sigma_interp, zdt_lut, srg_lut = build_lut_interpolator(h5, "sigma0")
        beta_interp,  _, _           = build_lut_interpolator(h5, "beta0")
        gamma_interp, _, _           = build_lut_interpolator(h5, "gamma0")
        inc_interp,   _, _           = build_incidence_interpolator(h5)

        # 3) Ejes físicos a resolución completa (si están; si no, se aproximan vía índice)
        zdt_full, srg_full, srg_spacing = read_full_axes(h5, freq=freq)

        # Construir vectores de tiempo y rango por píxel (línea/columna)
        if zdt_full is None:
            # fallback: índice de línea -> normalizamos al dominio de la LUT por proporción
            zdt_full = np.linspace(zdt_lut.min(), zdt_lut.max(), nlines)
        if srg_full is None:
            if srg_spacing is not None:
                srg_full = np.linspace(srg_lut.min(), srg_lut.min() + srg_spacing * (ncols - 1), ncols)
            else:
                srg_full = np.linspace(srg_lut.min(), srg_lut.max(), ncols)

        # 4) Exportar SLC complejo (radar grid) y calcular productos calibrados
        slc_out = outdir / f"NISAR_{freq}_{pol}_SLC.envi"
        sigma_out = outdir / f"NISAR_{freq}_{pol}_sigma0.tif"
        beta_out = outdir / f"NISAR_{freq}_{pol}_beta0.tif"
        gamma_out = outdir / f"NISAR_{freq}_{pol}_gamma0.tif"
        inc_out = outdir / f"NISAR_{freq}_{pol}_incidenceAngle.tif"

        # Construimos GCPs una vez
        gcps, epsg = build_gcps_from_geogrid(h5, nlines, ncols)

        # Prealocar salidas en disco (streaming por chunks)
        # SLC complejo
        # (escribimos al final para evitar crear archivo enorme si algo falla antes)
        slc_collect = []

        # Sigma/Beta/Gamma/Incidence (float32) -> escribimos por bloques
        with rasterio.open(
            sigma_out.as_posix(), "w",
            driver="GTiff", height=nlines, width=ncols, count=1, dtype="float32",
            transform=Affine.identity(), crs=None, tiled=True, compress="DEFLATE", predictor=2
        ) as ds_sigma, \
        rasterio.open(
            beta_out.as_posix(), "w",
            driver="GTiff", height=nlines, width=ncols, count=1, dtype="float32",
            transform=Affine.identity(), crs=None, tiled=True, compress="DEFLATE", predictor=2
        ) as ds_beta, \
        rasterio.open(
            gamma_out.as_posix(), "w",
            driver="GTiff", height=nlines, width=ncols, count=1, dtype="float32",
            transform=Affine.identity(), crs=None, tiled=True, compress="DEFLATE", predictor=2
        ) as ds_gamma, \
        rasterio.open(
            inc_out.as_posix(), "w",
            driver="GTiff", height=nlines, width=ncols, count=1, dtype="float32",
            transform=Affine.identity(), crs=None, tiled=True, compress="DEFLATE", predictor=2
        ) as ds_inc:

            # Añadimos GCPs una sola vez
            if gcps:
                ds_sigma.gcps = (gcps, None if epsg is None else rasterio.crs.CRS.from_epsg(epsg))
                ds_beta.gcps = ds_sigma.gcps
                ds_gamma.gcps = ds_sigma.gcps
                ds_inc.gcps = ds_sigma.gcps

            # Preparar mallas 1D
            srg_full = srg_full.astype(np.float64)
            zdt_full = zdt_full.astype(np.float64)

            for row0 in tqdm(range(0, nlines, chunk_lines), desc="Procesando bloques"):
                row1 = min(nlines, row0 + chunk_lines)
                rows = row1 - row0

                # Leemos bloque complejo
                slc_block = dset[row0:row1, :].astype(np.complex64)  # (rows, ncols)
                slc_collect.append(slc_block)  # temporal (podría stream a ENVI si se pre-crea)

                # Potencia (DN^2) en lineal
                power = (slc_block.real**2 + slc_block.imag**2).astype(np.float32)

                # Coordenadas físicas para el bloque (tiempo, rango)
                zdt_block = zdt_full[row0:row1]                    # (rows,)
                srg_vec = srg_full                                 # (ncols,)

                # Creamos las mallas 2D de consulta para interpolación
                # OJO: RegularGridInterpolator usa puntos como (time, range)
                T, R = np.meshgrid(zdt_block, srg_vec, indexing="ij")  # (rows, ncols)

                pts = np.stack([T.ravel(), R.ravel()], axis=1)

                # Interpolar factores LUT y ángulo de incidencia
                sigma_fac = sigma_interp(pts).reshape(rows, ncols).astype(np.float32)
                beta_fac  =  beta_interp(pts).reshape(rows, ncols).astype(np.float32)
                gamma_fac = gamma_interp(pts).reshape(rows, ncols).astype(np.float32)
                inc_deg   =   inc_interp(pts).reshape(rows, ncols).astype(np.float32)

                # Productos calibrados (lineal)
                sigma0 = power * sigma_fac
                beta0  = power * beta_fac
                gamma0 = power * gamma_fac

                # Escribir a disco
                window = rasterio.windows.Window(0, row0, ncols, rows)
                ds_sigma.write(sigma0[np.newaxis, ...], window=window, indexes=1)
                ds_beta.write(beta0[np.newaxis, ...], window=window, indexes=1)
                ds_gamma.write(gamma0[np.newaxis, ...], window=window, indexes=1)
                ds_inc.write(inc_deg[np.newaxis, ...], window=window, indexes=1)

        # Finalmente, escribir SLC complejo en ENVI (concatenamos filas)
        slc_full = np.vstack(slc_collect)
        write_complex_envi(slc_out, slc_full)

    print("\nHecho ✅")
    print(f"SLC (ENVI):             {slc_out}")
    print(f"sigma0 (GTiff):         {sigma_out}")
    print(f"beta0  (GTiff):         {beta_out}")
    print(f"gamma0 (GTiff):         {gamma_out}")
    print(f"incidenceAngle (GTiff): {inc_out}")
    print("\nNOTA: Los GeoTIFF llevan GCPs (sin resampling). Usa estos GCPs para el coregistro en tu método de inversión.")


def main():
    ap = argparse.ArgumentParser(description="Paso 1 NISAR: preparación sin resampling")
    ap.add_argument("input_h5", type=str, help="Archivo NISAR HDF5 (RSLC/SLC)")
    ap.add_argument("--outdir", type=str, default="out_nisar", help="Carpeta de salida")
    ap.add_argument("--freq", type=str, default="frequencyA", help="frequencyA | frequencyB")
    ap.add_argument("--pol", type=str, default="VV", help="HH|HV|VH|VV (según producto)")
    ap.add_argument("--chunk", type=int, default=1024, help="líneas por bloque (memoria)")
    args = ap.parse_args()

    run(Path(args.input_h5), Path(args.outdir), args.freq, args.pol, chunk_lines=args.chunk)

if __name__ == "__main__":
    main()
