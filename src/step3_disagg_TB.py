#!/usr/bin/env python3
"""step5_disagg_native.py

Disaggregate SMAP brightness temperature from 36 km to native resolution
using the equation:

TB_p(M_j) = TB_p(C) + β(C) * [(σ_pp(M_j) - σ_pp(C)) + Γ(C) * (σ_pq(C) - σ_pq(M_j))]

where:
- TB_p(C): SMAP radiometer brightness temperature at 36 km
- σ_pp(M_j): co-pol backscatter at native resolution (~3 km for SMAP L3)
- σ_pq(M_j): cross-pol backscatter at native resolution (from L1C)
- σ_pp(C): co-pol backscatter aggregated to 36 km
- σ_pq(C): cross-pol backscatter aggregated to 36 km
- β(C): scale factor [K/dB] from Step 3 regression
- Γ(C): heterogeneity parameter from Step 3 regression

Inputs:
- Step 3 outputs: beta, gamma maps (GeoTIFF or NPZ)
- SMAP TB 36 km (NPZ from step2)
- Native co-pol NPZ: aligned-smap-copol-<date>-{vv|hh}-native.npz
- Native cross-pol NPZ: aligned-smap-xpol-<date>-native.npz
- Coarse co-pol NPZ: aligned-smap-copol-<date>-{vv|hh}.npz
- Coarse cross-pol NPZ: aligned-smap-xpol-<date>.npz

Outputs:
- TB_fine_<date>_<pol>_native.npz: disaggregated TB at native resolution

Usage:
  python src/step5_disagg_native.py --date 20150502 --pol VV \
      --step3-dir data/processed --step2-dir data/interim --interim-dir data/interim \
      --out-dir data/processed
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.warp import reproject, Resampling


def load_geotiff(path: Path) -> tuple[np.ndarray, Affine, CRS]:
    """Load GeoTIFF and return array, transform, CRS."""
    with rasterio.open(path) as src:
        arr = src.read(1)
        tf = src.transform
        crs = src.crs
    return arr, tf, crs


def load_npz_with_transform(path: Path) -> tuple[np.ndarray, Affine, CRS]:
    """Load NPZ with transform and CRS."""
    data = np.load(path)
    
    # Determine array key
    if 'S_copol_dB_native' in data:
        arr = data['S_copol_dB_native']
    elif 'S_xpol_dB_native' in data:
        arr = data['S_xpol_dB_native']
    elif 'S_copol_dB' in data:
        arr = data['S_copol_dB']
    elif 'S_xpol_dB' in data:
        arr = data['S_xpol_dB']
    elif 'TB36_V' in data or 'TB36_H' in data:
        # TB data
        arr = None
    else:
        raise ValueError(f"Unknown NPZ format in {path}")
    
    # Get transform
    if 'transform_native' in data:
        tf_arr = data['transform_native']
        tf = Affine(tf_arr[0], tf_arr[1], tf_arr[2], tf_arr[3], tf_arr[4], tf_arr[5])
    elif 'transform' in data:
        tf_arr = data['transform']
        tf = Affine(tf_arr[0], tf_arr[1], tf_arr[2], tf_arr[3], tf_arr[4], tf_arr[5])
    elif 'lat_native' in data and 'lon_native' in data:
        # Compute transform from lat/lon arrays
        from rasterio.transform import from_origin
        lat = data['lat_native']
        lon = data['lon_native']
        dx = np.mean(np.diff(lon[0, :]))
        dy = np.mean(np.diff(lat[:, 0]))
        left = float(np.min(lon)) - abs(dx) / 2.0
        top = float(np.max(lat)) + abs(dy) / 2.0
        tf = from_origin(left, top, abs(dx), abs(dy))
    else:
        raise ValueError(f"No transform or lat/lon found in {path}")
    
    # Get CRS
    if 'crs_wkt' in data:
        crs = CRS.from_wkt(data['crs_wkt'].item())
    else:
        crs = CRS.from_epsg(6933)  # Default EASE2
    
    return arr, tf, crs


def reproject_to_target(
    src_arr: np.ndarray,
    src_tf: Affine,
    src_crs: CRS,
    dst_tf: Affine,
    dst_crs: CRS,
    dst_shape: tuple[int, int],
) -> np.ndarray:
    """Reproject source array to target grid."""
    dst_arr = np.full(dst_shape, np.nan, dtype=np.float32)
    
    reproject(
        source=src_arr,
        destination=dst_arr,
        src_transform=src_tf,
        src_crs=src_crs,
        dst_transform=dst_tf,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    
    return dst_arr


def main():
    parser = argparse.ArgumentParser(description="Step 5: Disaggregate TB to native resolution")
    parser.add_argument("--date", required=True, help="Date YYYYMMDD")
    parser.add_argument("--pol", required=True, choices=["VV", "HH"], help="Polarization")
    parser.add_argument("--step3-dir", required=True, help="Directory with Step 3 outputs (beta, gamma)")
    parser.add_argument("--step2-dir", required=True, help="Directory with Step 2 outputs (TB 36km)")
    parser.add_argument("--interim-dir", required=True, help="Directory with interim NPZ files")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--beta-file", help="Override beta file path")
    parser.add_argument("--gamma-file", help="Override gamma file path")
    
    args = parser.parse_args()
    
    date = args.date
    pol = args.pol
    pol_lower = pol.lower()
    
    step3_dir = Path(args.step3_dir)
    step2_dir = Path(args.step2_dir)
    interim_dir = Path(args.interim_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=== Step 5: Disaggregate TB to native resolution ===")
    print(f"Date: {date}")
    print(f"Polarization: {pol}")
    
    # 1. Load Step 3 outputs: β and Γ
    if args.beta_file:
        beta_path = Path(args.beta_file)
    else:
        beta_path = step3_dir / f"beta_{pol}_xpol.tif"
    
    if args.gamma_file:
        gamma_path = Path(args.gamma_file)
    else:
        gamma_path = step3_dir / f"gamma_{pol}_xpol.tif"
    
    # Load β and Γ from Step 3 NPZ
    step3_npz = step2_dir / "step3_beta_gamma.npz"
    
    print(f"\n[1/7] Loading β and Γ from {step3_npz}")
    if not step3_npz.exists():
        raise FileNotFoundError(f"Step 3 output not found: {step3_npz}")
    
    npz3 = np.load(step3_npz)
    beta = npz3["beta_K_per_dB"]
    gamma = npz3["Gamma_dimensionless"]
    
    # Get transform and CRS from Step 3
    tf_arr = npz3["transform"]
    beta_tf = Affine(tf_arr[0], tf_arr[1], tf_arr[2], tf_arr[3], tf_arr[4], tf_arr[5])
    gamma_tf = beta_tf
    beta_crs = CRS.from_wkt(str(npz3["crs_wkt"]))
    gamma_crs = beta_crs
    
    print(f"  β shape: {beta.shape}, finite: {np.isfinite(beta).sum()}")
    print(f"  Γ shape: {gamma.shape}, finite: {np.isfinite(gamma).sum()}")
    
    # 2. Load TB 36 km
    # Map radar pol to radiometer pol (VV->V, HH->H)
    tb_pol = 'V' if pol == 'VV' else 'H'
    tb_path = step2_dir / f"smap-tb36-{date}-{tb_pol.lower()}.npz"
    print(f"\n[3/7] Loading TB 36km from {tb_path}")
    tb_data = np.load(tb_path)
    
    # New TB extraction saves as TB_36km (polarization is in metadata)
    tb36 = tb_data['TB_36km']
    
    tb36_tf_arr = tb_data['transform']
    tb36_tf = Affine(tb36_tf_arr[0], tb36_tf_arr[1], tb36_tf_arr[2],
                     tb36_tf_arr[3], tb36_tf_arr[4], tb36_tf_arr[5])
    tb36_crs = CRS.from_wkt(str(tb_data['crs_wkt']))
    print(f"  TB36 shape: {tb36.shape}, finite: {np.isfinite(tb36).sum()}")
    
    # 3. Load native co-pol
    copol_native_path = interim_dir / f"aligned-smap-copol-{date}-{pol_lower}-native.npz"
    print(f"\n[4/7] Loading native co-pol from {copol_native_path}")
    copol_native, copol_native_tf, copol_native_crs = load_npz_with_transform(copol_native_path)
    print(f"  Native co-pol shape: {copol_native.shape}, finite: {np.isfinite(copol_native).sum()}")
    
    # 4. Load native cross-pol
    xpol_native_path = interim_dir / f"aligned-smap-xpol-{date}-native.npz"
    print(f"\n[5/7] Loading native cross-pol from {xpol_native_path}")
    xpol_native, xpol_native_tf, xpol_native_crs = load_npz_with_transform(xpol_native_path)
    print(f"  Native cross-pol shape: {xpol_native.shape}, finite: {np.isfinite(xpol_native).sum()}")
    
    # Check if co-pol and cross-pol grids match
    if copol_native.shape != xpol_native.shape:
        print(f"  [WARN] Co-pol and cross-pol shapes differ: {copol_native.shape} vs {xpol_native.shape}")
        print(f"  [INFO] Reprojecting cross-pol to co-pol grid...")
        xpol_native = reproject_to_target(
            xpol_native, xpol_native_tf, xpol_native_crs,
            copol_native_tf, copol_native_crs, copol_native.shape
        )
        print(f"  [OK] Cross-pol reprojected, finite: {np.isfinite(xpol_native).sum()}")
    
    # Use co-pol native grid as target grid
    target_shape = copol_native.shape
    target_tf = copol_native_tf
    target_crs = copol_native_crs
    
    # 5. Load coarse co-pol (36 km aggregated)
    copol_coarse_path = interim_dir / f"aligned-smap-copol-{date}-{pol_lower}.npz"
    print(f"\n[6/7] Loading coarse co-pol from {copol_coarse_path}")
    copol_coarse_data = np.load(copol_coarse_path)
    copol_coarse = copol_coarse_data['S_copol_dB']
    copol_coarse_tf_arr = copol_coarse_data['transform']
    copol_coarse_tf = Affine(copol_coarse_tf_arr[0], copol_coarse_tf_arr[1], copol_coarse_tf_arr[2],
                              copol_coarse_tf_arr[3], copol_coarse_tf_arr[4], copol_coarse_tf_arr[5])
    copol_coarse_crs = CRS.from_wkt(copol_coarse_data['crs_wkt'].item())
    print(f"  Coarse co-pol shape: {copol_coarse.shape}, finite: {np.isfinite(copol_coarse).sum()}")
    
    # 6. Load coarse cross-pol (36 km aggregated)
    xpol_coarse_path = interim_dir / f"aligned-smap-xpol-{date}.npz"
    print(f"[7/7] Loading coarse cross-pol from {xpol_coarse_path}")
    xpol_coarse_data = np.load(xpol_coarse_path)
    xpol_coarse = xpol_coarse_data['S_xpol_dB']
    xpol_coarse_tf_arr = xpol_coarse_data['transform']
    xpol_coarse_tf = Affine(xpol_coarse_tf_arr[0], xpol_coarse_tf_arr[1], xpol_coarse_tf_arr[2],
                             xpol_coarse_tf_arr[3], xpol_coarse_tf_arr[4], xpol_coarse_tf_arr[5])
    xpol_coarse_crs = CRS.from_wkt(xpol_coarse_data['crs_wkt'].item())
    print(f"  Coarse cross-pol shape: {xpol_coarse.shape}, finite: {np.isfinite(xpol_coarse).sum()}")
    
    # 7. Reproject all inputs to target (native) grid
    print(f"\n=== Reprojecting to native grid ===")
    print(f"Target grid: {target_shape}, transform: {target_tf}")
    
    print("Reprojecting TB36...")
    tb36_native = reproject_to_target(tb36, tb36_tf, tb36_crs, target_tf, target_crs, target_shape)
    print(f"  TB36 native: finite = {np.isfinite(tb36_native).sum()}")
    
    print("Reprojecting β...")
    beta_native = reproject_to_target(beta, beta_tf, beta_crs, target_tf, target_crs, target_shape)
    print(f"  β native: finite = {np.isfinite(beta_native).sum()}")
    
    print("Reprojecting Γ...")
    gamma_native = reproject_to_target(gamma, gamma_tf, gamma_crs, target_tf, target_crs, target_shape)
    print(f"  Γ native: finite = {np.isfinite(gamma_native).sum()}")
    
    print("Reprojecting coarse co-pol...")
    copol_coarse_native = reproject_to_target(
        copol_coarse, copol_coarse_tf, copol_coarse_crs, target_tf, target_crs, target_shape
    )
    print(f"  Coarse co-pol native: finite = {np.isfinite(copol_coarse_native).sum()}")
    
    print("Reprojecting coarse cross-pol...")
    xpol_coarse_native = reproject_to_target(
        xpol_coarse, xpol_coarse_tf, xpol_coarse_crs, target_tf, target_crs, target_shape
    )
    print(f"  Coarse cross-pol native: finite = {np.isfinite(xpol_coarse_native).sum()}")
    
    # 8. Apply disaggregation equation
    print(f"\n=== Applying disaggregation equation ===")
    
    # Δσ_pp = σ_pp(M_j) - σ_pp(C)
    delta_copol = copol_native - copol_coarse_native
    
    # Δσ_pq = σ_pq(C) - σ_pq(M_j)  [NOTE: reversed order in equation]
    delta_xpol = xpol_coarse_native - xpol_native
    
    # Calculate correction term
    correction_term = beta_native * (delta_copol + gamma_native * delta_xpol)
    
    # Quality control: limit correction term to physically reasonable range (±50 K)
    print("Applying quality control on correction term...")
    correction_term_qc = np.clip(correction_term, -50.0, 50.0)
    n_correction_clipped = np.sum(np.abs(correction_term_qc - correction_term) > 0.01)
    if n_correction_clipped > 0:
        print(f"  Correction term clipped: {n_correction_clipped} pixels")
        print(f"    Before QC: min={np.nanmin(correction_term):.2f}, max={np.nanmax(correction_term):.2f} K")
        print(f"    After QC: min={np.nanmin(correction_term_qc):.2f}, max={np.nanmax(correction_term_qc):.2f} K")
    
    # TB_fine = TB36 + correction_term
    tb_fine = tb36_native + correction_term_qc
    
    # Final quality control: limit TB to physical range (220-290 K)
    tb_fine_qc = np.clip(tb_fine, 220.0, 290.0)
    n_tb_clipped = np.sum(np.abs(tb_fine_qc - tb_fine) > 0.01)
    if n_tb_clipped > 0:
        print(f"  TB_fine clipped to [220, 290] K: {n_tb_clipped} pixels")
    
    finite_count = np.isfinite(tb_fine_qc).sum()
    print(f"\nTB_fine: shape = {tb_fine_qc.shape}, finite = {finite_count}")
    
    if finite_count > 0:
        valid_tb = tb_fine_qc[np.isfinite(tb_fine_qc)]
        print(f"TB_fine stats: min = {valid_tb.min():.2f} K, max = {valid_tb.max():.2f} K, mean = {valid_tb.mean():.2f} K")
    
    # 9. Save output
    out_path = out_dir / f"TB_fine_{date}_{pol}_native.npz"
    print(f"\n=== Saving output to {out_path} ===")
    
    np.savez_compressed(
        out_path,
        TB_fine=tb_fine_qc.astype(np.float32),
        transform=np.array([target_tf.a, target_tf.b, target_tf.c,
                           target_tf.d, target_tf.e, target_tf.f], dtype=np.float64),
        crs_wkt=target_crs.to_wkt(),
        height=np.int32(target_shape[0]),
        width=np.int32(target_shape[1]),
        date=date,
        pol=pol,
        # Include components for debugging
        TB36_native=tb36_native.astype(np.float32),
        beta_native=beta_native.astype(np.float32),
        gamma_native=gamma_native.astype(np.float32),
        delta_copol=delta_copol.astype(np.float32),
        delta_xpol=delta_xpol.astype(np.float32),
        correction_term=correction_term_qc.astype(np.float32),
    )
    
    print("[OK] Step 5 complete!")


if __name__ == "__main__":
    main()
