#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step3_smap_multidate.py — Estimate coarse-scale active–passive parameter β(C)
from multi-temporal SMAP radiometer–radar anomalies over Boulder.

This script:
    - Reads multiple NPZ files produced by step2_smap_boulder.py
      (each corresponding to one date / overpass).
    - Builds a time stack of TB and co-pol backscatter σ0 on the common 36 km grid.
    - Computes temporal anomalies and fits, per pixel, the linear model:

          ΔTB(C, t_k) ≈ a(C) Δσ_pp(C, t_k)

      (no cross-pol term available for SMAP radar here).

    - Outputs β(C) = a(C) and quality diagnostics.

Inputs (Step 2 outputs):
    A set of NPZ files, typically in data/interim/, each containing at least:
        - TBc_2d   : 2D array of brightness temperature [K]
        - S_pp_dB  : 2D array of co-pol backscatter [dB]
        - crs_wkt, transform, height, width, meta

    Files are selected via a glob pattern, e.g.:
        data/interim/step2_smap_boulder_*.npz

Outputs:
    data/interim/step3_smap_multidate_inversion_params.npz with:
        - beta_K_per_dB   : β(C) [K/dB]
        - a_coef          : a(C) [K/dB]  (same as beta)
        - n_samples       : number of time samples used per pixel
        - r2              : coefficient of determination per pixel
        - crs_wkt, transform, height, width
        - files           : list of input filenames (for traceability)
        - meta            : text metadata about the run

Usage example:
    python step3_smap_multidate.py \
        --stack-pattern "data/interim/step2_smap_boulder_*.npz" \
        --out "data/interim/step3_smap_multidate_inversion_params.npz"
"""

import argparse
import glob
import warnings
from pathlib import Path

import numpy as np

# -----------------------------
# Project paths
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
INTERIM.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Loading utilities
# -----------------------------
def load_time_stack(pattern: str):
    """Load multi-temporal stacks TB and S_pp from a set of Step 2 NPZ files.

    Args:
        pattern: Glob pattern for NPZ files
                 (e.g., 'data/interim/step2_smap_boulder_*.npz').

    Returns:
        TB_stack   : np.ndarray (T, H, W) of brightness temperature [K]
        Spp_stack  : np.ndarray (T, H, W) of co-pol backscatter [dB]
        meta       : dict with grid metadata and list of files
    """
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No NPZ files found for pattern: {pattern}")

    TB_list = []
    Spp_list = []
    first_meta = {}

    for i, fp in enumerate(files):
        npz = np.load(fp, allow_pickle=True)
        if i == 0:
            if "TBc_2d" not in npz or "S_pp_dB" not in npz:
                raise KeyError(
                    f"File {fp} is missing required keys 'TBc_2d' and 'S_pp_dB'. "
                    "Make sure it was produced by step2_smap_boulder.py."
                )
            h, w = npz["TBc_2d"].shape
            first_meta["height"] = int(npz.get("height", h))
            first_meta["width"] = int(npz.get("width", w))
            first_meta["crs_wkt"] = str(npz.get("crs_wkt", ""))
            first_meta["transform"] = np.array(
                npz.get("transform", np.eye(3)), dtype=float
            )

        TB_list.append(np.asarray(npz["TBc_2d"], dtype=np.float32))
        Spp_list.append(np.asarray(npz["S_pp_dB"], dtype=np.float32))

    TB_stack = np.stack(TB_list, axis=0)   # (T, H, W)
    Spp_stack = np.stack(Spp_list, axis=0) # (T, H, W)

    meta = {
        "height": first_meta["height"],
        "width": first_meta["width"],
        "crs_wkt": first_meta["crs_wkt"],
        "transform": first_meta["transform"],
        "files": files,
    }
    return TB_stack, Spp_stack, meta


# -----------------------------
# Core utilities
# -----------------------------
def compute_anomalies(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute temporal mean and anomalies along axis 0, ignoring NaNs.

    Args:
        stack: Array (T, H, W).

    Returns:
        mean_2d : 2D mean over time (H, W)
        anom    : anomalies stack (T, H, W), stack - mean_2d
    """
    mean_2d = np.nanmean(stack, axis=0)
    anom = stack - mean_2d[None, :, :]
    return mean_2d, anom


def estimate_beta_per_pixel(
    dTB: np.ndarray,
    dSpp: np.ndarray,
    min_samples_1d: int = 6,
    eps_a: float = 1e-3,
):
    """Estimate a(C) and β(C) from ΔTB ~ a(C) Δσ_pp per pixel (1D regression only).

    Args:
        dTB:  (T, H, W) anomalies of TB.
        dSpp:(T, H, W) anomalies of σ_pp [dB].
        min_samples_1d: Minimum number of valid samples for regression.
        eps_a: Threshold below which a(C) is considered too small.

    Returns:
        beta       : β(C) = a(C) [K/dB]
        a_coef     : a(C) [K/dB]
        n_samples  : number of samples used per pixel
        r2         : coefficient of determination per pixel
    """
    T, H, W = dTB.shape
    n_pix = H * W

    # Flatten spatial dimensions
    y_all = dTB.reshape(T, n_pix)
    x1_all = dSpp.reshape(T, n_pix)

    a_flat = np.full(n_pix, np.nan, dtype=np.float32)
    beta_flat = np.full(n_pix, np.nan, dtype=np.float32)
    n_flat = np.zeros(n_pix, dtype=np.int16)
    r2_flat = np.full(n_pix, np.nan, dtype=np.float32)

    for i in range(n_pix):
        y = y_all[:, i]
        x = x1_all[:, i]

        mask = np.isfinite(y) & np.isfinite(x)
        n = int(mask.sum())
        if n < min_samples_1d:
            # Not enough temporal samples → leave NaNs
            continue

        yy = y[mask]
        xx = x[mask]

        denom = float(np.nansum(xx * xx))
        if denom == 0.0:
            continue

        a_val = float(np.nansum(xx * yy) / denom)
        n_used = n

        # R^2 for 1D fit
        y_hat = a_val * xx
        ss_res = float(np.nansum((yy - y_hat) ** 2))
        ss_tot = float(np.nansum((yy - np.nanmean(yy)) ** 2))
        r2_val = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        a_flat[i] = a_val
        n_flat[i] = n_used
        r2_flat[i] = r2_val

        # β = a, but drop values that are too small
        if abs(a_val) > eps_a:
            beta_flat[i] = a_val
        else:
            beta_flat[i] = np.nan

    beta = beta_flat.reshape(H, W)
    a_coef = a_flat.reshape(H, W)
    n_samples = n_flat.reshape(H, W)
    r2 = r2_flat.reshape(H, W)

    return beta, a_coef, n_samples, r2


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 3 (multi-date SMAP): estimate β(C) from SMAP radiometer–radar "
            "anomalies using multiple step2_smap_boulder NPZ files."
        )
    )
    parser.add_argument(
        "--stack-pattern",
        type=str,
        default=str(INTERIM / "step2_smap_boulder_*.npz"),
        help=(
            "Glob pattern for Step 2 NPZ files (default: "
            "data/interim/step2_smap_boulder_*.npz)."
        ),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(INTERIM / "step3_smap_multidate_inversion_params.npz"),
        help="Output NPZ path for β maps.",
    )
    parser.add_argument(
        "--min-samples-1d",
        type=int,
        default=6,
        help="Minimum valid time samples for 1D regression (TB ~ Δσ_pp).",
    )
    parser.add_argument(
        "--eps-a",
        type=float,
        default=1e-3,
        help="Threshold |a| > eps_a below which β is set to NaN.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[INFO] Loading time stack from pattern: {args.stack_pattern}")
    TB_stack, Spp_stack, meta = load_time_stack(args.stack_pattern)

    print(f"[INFO] Stack shapes: TB={TB_stack.shape}, Spp={Spp_stack.shape}")
    T, H, W = TB_stack.shape
    if T < args.min_samples_1d:
        warnings.warn(
            f"Only T={T} time samples found. This is less than min_samples_1d={args.min_samples_1d}. "
            "Most pixels may end up with NaN β due to insufficient temporal sampling."
        )

    # Compute anomalies
    _, dTB = compute_anomalies(TB_stack)
    _, dSpp = compute_anomalies(Spp_stack)

    print("[INFO] Estimating per-pixel coefficients a(C), β(C)...")
    beta, a_coef, n_samples, r2 = estimate_beta_per_pixel(
        dTB,
        dSpp,
        min_samples_1d=args.min_samples_1d,
        eps_a=args.eps_a,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        beta_K_per_dB=beta,
        a_coef=a_coef,
        n_samples=n_samples,
        r2=r2,
        crs_wkt=meta["crs_wkt"],
        transform=meta["transform"],
        height=np.int32(meta["height"]),
        width=np.int32(meta["width"]),
        files=np.array(meta["files"], dtype=object),
        meta=np.array(
            [
                "β(C) estimated from multi-temporal SMAP ΔTB and Δσ_pp via 1D least squares",
                f"min_samples_1d={args.min_samples_1d}",
                f"eps_a={args.eps_a}",
                f"stack_pattern={args.stack_pattern}",
            ],
            dtype=object,
        ),
    )
    print(f"[OK] Saved multi-date inversion parameters to {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        warnings.warn(f"[ERROR] {e}")
        raise


"""python step3_smap_multidate.py --stack-pattern "data/interim/step2_smap_boulder_*.npz" """