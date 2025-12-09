#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step3_smap_multidate_1d.py — Estimate coarse-scale active–passive parameter
β(C) from multi-temporal SMAP radiometer–radar anomalies.

Model:
    ΔTB(C, t_k) ≈ β(C) Δσ_pp(C, t_k)

Inputs:
    A set of NPZ files (Step 2 outputs) with at least:
        - TBc_2d   : 2D array of brightness temperature [K]
        - S_pp_dB  : 2D array of co-pol backscatter [dB]
        - crs_wkt, transform, height, width (grid metadata)

Outputs:
    data/interim/step3_smap_multidate_1d_inversion_params.npz with:
        - beta_K_per_dB   : β(C) [K/dB]
        - n_samples       : number of time samples used per pixel
        - r2              : coefficient of determination per pixel
        - crs_wkt, transform, height, width
        - files           : list of input filenames
        - meta            : text metadata
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
    """Load multi-temporal stacks TB, S_pp from Step 2 NPZ files.

    Args:
        pattern: Glob pattern for NPZ files.

    Returns:
        TB_stack   : np.ndarray (T, H, W)
        Spp_stack  : np.ndarray (T, H, W)
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
            # Check required keys
            for key in ("TBc_2d", "S_pp_dB"):
                if key not in npz:
                    raise KeyError(
                        f"File {fp} is missing required key '{key}'. "
                        "Make sure it was produced by the new Step 2 "
                        "with 'TBc_2d' and 'S_pp_dB'."
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
    ridge_lambda: float = 0.0,
    eps_beta: float = 1e-3,
):
    """Estimate β(C) from ΔTB ~ β Δσ_pp per pixel (1D regression).

    Args:
        dTB      : (T, H, W) anomalies of TB.
        dSpp     : (T, H, W) anomalies of σ_pp [dB].
        min_samples_1d: Minimum number of valid samples for 1D regression.
        ridge_lambda : Ridge regularization parameter (λ >= 0). For 1D,
                       this effectively adds λ to ∑x².
        eps_beta : Threshold below which |β| is set to NaN.

    Returns:
        beta       : β(C) [K/dB]
        n_samples  : number of samples used per pixel
        r2         : coefficient of determination per pixel
    """
    T, H, W = dTB.shape
    n_pix = H * W

    # Flatten spatial dims
    y_all = dTB.reshape(T, n_pix)   # (T, P)
    x_all = dSpp.reshape(T, n_pix)  # (T, P)

    beta_flat = np.full(n_pix, np.nan, dtype=np.float32)
    n_flat = np.zeros(n_pix, dtype=np.int16)
    r2_flat = np.full(n_pix, np.nan, dtype=np.float32)

    for i in range(n_pix):
        y = y_all[:, i]
        x = x_all[:, i]

        mask = np.isfinite(y) & np.isfinite(x)
        n = int(mask.sum())
        if n < min_samples_1d:
            continue

        yy = y[mask]
        xx = x[mask]

        # Normal equations for 1D: (∑x² + λ) β = ∑ x y
        sum_xx = float(np.sum(xx * xx))
        sum_xy = float(np.sum(xx * yy))

        denom = sum_xx + ridge_lambda
        if denom <= 0:
            continue

        beta_val = sum_xy / denom

        # Predicted TB and R²
        y_hat = beta_val * xx
        ss_res = float(np.sum((yy - y_hat) ** 2))
        ss_tot = float(np.sum((yy - np.mean(yy)) ** 2))
        r2_val = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        n_flat[i] = n
        if abs(beta_val) > eps_beta:
            beta_flat[i] = beta_val
        else:
            beta_flat[i] = np.nan

        r2_flat[i] = r2_val

    beta = beta_flat.reshape(H, W)
    n_samples = n_flat.reshape(H, W)
    r2 = r2_flat.reshape(H, W)

    return beta, n_samples, r2


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 3 (multi-date SMAP, 1D): estimate β(C) from "
            "SMAP radiometer–radar anomalies using co-pol stacks only."
        )
    )
    parser.add_argument(
        "--stack-pattern",
        type=str,
        default=str(INTERIM / "step2_smap_l1c_*.npz"),
        help=(
            "Glob pattern for Step 2 NPZ files (default: "
            "data/interim/step2_smap_l1c_*.npz)."
        ),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(INTERIM / "step3_smap_multidate_1d_inversion_params.npz"),
        help="Output NPZ path for β maps.",
    )
    parser.add_argument(
        "--min-samples-1d",
        type=int,
        default=6,
        help="Minimum valid time samples for 1D regression (ΔTB ~ βΔσ_pp).",
    )
    parser.add_argument(
        "--ridge-lambda",
        type=float,
        default=0.0,
        help="Ridge regularization parameter λ (default 0: ordinary least squares).",
    )
    parser.add_argument(
        "--eps-beta",
        type=float,
        default=1e-3,
        help="Threshold |β| > eps_beta below which β is set to NaN.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[INFO] Loading time stack from pattern: {args.stack_pattern}")
    TB_stack, Spp_stack, meta = load_time_stack(args.stack_pattern)

    print(
        f"[INFO] Stack shapes: TB={TB_stack.shape}, "
        f"Spp={Spp_stack.shape}"
    )
    T, H, W = TB_stack.shape
    if T < args.min_samples_1d:
        warnings.warn(
            f"Only T={T} time samples found. This is less than "
            f"min_samples_1d={args.min_samples_1d}. "
            "Most pixels may end up with NaN β due to insufficient temporal sampling."
        )

    # Compute anomalies
    _, dTB = compute_anomalies(TB_stack)
    _, dSpp = compute_anomalies(Spp_stack)

    print("[INFO] Estimating per-pixel coefficient β(C)...")
    beta, n_samples, r2 = estimate_beta_per_pixel(
        dTB,
        dSpp,
        min_samples_1d=args.min_samples_1d,
        ridge_lambda=args.ridge_lambda,
        eps_beta=args.eps_beta,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        beta_K_per_dB=beta,
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
                f"ridge_lambda={args.ridge_lambda}",
                f"eps_beta={args.eps_beta}",
                f"stack_pattern={args.stack_pattern}",
            ],
            dtype=object,
        ),
    )
    print(f"[OK] Saved multi-date 1D inversion parameters to {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        warnings.warn(f"[ERROR] {e}")
        raise
