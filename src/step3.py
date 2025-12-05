#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step3_inversion.py — Estimate coarse-scale active–passive parameters β(C) and Γ(C)
from multi-temporal SMAP–NISAR anomalies (Step 3 of the workflow).

Inputs (Step 2 outputs):
    A set of NPZ files, typically in data/interim/, each containing at least:
        - TBc_2d   : 2D array of brightness temperature [K]
        - S_pp_dB  : 2D array of co-pol backscatter [dB]
        - S_pq_dB  : 2D array of cross-pol backscatter [dB] (may be all-NaN)
        - crs_wkt, transform, height, width

    Files are selected via a glob pattern, e.g.:
        data/interim/step2_*.npz

Processing:
    For each grid cell C (template pixel), and a time series {t_k}:
        ΔTB_p(C, t_k)  = TB_p(C, t_k)  - mean_t TB_p(C)
        Δσ_pp(C, t_k)  = σ_pp(C, t_k)  - mean_t σ_pp(C)
        z_pq(C, t_k)   = mean_t σ_pq(C) - σ_pq(C, t_k)

    A linear model is fitted per pixel:
        ΔTB_p(C, t_k) ≈ a(C) Δσ_pp(C, t_k) + b(C) z_pq(C, t_k)

    Then:
        β(C)   = a(C)
        Γ(C)   = b(C) / a(C)   if |a(C)| > eps, else NaN

Outputs:
    data/interim/step3_inversion_params.npz with:
        - beta_K_per_dB   : β(C) [K/dB]
        - gamma_unitless  : Γ(C) [–]
        - a_coef          : a(C) [K/dB]
        - b_coef          : b(C) [K/dB]
        - n_samples       : number of time samples used per pixel
        - r2              : coefficient of determination per pixel
        - crs_wkt, transform, height, width
        - files           : list of input filenames (for traceability)

Usage example:
    python step3_inversion.py --stack-pattern "data/interim/step2_*.npz" \
                              --out "data/interim/step3_inversion_params.npz"
"""

import argparse
import glob
import warnings
from pathlib import Path

import numpy as np

# -----------------------------
# Project paths (same style as step1/step2)
# -----------------------------
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
INTERIM.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Loading utilities
# -----------------------------
def load_time_stack(pattern: str):
    """Load multi-temporal stacks TB, S_pp and S_pq from a set of NPZ files.

    Args:
        pattern: Glob pattern for NPZ files (e.g., 'data/interim/step2_*.npz').

    Returns:
        TB_stack   : np.ndarray (T, H, W) of brightness temperature [K]
        Spp_stack  : np.ndarray (T, H, W) of co-pol backscatter [dB]
        Spq_stack  : np.ndarray (T, H, W) of cross-pol backscatter [dB]
        meta       : dict with grid metadata and list of files
    """
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No NPZ files found for pattern: {pattern}")

    TB_list = []
    Spp_list = []
    Spq_list = []
    first_meta = {}

    for i, fp in enumerate(files):
        npz = np.load(fp, allow_pickle=True)
        if i == 0:
            # basic checks and metadata
            for key in ("TBc_2d", "S_pp_dB", "S_pq_dB"):
                if key not in npz:
                    raise KeyError(f"Required key '{key}' missing in {fp}")
            h, w = npz["S_pp_dB"].shape
            first_meta["height"] = int(npz.get("height", h))
            first_meta["width"] = int(npz.get("width", w))
            first_meta["crs_wkt"] = str(npz.get("crs_wkt", ""))
            first_meta["transform"] = np.array(
                npz.get("transform", np.eye(3)), dtype=float
            )

        TB_list.append(np.asarray(npz["TBc_2d"], dtype=np.float32))
        Spp_list.append(np.asarray(npz["S_pp_dB"], dtype=np.float32))
        Spq_list.append(np.asarray(npz["S_pq_dB"], dtype=np.float32))

    TB_stack = np.stack(TB_list, axis=0)   # (T, H, W)
    Spp_stack = np.stack(Spp_list, axis=0) # (T, H, W)
    Spq_stack = np.stack(Spq_list, axis=0) # (T, H, W)

    meta = {
        "height": first_meta["height"],
        "width": first_meta["width"],
        "crs_wkt": first_meta["crs_wkt"],
        "transform": first_meta["transform"],
        "files": files,
    }
    return TB_stack, Spp_stack, Spq_stack, meta


# -----------------------------
# Core inversion
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


def estimate_coefficients_per_pixel(
    dTB: np.ndarray,
    dSpp: np.ndarray,
    zpq: np.ndarray,
    min_samples_2d: int = 10,
    min_samples_1d: int = 6,
    ridge_lambda: float = 0.0,
    eps_a: float = 1e-3,
):
    """Estimate a(C), b(C) and derived β(C), Γ(C) per pixel using least squares.

    Args:
        dTB:  (T, H, W) anomalies of TB.
        dSpp:(T, H, W) anomalies of σ_pp [dB].
        zpq: (T, H, W) z_pq = mean(σ_pq) - σ_pq(t) [dB].
        min_samples_2d: Minimum number of valid samples for 2D regression.
        min_samples_1d: Minimum number of valid samples for 1D regression.
        ridge_lambda: L2 regularisation parameter (added to diagonal).
        eps_a: Threshold below which a(C) is considered too small for Γ.

    Returns:
        beta       : β(C) = a(C) [K/dB]
        gamma      : Γ(C) [–]
        a_coef     : a(C) [K/dB]
        b_coef     : b(C) [K/dB]
        n_samples  : number of samples used per pixel
        r2         : coefficient of determination per pixel
    """
    T, H, W = dTB.shape
    n_pix = H * W

    # Flatten spatial dims
    y_all = dTB.reshape(T, n_pix)
    x1_all = dSpp.reshape(T, n_pix)
    x2_all = zpq.reshape(T, n_pix)

    a_flat = np.full(n_pix, np.nan, dtype=np.float32)
    b_flat = np.full(n_pix, np.nan, dtype=np.float32)
    beta_flat = np.full(n_pix, np.nan, dtype=np.float32)
    gamma_flat = np.full(n_pix, np.nan, dtype=np.float32)
    n_flat = np.zeros(n_pix, dtype=np.int16)
    r2_flat = np.full(n_pix, np.nan, dtype=np.float32)

    all_spq_nan = np.all(~np.isfinite(x2_all))

    for i in range(n_pix):
        y = y_all[:, i]
        x1 = x1_all[:, i]
        x2 = x2_all[:, i]

        # 2D regression if cross-pol reasonably available
        if not all_spq_nan:
            mask_2d = np.isfinite(y) & np.isfinite(x1) & np.isfinite(x2)
            n2 = int(mask_2d.sum())
        else:
            n2 = 0

        if n2 >= min_samples_2d:
            # 2D regression: [x1, x2]
            yy = y[mask_2d]
            X = np.stack([x1[mask_2d], x2[mask_2d]], axis=1)  # (n2, 2)

            # Ridge regularization: (X^T X + λI)^{-1} X^T y
            XtX = X.T @ X
            if ridge_lambda > 0.0:
                XtX += ridge_lambda * np.eye(2, dtype=np.float64)
            Xty = X.T @ yy
            try:
                coef = np.linalg.solve(XtX, Xty)
            except np.linalg.LinAlgError:
                # Fallback to lstsq
                coef, *_ = np.linalg.lstsq(X, yy, rcond=None)

            a_val = float(coef[0])
            b_val = float(coef[1])
            n_used = n2

            # R^2
            y_hat = X @ coef
            ss_res = float(np.nansum((yy - y_hat) ** 2))
            ss_tot = float(np.nansum((yy - np.nanmean(yy)) ** 2))
            r2_val = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        else:
            # 1D regression only with Δσ_pp (no cross-pol)
            mask_1d = np.isfinite(y) & np.isfinite(x1)
            n1 = int(mask_1d.sum())
            if n1 < min_samples_1d:
                continue  # leave as NaN
            yy = y[mask_1d]
            x = x1[mask_1d]

            # slope a = Σ(x y) / Σ(x^2)
            denom = float(np.nansum(x * x))
            if denom == 0.0:
                continue
            a_val = float(np.nansum(x * yy) / denom)
            b_val = 0.0  # no cross-pol term
            n_used = n1

            # R^2 for 1D fit
            y_hat = a_val * x
            ss_res = float(np.nansum((yy - y_hat) ** 2))
            ss_tot = float(np.nansum((yy - np.nanmean(yy)) ** 2))
            r2_val = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        a_flat[i] = a_val
        b_flat[i] = b_val
        n_flat[i] = n_used
        r2_flat[i] = r2_val

        # β = a
        beta_flat[i] = a_val

        # Γ = b/a if |a| > eps_a
        if abs(a_val) > eps_a:
            gamma_flat[i] = b_val / a_val
        else:
            gamma_flat[i] = np.nan

        # optional: progress print every ~100k pixels
        # if i % 100000 == 0 and i > 0:
        #     print(f"[INFO] Processed {i}/{n_pix} pixels")

    # Reshape back to (H, W)
    beta = beta_flat.reshape(H, W)
    gamma = gamma_flat.reshape(H, W)
    a_coef = a_flat.reshape(H, W)
    b_coef = b_flat.reshape(H, W)
    n_samples = n_flat.reshape(H, W)
    r2 = r2_flat.reshape(H, W)

    if all_spq_nan:
        warnings.warn(
            "Cross-pol (S_pq_dB) appears to be all-NaN in the stack. "
            "Γ(C) cannot be reliably estimated; only β(C)=a(C) is meaningful."
        )

    return beta, gamma, a_coef, b_coef, n_samples, r2


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the inversion step."""
    p = argparse.ArgumentParser(
        description=(
            "Estimate active–passive inversion parameters β(C) and Γ(C) from "
            "multi-temporal SMAP–NISAR anomalies (Step 3)."
        )
    )
    p.add_argument(
        "--stack-pattern",
        type=str,
        default=str(INTERIM / "step2_*.npz"),
        help="Glob pattern for Step 2 NPZ files (default: data/interim/step2_*.npz).",
    )
    p.add_argument(
        "--out",
        type=str,
        default=str(INTERIM / "step3_inversion_params.npz"),
        help="Output NPZ path for β and Γ maps.",
    )
    p.add_argument(
        "--min-samples-2d",
        type=int,
        default=10,
        help="Minimum valid time samples for 2D regression (TB ~ Δσ_pp + z_pq).",
    )
    p.add_argument(
        "--min-samples-1d",
        type=int,
        default=6,
        help="Minimum valid time samples for 1D regression (TB ~ Δσ_pp only).",
    )
    p.add_argument(
        "--ridge-lambda",
        type=float,
        default=0.0,
        help="Ridge regularisation parameter λ (default 0.0 = no regularisation).",
    )
    p.add_argument(
        "--eps-a",
        type=float,
        default=1e-3,
        help="Threshold |a| > eps_a to compute Γ = b/a.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[INFO] Loading time stack from pattern: {args.stack_pattern}")
    TB_stack, Spp_stack, Spq_stack, meta = load_time_stack(args.stack_pattern)

    print(f"[INFO] Stack shapes: TB={TB_stack.shape}, Spp={Spp_stack.shape}, Spq={Spq_stack.shape}")

    # Compute anomalies
    _, dTB = compute_anomalies(TB_stack)
    _, dSpp = compute_anomalies(Spp_stack)
    mean_Spq, _ = compute_anomalies(Spq_stack)
    zpq = mean_Spq[None, :, :] - Spq_stack  # z_pq(C, t) = mean(σ_pq) - σ_pq(t)

    print("[INFO] Estimating per-pixel coefficients a(C), b(C), β(C), Γ(C)...")
    beta, gamma, a_coef, b_coef, n_samples, r2 = estimate_coefficients_per_pixel(
        dTB,
        dSpp,
        zpq,
        min_samples_2d=args.min_samples_2d,
        min_samples_1d=args.min_samples_1d,
        ridge_lambda=args.ridge_lambda,
        eps_a=args.eps_a,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        beta_K_per_dB=beta,
        gamma_unitless=gamma,
        a_coef=a_coef,
        b_coef=b_coef,
        n_samples=n_samples,
        r2=r2,
        crs_wkt=meta["crs_wkt"],
        transform=meta["transform"],
        height=np.int32(meta["height"]),
        width=np.int32(meta["width"]),
        files=np.array(meta["files"], dtype=object),
        meta=np.array(
            [
                "β(C) and Γ(C) estimated from ΔTB, Δσ_pp, z_pq via least squares",
                f"min_samples_2d={args.min_samples_2d}",
                f"min_samples_1d={args.min_samples_1d}",
                f"ridge_lambda={args.ridge_lambda}",
                f"eps_a={args.eps_a}",
            ],
            dtype=object,
        ),
    )
    print(f"[OK] Saved inversion parameters to {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        warnings.warn(f"[ERROR] {e}")
        raise
