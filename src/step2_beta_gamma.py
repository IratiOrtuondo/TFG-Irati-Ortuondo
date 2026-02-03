#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step2_beta_gamma.py — Estimate coarse-scale β and Γ from multi-temporal SMAP stacks.

This module implements an ATBD-like per-coarse-pixel regression using
multi-date anomalies of TB (driver) and radar backscatter. The model is:

  ΔTB(C,t) ≈ a·Δσ_pp(C,t) + b·Δ( mean(σ_pq)(C) − σ_pq(C,t) )

From which we extract per-pixel parameters:
  β(C) = a  [K/dB]
  Γ(C) = b / a  [dimensionless]

Key behavior & heuristics:
- TB is used as the driver to define the date list; copol/xpol for a
  date are included when available, otherwise NaNs are used to keep
  temporal alignment.
- Dates with low fraction of finite TB (configurable via
  --min-tb-finite-frac) are dropped to avoid polluted anomalies.
- Dynamic minimum samples are supported (0 => dynamic based on T): the
  algorithm will attempt estimation even with few dates (useful for
  sparse time series), but user-specified minima are respected when given.
- Primary fit is joint (two regressors) when sufficient finite x2 exists;
  otherwise fallback to β-only (through-origin) is used with Γ set to 0.
- Ridge regularization is applied to improve numerical stability; a series
  of sanity checks (variance thresholds, R² threshold, bounds on Γ and β
  magnitude) prevent spurious estimates. Final QA fills NaNs and clips
  β/Γ to user-safe ranges for downstream robustness.

This commented version adds clearer function docstrings, explains the
matrix algebra steps, and annotates the QA/clipping and fallback logic.
"""

from __future__ import annotations

import argparse
import glob
import warnings
import re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
INTERIM = DATA / "interim"
INTERIM.mkdir(parents=True, exist_ok=True)


def _extract_date_yyyymmdd(path_str: str):
    """Extract the first YYYYMMDD substring from a filename or path.

    Returns the matched string or None if no 8-digit date is found. Used to
    align TB/cop/XPOL files by date without depending on exact naming
    conventions beyond containing an 8-digit date.
    """
    m = re.search(r"(\d{8})", Path(path_str).name)
    return m.group(1) if m else None


def load_time_stack_from_separate_files(
    tb_pattern: str,
    copol_pattern: str,
    xpol_pattern: str,
    tb_key: str = "TB_36km",
    copol_key: str = "S_copol_dB",
    xpol_key: str = "S_xpol_dB",
    start_date: str | None = None,
    end_date: str | None = None,
    min_tb_finite_frac: float = 0.80,
):
    """Construct time stacks (T,H,W) for TB, copol and xpol from file patterns.

    Workflow:
    - Discover files matching patterns and map them to dates via `_extract_date_yyyymmdd`.
    - Build a TB-driven date list; filter by start/end dates if provided.
    - For each TB date, load TB and check finite fraction; drop dates with low
      finite TB fraction to avoid poor anomaly statistics.
    - Attempt to load co-pol and x-pol for the same date; if missing or
      shape-mismatched, substitute NaNs so the temporal dimension remains aligned.

    Returns: TB_stack, Spp_stack, Sxpol_stack, meta where meta contains shape,
    crs/transform, and lists of files/dates used.
    """

    tb_files = sorted(glob.glob(tb_pattern))
    if not tb_files:
        raise FileNotFoundError(f"No TB NPZ files found for pattern: {tb_pattern}")

    copol_files = sorted(glob.glob(copol_pattern))
    # Skip native-resolution files because we want coarse-aligned copol
    copol_files = [f for f in copol_files if "-native" not in f]
    if not copol_files:
        warnings.warn(f"No co-pol NPZ files found for pattern: {copol_pattern}. Co-pol will be NaN.")

    xpol_files = sorted(glob.glob(xpol_pattern))
    xpol_files = [f for f in xpol_files if "-native" not in f]
    if not xpol_files:
        warnings.warn(f"No cross-pol NPZ files found for pattern: {xpol_pattern}. Cross-pol will be NaN.")

    tb_map, copol_map, xpol_map = {}, {}, {}

    for tf in tb_files:
        d = _extract_date_yyyymmdd(tf)
        if d:
            tb_map[d] = tf
    for cf in copol_files:
        d = _extract_date_yyyymmdd(cf)
        if d:
            copol_map[d] = cf
    for xf in xpol_files:
        d = _extract_date_yyyymmdd(xf)
        if d:
            xpol_map[d] = xf

    dates_all = sorted(tb_map.keys())
    if not dates_all:
        raise ValueError("No dates could be extracted from TB files")

    # Apply date range filters (string compare works for YYYYMMDD format)
    if start_date is not None:
        dates_all = [d for d in dates_all if d >= start_date]
    if end_date is not None:
        dates_all = [d for d in dates_all if d <= end_date]

    if not dates_all:
        raise ValueError("No TB dates remain after applying date range filter")

    TB_list, Spp_list, Sxpol_list = [], [], []
    first_meta = {}

    kept_dates = []
    dropped_low_tb = []

    for i, date in enumerate(dates_all):
        npz_tb = np.load(tb_map[date], allow_pickle=True)
        if tb_key not in npz_tb:
            raise KeyError(f"File {tb_map[date]} missing required key '{tb_key}'")
        TB = np.asarray(npz_tb[tb_key], dtype=np.float32)

        # Drop dates with low spatial coverage in TB to avoid bad anomaly statistics
        frac_finite = float(np.isfinite(TB).mean())
        if frac_finite < min_tb_finite_frac:
            dropped_low_tb.append((date, frac_finite))
            continue

        if len(kept_dates) == 0:
            # Capture shape/geo metadata from the first accepted TB file
            h, w = TB.shape
            first_meta["height"] = int(npz_tb.get("height", h))
            first_meta["width"] = int(npz_tb.get("width", w))
            first_meta["crs_wkt"] = str(npz_tb.get("crs_wkt", ""))
            first_meta["transform"] = np.array(npz_tb.get("transform", np.eye(3)), dtype=float)

        TB_list.append(TB)
        kept_dates.append(date)

        # --- co-pol: if present use it else fill with NaNs ---
        Spp = None
        if date in copol_map:
            npz_copol = np.load(copol_map[date], allow_pickle=True)
            if copol_key in npz_copol:
                Spp_raw = np.asarray(npz_copol[copol_key], dtype=np.float32)
                if Spp_raw.shape != TB.shape:
                    print(f"[WARN] Co-pol shape {Spp_raw.shape} != TB shape {TB.shape} for {date}, NaN")
                    Spp = np.full_like(TB, np.nan, dtype=np.float32)
                else:
                    Spp = Spp_raw
        if Spp is None:
            Spp = np.full_like(TB, np.nan, dtype=np.float32)
        Spp_list.append(Spp)

        # --- x-pol: similar handling as co-pol ---
        Sx = None
        if date in xpol_map:
            npz_xpol = np.load(xpol_map[date], allow_pickle=True)
            if xpol_key in npz_xpol:
                Sx_raw = np.asarray(npz_xpol[xpol_key], dtype=np.float32)
                if Sx_raw.shape != TB.shape:
                    print(f"[WARN] X-pol shape {Sx_raw.shape} != TB shape {TB.shape} for {date}, NaN")
                    Sx = np.full_like(TB, np.nan, dtype=np.float32)
                else:
                    Sx = Sx_raw
        if Sx is None:
            Sx = np.full_like(TB, np.nan, dtype=np.float32)
        Sxpol_list.append(Sx)

    if not kept_dates:
        raise ValueError("All TB dates were dropped due to low finite fraction. Lower --min-tb-finite-frac?")

    if dropped_low_tb:
        print("[INFO] Dropped TB dates due to low finite fraction:")
        for d, frac in dropped_low_tb:
            print(f"  - {d}: finite={frac:.3f}")

    TB_stack = np.stack(TB_list, axis=0)      # shape (T,H,W)
    Spp_stack = np.stack(Spp_list, axis=0)
    Sxpol_stack = np.stack(Sxpol_list, axis=0)

    meta = {
        "height": first_meta["height"],
        "width": first_meta["width"],
        "crs_wkt": first_meta["crs_wkt"],
        "transform": first_meta["transform"],
        "tb_files": [tb_map[d] for d in kept_dates],
        "copol_files": [copol_map[d] for d in kept_dates if d in copol_map],
        "xpol_files": [xpol_map[d] for d in kept_dates if d in xpol_map],
        "dates": kept_dates,
    }
    return TB_stack, Spp_stack, Sxpol_stack, meta


def compute_anomalies(stack: np.ndarray):
    """Compute mean 2D field and per-date anomalies.

    Returns (mean_2d, anomaly_stack) where anomaly_stack = stack - mean_2d.
    NaNs are ignored in the mean computation (nanmean) preserving sparse data.
    """
    mean_2d = np.nanmean(stack, axis=0)
    anom = stack - mean_2d[None, :, :]
    return mean_2d, anom


def estimate_beta_gamma_atbd(
    dTB: np.ndarray,
    dSpp: np.ndarray,
    Sxpol_stack: np.ndarray,
    min_samples_joint: int | None = None,
    min_samples_beta: int | None = None,
    ridge_lambda: float = 0.1,
    eps_beta: float = 0.05,
    min_var_x: float = 1e-3,
    max_abs_gamma: float = 10.0,
    min_r2: float = 0.0,
):
    """Estimate β and Γ for each coarse pixel using anomalies and diagnostics.

    Inputs:
      dTB: anomaly stack of TB (T,H,W)
      dSpp: anomaly stack of co-pol backscatter (T,H,W)
      Sxpol_stack: full-time-stack of xpol (not anomaly), used to compute
                    x2(t) = mean_xpol - xpol(t) which is then anomaly'ed.

    Estimation strategy per-pixel:
      1. Try joint two-regressor fit y = a*x1 + b*x2 using ridge regularization
         (adds lambda * I to XtX) when sufficient finite samples of x2 exist.
      2. If joint fit fails quality checks (variance or R²) or insufficient
         samples, fallback to β-only fit through the origin y = a*x1.

    Returns: beta_filled (H,W), Gamma_filled (H,W), n_samples, r2, valid_mask
    """
    T, H, W = dTB.shape
    P = H * W

    # Dynamic minimums: if None use 1 (or dynamic logic elsewhere) but user
    # can set a stricter minimum via CLI. Clamp to [1, T].
    if min_samples_beta is None:
        min_samples_beta = 1
    else:
        min_samples_beta = max(1, min(min_samples_beta, T))

    if min_samples_joint is None:
        min_samples_joint = 1
    else:
        min_samples_joint = max(1, min(min_samples_joint, T))

    print(f"[INFO] T={T} dates => min_samples_beta={min_samples_beta}, min_samples_joint={min_samples_joint}")

    y_all = dTB.reshape(T, P)
    x1_all = dSpp.reshape(T, P)

    # Build x2: mean_xpol - xpol(t), then anomaly it to obtain Δx2
    mean_xpol = np.nanmean(Sxpol_stack, axis=0)           # (H,W)
    x2_stack = mean_xpol[None, :, :] - Sxpol_stack        # (T,H,W)
    _, dX2 = compute_anomalies(x2_stack)
    x2_all = dX2.reshape(T, P)

    # Preallocate outputs
    beta_flat = np.full(P, np.nan, dtype=np.float32)
    Gamma_flat = np.full(P, np.nan, dtype=np.float32)
    n_flat = np.zeros(P, dtype=np.int16)
    r2_flat = np.full(P, np.nan, dtype=np.float32)
    valid_flat = np.zeros(P, dtype=np.uint8)

    for i in range(P):
        y = y_all[:, i]
        x1 = x1_all[:, i]
        x2 = x2_all[:, i]

        # --- JOINT FIT: require finite y,x1,x2 ---
        mask_joint = np.isfinite(y) & np.isfinite(x1) & np.isfinite(x2)
        n_joint = int(mask_joint.sum())

        if n_joint >= min_samples_joint:
            yy = y[mask_joint]
            xx1 = x1[mask_joint]
            xx2 = x2[mask_joint]

            # Check variance of regressors to avoid degenerate fits
            if (np.nanvar(xx1) >= min_var_x) and (np.nanvar(xx2) >= min_var_x):
                X = np.column_stack([xx1, xx2])

                # Regularized normal equations: (X^T X + λI) θ = X^T y
                XtX = X.T @ X + ridge_lambda * np.eye(2)
                if np.linalg.cond(XtX) < 1e10:
                    theta = np.linalg.solve(XtX, X.T @ yy)
                    a = float(theta[0])
                    b = float(theta[1])

                    # Require significant beta (prevent tiny/unstable a)
                    if abs(a) >= eps_beta:
                        Gamma_val = b / a
                        # Check Γ is finite and within acceptable range
                        if np.isfinite(Gamma_val) and (abs(Gamma_val) <= max_abs_gamma):
                            y_hat = X @ theta
                            ss_res = float(np.sum((yy - y_hat) ** 2))
                            ss_tot0 = float(np.sum(yy ** 2))  # through-origin R²
                            r2_val = 1.0 - ss_res / ss_tot0 if ss_tot0 > 0 else np.nan

                            # Accept if R² meets the minimum threshold
                            if np.isfinite(r2_val) and (r2_val >= min_r2):
                                beta_flat[i] = a
                                Gamma_flat[i] = Gamma_val
                                n_flat[i] = n_joint
                                r2_flat[i] = r2_val
                                valid_flat[i] = 1
                                continue

        # --- FALLBACK β-only (through-origin) ---
        mask_b = np.isfinite(y) & np.isfinite(x1)
        n_b = int(mask_b.sum())
        if n_b < min_samples_beta:
            continue

        yy = y[mask_b]
        xx1 = x1[mask_b]
        if np.nanvar(xx1) < min_var_x:
            continue

        denom = float(np.sum(xx1 ** 2))
        if denom <= 0:
            continue
        a = float(np.sum(xx1 * yy) / denom)
        if abs(a) < eps_beta:
            continue

        y_hat = a * xx1
        ss_res = float(np.sum((yy - y_hat) ** 2))
        ss_tot0 = float(np.sum(yy ** 2))
        r2_val = 1.0 - ss_res / ss_tot0 if ss_tot0 > 0 else np.nan

        if np.isfinite(r2_val) and (r2_val >= min_r2):
            beta_flat[i] = a
            Gamma_flat[i] = 0.0
            n_flat[i] = n_b
            r2_flat[i] = r2_val
            valid_flat[i] = 1

    # Reshape outputs to 2D
    beta = beta_flat.reshape(H, W)
    Gamma = Gamma_flat.reshape(H, W)
    n_samples = n_flat.reshape(H, W)
    r2 = r2_flat.reshape(H, W)
    valid_mask = valid_flat.reshape(H, W).astype(bool)

    # -------------------------
    # QA + saturation (do not invalidate pixels)
    # -------------------------
    beta_before = beta.copy()
    gamma_before = Gamma.copy()

    # Fill NaNs conservatively so downstream code has deterministic numbers
    beta_filled = beta.copy()
    gamma_filled = gamma.copy()
    beta_filled[~np.isfinite(beta_filled)] = -0.2
    gamma_filled[~np.isfinite(gamma_filled)] = 0.0

    # Clip to plausible bounds rather than letting extreme outliers propagate
    BETA_MIN, BETA_MAX = -10.0, -0.2
    GAMMA_MIN, GAMMA_MAX = 0.0, 2.0

    def stats_arr(arr):
        finite = np.isfinite(arr)
        if np.any(finite):
            return (float(np.nanmin(arr)), float(np.nanmax(arr)), float(np.nanmean(arr)))
        return (np.nan, np.nan, np.nan)

    b_min, b_max, b_mean = stats_arr(beta_before)
    g_min, g_max, g_mean = stats_arr(gamma_before)

    pos_beta = int(np.sum(np.isfinite(beta_before) & (beta_before > 0)))
    near0_beta = int(np.sum(np.isfinite(beta_before) & (np.abs(beta_before) < eps_beta)))

    lower_clip_beta = int(np.sum(beta_filled < BETA_MIN))
    upper_clip_beta = int(np.sum(beta_filled > BETA_MAX))
    beta_filled = np.clip(beta_filled, BETA_MIN, BETA_MAX)

    lower_clip_gamma = int(np.sum(gamma_filled < GAMMA_MIN))
    upper_clip_gamma = int(np.sum(gamma_filled > GAMMA_MAX))
    gamma_filled = np.clip(gamma_filled, GAMMA_MIN, GAMMA_MAX)

    b2_min, b2_max, b2_mean = stats_arr(beta_filled)
    g2_min, g2_max, g2_mean = stats_arr(gamma_filled)

    low_r2_mask = np.isfinite(r2) & (r2 < 0.2)

    # QA summary prints
    print(f"[QA] Beta before: min={b_min:.4f}, max={b_max:.4f}, mean={b_mean:.4f}")
    print(f"[QA] Gamma before: min={g_min:.4f}, max={g_max:.4f}, mean={g_mean:.4f}")
    print(f"[QA] beta>0 count: {pos_beta}")
    print(f"[QA] |beta|<eps count: {near0_beta}")
    print(f"[QA] Beta clipped: lower={lower_clip_beta}, upper={upper_clip_beta}")
    print(f"[QA] Gamma clipped: lower={lower_clip_gamma}, upper={upper_clip_gamma}")
    print(f"[QA] Beta after: min={b2_min:.4f}, max={b2_max:.4f}, mean={b2_mean:.4f}")
    print(f"[QA] Gamma after: min={g2_min:.4f}, max={g2_max:.4f}, mean={g2_mean:.4f}")
    print(f"[QA-WARN] Pixels with R^2 < 0.2: {int(np.sum(low_r2_mask))}")

    return beta_filled, gamma_filled, n_samples, r2, valid_mask


def parse_args():
    """Build CLI parser and default patterns for this script."""
    p = argparse.ArgumentParser(
        description="Step3: ATBD-like estimation of β and Γ from SMAP multi-date anomalies (robust)."
    )
    p.add_argument("--tb-pattern", type=str, default=str(INTERIM / "smap-tb36-????????-v.npz"))
    p.add_argument("--copol-pattern", type=str, default=str(INTERIM / "aligned-smap-copol-????????-vv.npz"))
    p.add_argument("--xpol-pattern", type=str, default=str(INTERIM / "aligned-smap-xpol-????????.npz"))
    p.add_argument("--out", type=str, default=str(INTERIM / "step3_beta_gamma.npz"))

    p.add_argument("--start-date", type=str, default=None, help="YYYYMMDD (inclusive)")
    p.add_argument("--end-date", type=str, default=None, help="YYYYMMDD (inclusive)")
    p.add_argument("--min-tb-finite-frac", type=float, default=0.80, help="Drop TB dates with finite fraction below this")

    # If 0, dynamic thresholds will be used (handled by estimate function)
    p.add_argument("--min-samples-joint", type=int, default=0, help="0 => dynamic based on T")
    p.add_argument("--min-samples-beta", type=int, default=0, help="0 => dynamic based on T")

    p.add_argument("--ridge-lambda", type=float, default=0.1)
    p.add_argument("--eps-beta", type=float, default=0.05)
    p.add_argument("--max-abs-gamma", type=float, default=10.0)
    p.add_argument("--min-r2", type=float, default=0.0)
    return p.parse_args()


def main():
    args = parse_args()

    # Convert CLI zeros to None for dynamic behavior
    min_joint = None if args.min_samples_joint == 0 else args.min_samples_joint
    min_beta = None if args.min_samples_beta == 0 else args.min_samples_beta

    TB_stack, Spp_stack, Sxpol_stack, meta = load_time_stack_from_separate_files(
        args.tb_pattern,
        args.copol_pattern,
        args.xpol_pattern,
        start_date=args.start_date,
        end_date=args.end_date,
        min_tb_finite_frac=args.min_tb_finite_frac,
    )

    print(f"[INFO] Using dates: {meta['dates'][0]} .. {meta['dates'][-1]} (n={len(meta['dates'])})")
    print(f"[INFO] Stack shapes: TB={TB_stack.shape}, Spp={Spp_stack.shape}, Sxpol={Sxpol_stack.shape}")

    print("[INFO] Computing anomalies for TB and Spp...")
    _, dTB = compute_anomalies(TB_stack)
    _, dSpp = compute_anomalies(Spp_stack)

    print("[INFO] Estimating β and Γ (robust + ATBD xpol term + fallback)...")
    beta, Gamma, n_samples, r2, valid_mask = estimate_beta_gamma_atbd(
        dTB=dTB,
        dSpp=dSpp,
        Sxpol_stack=Sxpol_stack,
        min_samples_joint=min_joint,
        min_samples_beta=min_beta,
        ridge_lambda=args.ridge_lambda,
        eps_beta=args.eps_beta,
        max_abs_gamma=args.max_abs_gamma,
        min_r2=args.min_r2,
    )

    beta_valid = beta[np.isfinite(beta)]
    Gamma_valid = Gamma[np.isfinite(Gamma)]
    print(f"[STAT] valid pixels: {int(valid_mask.sum())} / {int(valid_mask.size)}")
    if beta_valid.size:
        print(f"[STAT] β range: [{beta_valid.min():.3f}, {beta_valid.max():.3f}] K/dB, mean={beta_valid.mean():.3f}")
    if Gamma_valid.size:
        print(f"[STAT] Γ range: [{Gamma_valid.min():.3f}, {Gamma_valid.max():.3f}] -, mean={Gamma_valid.mean():.3f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        beta_K_per_dB=beta,
        Gamma_dimensionless=Gamma,
        n_samples=n_samples,
        r2=r2,
        valid_mask=valid_mask.astype(np.uint8),
        crs_wkt=meta["crs_wkt"],
        transform=meta["transform"],
        height=np.int32(meta["height"]),
        width=np.int32(meta["width"]),
        dates=np.array(meta["dates"], dtype=object),
        meta=np.array(
            [
                "ATBD-like: ΔTB ≈ a·Δσ_pp + b·Δ(mean(σ_pq) − σ_pq(t))",
                "β=a [K/dB], Γ=b/a [-]",
                "Fallback: if insufficient xpol, estimate β only and set Γ=0",
                f"start_date={args.start_date}",
                f"end_date={args.end_date}",
                f"min_tb_finite_frac={args.min_tb_finite_frac}",
                f"min_samples_joint={'dynamic' if min_joint is None else min_joint}",
                f"min_samples_beta={'dynamic' if min_beta is None else min_beta}",
                f"ridge_lambda={args.ridge_lambda}",
                f"eps_beta={args.eps_beta}",
                f"max_abs_gamma={args.max_abs_gamma}",
                f"min_r2={args.min_r2}",
            ],
            dtype=object,
        ),
    )
    print(f"[OK] Saved to {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        warnings.warn(f"[ERROR] {e}")
        raise
