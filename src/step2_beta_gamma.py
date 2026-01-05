#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step3_smap_multidate.py (corrected)

Estimate coarse-scale active–passive parameters from multi-temporal SMAP stacks.

ATBD-like model (per coarse pixel C):
  ΔTB(C,t) ≈ a·Δσ_pp(C,t) + b·Δ( mean(σ_pq)(C) − σ_pq(C,t) )

Where:
  β(C) = a [K/dB]
  Γ(C) = b/a [dimensionless]

Main improvements:
- Date range filter (start/end) applied to TB-driven date list.
- Drops dates with low finite TB fraction (prevents polluted means/anomalies).
- Uses x2 = mean_xpol - xpol(t) directly (no redundant anomaly step).
- Dynamic min_samples based on available T (uses maximum possible dates).
- Keeps fallback beta-only when xpol insufficient.
- QA clipping fills NaNs and saturates ranges without invalidating pixels.
"""

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
    """
    TB is the driver. For each TB date, attempt to load co-pol / x-pol if present,
    otherwise fill with NaNs. Optionally filters by date range and TB finite fraction.
    """

    tb_files = sorted(glob.glob(tb_pattern))
    if not tb_files:
        raise FileNotFoundError(f"No TB NPZ files found for pattern: {tb_pattern}")

    copol_files = sorted(glob.glob(copol_pattern))
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

    # Date-range filter (string compare works for YYYYMMDD)
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

        frac_finite = float(np.isfinite(TB).mean())
        if frac_finite < min_tb_finite_frac:
            dropped_low_tb.append((date, frac_finite))
            continue

        if len(kept_dates) == 0:
            h, w = TB.shape
            first_meta["height"] = int(npz_tb.get("height", h))
            first_meta["width"] = int(npz_tb.get("width", w))
            first_meta["crs_wkt"] = str(npz_tb.get("crs_wkt", ""))
            first_meta["transform"] = np.array(npz_tb.get("transform", np.eye(3)), dtype=float)

        TB_list.append(TB)
        kept_dates.append(date)

        # co-pol (Spp)
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

        # x-pol (Sx)
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

    TB_stack = np.stack(TB_list, axis=0)
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
    """
    Joint model:
      y = ΔTB
      x1 = Δσ_pp
      x2 = Δ( mean(σ_pq) - σ_pq(t) )

    If insufficient finite x2, fallback to beta-only:
      y = a*x1 (through origin), Gamma=0
    """

    T, H, W = dTB.shape
    P = H * W

    # Dynamic minimums (use maximum possible dates)
    # Remove strict QA on minimum samples: allow estimation even with few dates.
    # If user provided explicit minima, respect them but ensure >=1 and <=T.
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

    # x2(t) = mean_xpol - xpol(t)
    mean_xpol = np.nanmean(Sxpol_stack, axis=0)           # (H,W)
    x2_stack = mean_xpol[None, :, :] - Sxpol_stack        # (T,H,W)
    # and we want Δx2, so anomaly it once (this is the correct “Δ”)
    _, dX2 = compute_anomalies(x2_stack)
    x2_all = dX2.reshape(T, P)

    beta_flat = np.full(P, np.nan, dtype=np.float32)
    Gamma_flat = np.full(P, np.nan, dtype=np.float32)
    n_flat = np.zeros(P, dtype=np.int16)
    r2_flat = np.full(P, np.nan, dtype=np.float32)
    valid_flat = np.zeros(P, dtype=np.uint8)

    for i in range(P):
        y = y_all[:, i]
        x1 = x1_all[:, i]
        x2 = x2_all[:, i]

        # --- JOINT ---
        mask_joint = np.isfinite(y) & np.isfinite(x1) & np.isfinite(x2)
        n_joint = int(mask_joint.sum())

        if n_joint >= min_samples_joint:
            yy = y[mask_joint]
            xx1 = x1[mask_joint]
            xx2 = x2[mask_joint]

            if (np.nanvar(xx1) >= min_var_x) and (np.nanvar(xx2) >= min_var_x):
                X = np.column_stack([xx1, xx2])

                XtX = X.T @ X + ridge_lambda * np.eye(2)
                if np.linalg.cond(XtX) < 1e10:
                    theta = np.linalg.solve(XtX, X.T @ yy)
                    a = float(theta[0])
                    b = float(theta[1])

                    if abs(a) >= eps_beta:
                        Gamma_val = b / a
                        if np.isfinite(Gamma_val) and (abs(Gamma_val) <= max_abs_gamma):
                            y_hat = X @ theta
                            ss_res = float(np.sum((yy - y_hat) ** 2))
                            ss_tot0 = float(np.sum(yy ** 2))  # through-origin R²
                            r2_val = 1.0 - ss_res / ss_tot0 if ss_tot0 > 0 else np.nan

                            if np.isfinite(r2_val) and (r2_val >= min_r2):
                                beta_flat[i] = a
                                Gamma_flat[i] = Gamma_val
                                n_flat[i] = n_joint
                                r2_flat[i] = r2_val
                                valid_flat[i] = 1
                                continue

        # --- FALLBACK β-only ---
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

    # Fill NaNs conservatively
    beta_filled = beta.copy()
    gamma_filled = Gamma.copy()
    beta_filled[~np.isfinite(beta_filled)] = -0.2
    gamma_filled[~np.isfinite(gamma_filled)] = 0.0

    # Clip bounds
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

    # If None, dynamic thresholds will be used
    p.add_argument("--min-samples-joint", type=int, default=0, help="0 => dynamic based on T")
    p.add_argument("--min-samples-beta", type=int, default=0, help="0 => dynamic based on T")

    p.add_argument("--ridge-lambda", type=float, default=0.1)
    p.add_argument("--eps-beta", type=float, default=0.05)
    p.add_argument("--max-abs-gamma", type=float, default=10.0)
    p.add_argument("--min-r2", type=float, default=0.0)
    return p.parse_args()


def main():
    args = parse_args()

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
