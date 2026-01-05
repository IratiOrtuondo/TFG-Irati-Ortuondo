#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step6_sm_from_tb_tauomega_atbd.py

Step 6 (ATBD-consistent): Retrieve soil moisture at fine/native grid by inverting the SMAP
tau–omega radiative transfer model (ATBD Eq. 1) from disaggregated TB.

ATBD tau–omega (Eq. 1):
  TB_p = Ts * e_p * gamma + Tc * (1 - omega_p) * (1 - gamma) * (1 + r_p * gamma)
  gamma = exp(-tau_p * sec(theta))
  e_p = 1 - r_p  (rough-surface emissivity/reflectivity at look angle)

Roughness parameterization (ATBD, simplified from the general Q-mixing form):
  r_rough ≈ r_smooth * exp(-h * cos(theta)^x) , with x=2 in SMAP operational processing

Inversion steps:
  1) Solve ATBD Eq.1 for r_rough (analytic)
  2) Undo roughness to get r_smooth
  3) Invert Fresnel (smooth) -> epsilon (real) via bisection
  4) epsilon -> SM using Topp polynomial (approx).

Regularization (to avoid SM=0 saturation and NaNs):
  - Clamp emissivity (e) to [e_min, e_max] by mapping r <-> e
  - Clamp final SM to [sm_min, sm_max]
  - Protect against near-zero denominators in analytic r inversion

Inputs:
  --tb : NPZ with TB on fine grid (key auto-detected)
Optional maps (NPZ on same grid):
  --Teff-npz : key Teff_K (used for Ts and Tc by default)
  --Ts-npz   : key Ts_K
  --Tc-npz   : key Tc_K
  --theta-npz: key theta_deg  (incidence angle per-pixel)
  --tau-npz  : key tau
  --omega-npz: key omega
  --h-npz    : key h

If map not provided, constants are used.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

# Optional georef dependencies; script works without them
try:
    from affine import Affine
except Exception:
    Affine = None


# -----------------------------
# Helpers: IO / parsing
# -----------------------------
def npz_load(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return dict(np.load(path, allow_pickle=True))


def guess_date_from_name(name: str) -> Optional[str]:
    m = re.search(r"(19|20)\d{6}", name)  # YYYYMMDD
    return m.group(0) if m else None


def find_key(d: Dict, candidates) -> Optional[str]:
    for k in candidates:
        if k in d:
            return k
    return None


def find_tb_key(d: Dict) -> str:
    candidates = [
        "TB_fine", "TB_FINE", "tb_fine",
        "TB", "tb",
        "brightness_temperature", "brightness_temperature_K",
    ]
    k = find_key(d, candidates)
    if k is not None:
        return k
    # fallback: first 2D numeric array
    for kk, vv in d.items():
        if isinstance(vv, np.ndarray) and vv.ndim == 2 and np.issubdtype(vv.dtype, np.number):
            return kk
    raise KeyError("Could not find TB array in NPZ (expected key like 'TB_fine' or 'TB').")


def to_affine(transform_obj):
    if transform_obj is None:
        return None
    if Affine is not None and isinstance(transform_obj, Affine):
        return tuple(transform_obj)
    if isinstance(transform_obj, (tuple, list)):
        return tuple(transform_obj)
    if isinstance(transform_obj, np.ndarray):
        return tuple(transform_obj.tolist())
    return transform_obj


def get_hw_from_tb_and_meta(TB: np.ndarray, d: Dict) -> Tuple[int, int]:
    h, w = TB.shape
    if "height" in d and "width" in d:
        hh = int(d["height"])
        ww = int(d["width"])
        if (hh, ww) == (h, w):
            return hh, ww
    return h, w


def load_map_or_const(npz_path: Optional[Path], key: str, shape: Tuple[int, int], const: float) -> np.ndarray:
    if npz_path is None:
        return np.full(shape, float(const), dtype=np.float32)
    dd = npz_load(npz_path)
    if key not in dd:
        raise KeyError(f"{npz_path} must contain variable '{key}'. Keys: {list(dd.keys())[:20]}")
    arr = np.array(dd[key], dtype=np.float32)
    if arr.shape != shape:
        raise ValueError(f"{key} shape {arr.shape} does not match expected {shape}.")
    return arr


# -----------------------------
# Physics: Fresnel + inversion
# -----------------------------
def fresnel_rh(eps: np.ndarray, theta_rad: np.ndarray) -> np.ndarray:
    st = np.sin(theta_rad)
    ct = np.cos(theta_rad)
    root = np.sqrt(np.maximum(eps - st * st, 1e-12))
    g = (ct - root) / (ct + root)
    return g * g


def fresnel_rv(eps: np.ndarray, theta_rad: np.ndarray) -> np.ndarray:
    st = np.sin(theta_rad)
    ct = np.cos(theta_rad)
    root = np.sqrt(np.maximum(eps - st * st, 1e-12))
    g = (eps * ct - root) / (eps * ct + root)
    return g * g


def invert_eps_from_r(r_target: np.ndarray, theta_rad: np.ndarray, tb_pol: str, iters: int = 40) -> np.ndarray:
    """
    Vectorized bisection inversion for smooth-surface Fresnel reflectivity:
      r_smooth(eps) = r_target, eps in [1, 80]
    """
    r = np.clip(r_target.astype(np.float32), 0.0, 0.9999)
    lo = np.full_like(r, 1.0, dtype=np.float32)
    hi = np.full_like(r, 80.0, dtype=np.float32)

    pol = tb_pol.upper()
    if pol not in ("H", "V"):
        raise ValueError("--tb-pol must be 'H' or 'V' (radiometer polarization).")

    f = fresnel_rh if pol == "H" else fresnel_rv

    for _ in range(iters):
        mid = (lo + hi) * 0.5
        fmid = f(mid, theta_rad)
        go_hi = fmid > r
        hi = np.where(go_hi, mid, hi)
        lo = np.where(go_hi, lo, mid)

    return (lo + hi) * 0.5


def topp_eps_to_sm(eps: np.ndarray) -> np.ndarray:
    """
    Topp et al. polynomial (approx). Clips to [0, 0.6].
    """
    e = eps.astype(np.float32)
    mv = (-0.053 +
          0.0292 * e +
          (-0.00055) * e * e +
          0.0000043 * e * e * e)
    return np.clip(mv, 0.0, 0.6).astype(np.float32)


# -----------------------------
# ATBD tau-omega inversion
# -----------------------------
def solve_r_rough_from_tb_atbd(
    TB: np.ndarray,
    Ts: np.ndarray,
    Tc: np.ndarray,
    tau: np.ndarray,
    omega: np.ndarray,
    theta_rad: np.ndarray,
    den_eps: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Invert ATBD Eq.1 for rough-surface reflectivity r_p at look angle.

    TB = Ts*(1-r)*gamma + Tc*(1-omega)*(1-gamma)*(1 + r*gamma)
    where gamma = exp(-tau * sec(theta))

    Returns:
      r_rough, gamma
    """
    sec = 1.0 / np.maximum(np.cos(theta_rad), 1e-6)
    gamma = np.exp(-tau * sec).astype(np.float32)

    A = (1.0 - omega) * (1.0 - gamma)  # canopy emission factor

    # Rearranged:
    # TB = Ts*gamma - Ts*gamma*r + Tc*A + Tc*A*r*gamma
    # TB - (Ts*gamma + Tc*A) = r*gamma*(Tc*A - Ts)
    num = (TB - (Ts * gamma + Tc * A)).astype(np.float32)
    den = (gamma * (Tc * A - Ts)).astype(np.float32)

    # Protect ill-conditioned pixels: when den ~ 0, r can blow up
    den_abs = np.abs(den)
    den_safe = np.where(den_abs < float(den_eps),
                        np.sign(den) * float(den_eps) + (den == 0) * float(den_eps),
                        den).astype(np.float32)

    with np.errstate(divide="ignore", invalid="ignore"):
        r = num / den_safe

    return r.astype(np.float32), gamma


def undo_roughness(r_rough: np.ndarray, h: np.ndarray, theta_rad: np.ndarray, x: float = 2.0) -> np.ndarray:
    """
    Simplified ATBD roughness:
      r_rough ≈ r_smooth * exp(-h * cos(theta)^x)
      => r_smooth = r_rough * exp(+h * cos(theta)^x)
    """
    c = np.maximum(np.cos(theta_rad), 0.0).astype(np.float32)
    factor = np.exp(h * (c ** x)).astype(np.float32)
    r_smooth = r_rough * factor
    return r_smooth.astype(np.float32)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tb", type=Path, required=True, help="Path to TB fine NPZ")
    ap.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: TB dir)")
    ap.add_argument("--date", type=str, default=None, help="YYYYMMDD (optional; inferred)")
    ap.add_argument("--tb-pol", type=str, required=True, help="Radiometer TB polarization: H or V")
    ap.add_argument("--out-name", type=str, default=None, help="Override output filename")

    # Temperature inputs
    ap.add_argument("--Teff-npz", type=Path, default=None, help="NPZ with Teff_K (used for Ts & Tc)")
    ap.add_argument("--Ts-npz", type=Path, default=None, help="NPZ with Ts_K")
    ap.add_argument("--Tc-npz", type=Path, default=None, help="NPZ with Tc_K")
    ap.add_argument("--Teff-const", type=float, default=300.0, help="Fallback Teff (K)")

    # Geometry / tau-omega / roughness
    ap.add_argument("--theta-npz", type=Path, default=None, help="NPZ with theta_deg")
    ap.add_argument("--theta-const-deg", type=float, default=40.0, help="Fallback incidence angle (deg)")

    ap.add_argument("--tau-npz", type=Path, default=None, help="NPZ with tau")
    ap.add_argument("--tau-const", type=float, default=0.10, help="Fallback tau")

    ap.add_argument("--omega-npz", type=Path, default=None, help="NPZ with omega")
    ap.add_argument("--omega-const", type=float, default=0.05, help="Fallback omega")

    ap.add_argument("--h-npz", type=Path, default=None, help="NPZ with h (roughness coefficient)")
    ap.add_argument("--h-const", type=float, default=0.0, help="Fallback h (0 => no roughness correction)")
    ap.add_argument("--roughness-x", type=float, default=2.0,
                    help="Exponent x in exp(-h cos^x theta). SMAP operational uses x=2.")

    # Dielectric -> SM
    ap.add_argument("--sm-method", type=str, default="topp", choices=["topp"], help="epsilon->SM (topp for now)")

    # Regularization (NO NaNs, avoid SM=0 saturation)
    ap.add_argument("--e-min", type=float, default=0.65, help="Min emissivity clamp (physical regularization)")
    ap.add_argument("--e-max", type=float, default=0.98, help="Max emissivity clamp (avoid e->1 saturation)")
    ap.add_argument("--sm-min", type=float, default=0.05, help="Min soil moisture (avoid SM=0 artifacts)")  # <- requested
    ap.add_argument("--sm-max", type=float, default=0.45, help="Max soil moisture cap")
    ap.add_argument("--den-eps", type=float, default=1e-3, help="Min abs(denominator) to avoid blow-ups")

    args = ap.parse_args()

    tb_path = args.tb
    d = npz_load(tb_path)

    date = args.date or guess_date_from_name(tb_path.name) or "unknown_date"
    tb_pol = args.tb_pol.upper()
    if tb_pol not in ("H", "V"):
        raise ValueError("--tb-pol must be 'H' or 'V'.")

    tb_key = find_tb_key(d)
    TB = np.array(d[tb_key], dtype=np.float32)
    if TB.ndim != 2:
        raise ValueError(f"TB array '{tb_key}' is not 2D. Found shape: {TB.shape}")

    hgt, wdt = get_hw_from_tb_and_meta(TB, d)
    shape = (hgt, wdt)

    # Temperatures: prefer explicit Ts/Tc; else Teff
    Teff = load_map_or_const(args.Teff_npz, "Teff_K", shape, args.Teff_const)
    Ts = load_map_or_const(args.Ts_npz, "Ts_K", shape, args.Teff_const) if args.Ts_npz else Teff.copy()
    Tc = load_map_or_const(args.Tc_npz, "Tc_K", shape, args.Teff_const) if args.Tc_npz else Teff.copy()

    # Theta
    theta_deg = load_map_or_const(args.theta_npz, "theta_deg", shape, args.theta_const_deg)
    theta_rad = np.deg2rad(theta_deg.astype(np.float32))

    # tau, omega, h
    tau = load_map_or_const(args.tau_npz, "tau", shape, args.tau_const)
    omega = load_map_or_const(args.omega_npz, "omega", shape, args.omega_const)
    hcoef = load_map_or_const(args.h_npz, "h", shape, args.h_const)

    # 1) Invert ATBD Eq.1 for rough reflectivity
    r_rough, gamma = solve_r_rough_from_tb_atbd(
        TB, Ts=Ts, Tc=Tc, tau=tau, omega=omega, theta_rad=theta_rad, den_eps=float(args.den_eps)
    )

    # Track invalids, but DO NOT produce NaNs in output SM (we'll regularize instead)
    r_rough_raw = r_rough.copy()
    invalid = ~np.isfinite(r_rough_raw)

    # --- Physical regularization via emissivity bounds (prevents SM -> 0 saturation) ---
    e_rough_reg = np.clip(1.0 - r_rough, float(args.e_min), float(args.e_max)).astype(np.float32)
    r_rough = (1.0 - e_rough_reg).astype(np.float32)

    # Bound reflectivity (rough) to [0, 0.999]
    r_rough = np.clip(r_rough, 0.0, 0.999).astype(np.float32)

    # 2) Undo roughness to get smooth reflectivity
    r_smooth = undo_roughness(r_rough, h=hcoef, theta_rad=theta_rad, x=float(args.roughness_x))
    r_smooth_raw = r_smooth.copy()
    invalid |= ~np.isfinite(r_smooth_raw)
    r_smooth = np.clip(r_smooth, 0.0, 0.999).astype(np.float32)

    # 3) Fresnel inversion (smooth)
    eps = invert_eps_from_r(r_smooth, theta_rad=theta_rad, tb_pol=tb_pol, iters=40).astype(np.float32)

    # 4) epsilon -> SM
    if args.sm_method == "topp":
        SM = topp_eps_to_sm(eps)
    else:
        raise ValueError(f"Unknown --sm-method: {args.sm_method}")

    # Final SM clamp (requested: no NaNs, no SM=0)
    SM = np.clip(SM, float(args.sm_min), float(args.sm_max)).astype(np.float32)

    # emissivities
    e_rough = (1.0 - r_rough).astype(np.float32)
    e_smooth = (1.0 - r_smooth).astype(np.float32)

    # Output
    out_dir = args.out_dir if args.out_dir is not None else tb_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = args.out_name or f"SM_fine_{date}_TB{tb_pol}_tauomega_ATBD_reg.npz"
    out_path = out_dir / out_name

    crs_wkt = None
    if "crs_wkt" in d:
        crs_wkt = str(d["crs_wkt"])
    elif "crs" in d:
        crs_wkt = str(d["crs"])

    transform = to_affine(d.get("transform", None))

    np.savez_compressed(
        out_path,
        soil_moisture=SM,
        epsilon_smooth=eps,
        r_rough=r_rough,
        r_smooth=r_smooth,
        e_rough=e_rough,
        e_smooth=e_smooth,
        gamma=gamma,
        TB=TB,
        Ts_K=Ts,
        Tc_K=Tc,
        Teff_K=Teff,
        tau=tau,
        omega=omega,
        h=hcoef,
        theta_deg=theta_deg,
        tb_pol=tb_pol,
        tb_key=tb_key,
        invalid_mask=invalid.astype(np.uint8),
        r_rough_raw=r_rough_raw,
        r_smooth_raw=r_smooth_raw,
        crs_wkt=crs_wkt,
        transform=transform,
        height=hgt,
        width=wdt,
        date=date,
        source_tb_npz=str(tb_path),
        # record regularization params used
        e_min=float(args.e_min),
        e_max=float(args.e_max),
        sm_min=float(args.sm_min),
        sm_max=float(args.sm_max),
        den_eps=float(args.den_eps),
    )

    print(f"[OK] TB key used: {tb_key}")
    print(f"[OK] Saved: {out_path}")
    bad = int(np.sum(invalid))
    print(f"[INFO] invalid intermediate pixels (pre-regularization): {bad} / {hgt*wdt}")
    print(f"[INFO] SM clipped to [{args.sm_min:.3f}, {args.sm_max:.3f}] ; e clipped to [{args.e_min:.3f}, {args.e_max:.3f}] ; den_eps={args.den_eps:g}")


if __name__ == "__main__":
    main()
