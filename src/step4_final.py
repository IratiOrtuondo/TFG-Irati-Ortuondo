#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ATBD-consistent tau–omega inversion to retrieve soil moisture from TB.

This script inverts the SMAP tau–omega radiative transfer model (ATBD Eq.1)
at native grid scale to retrieve soil moisture (SM) from brightness
temperature (TB). It follows these high-level steps per pixel:

1. Read TB (driver) from an input NPZ (auto-detects the TB key).
2. Prepare auxiliary inputs: Ts/Tc (surface/canopy temperature), tau,
   omega (single-scattering albedo), theta (incidence), and h (roughness).
   Each input can be supplied as a per-pixel NPZ or as a scalar fallback.
3. Invert the analytic ATBD equation for rough-surface reflectivity r_rough
   and compute gamma = exp(-tau * sec(theta)).
4. Undo roughness to get smooth reflectivity r_smooth (empirical roughness
   parameterization using h and exponent x ~ 2).
5. Solve the Fresnel equation for dielectric permittivity (ε) by bisection
   inversion of r_smooth(ε, θ) for the requested polarization (H/V).
6. Convert ε -> SM using a dielectric-to-SM mapping (Topp polynomial by default).
7. Apply regularization: clamp emissivity and SM to physically plausible ranges
   and avoid NaN / division blow-ups by stabilizing denominators.

Outputs: compressed NPZ containing soil_moisture, intermediate fields,
metadata (crs/transform/shape/date) and a small invalid mask for diagnostics.

Notes and design decisions:
- The implementation is vectorized over the 2D array using NumPy for speed.
- Regularization choices (e_min, e_max, sm_min, sm_max, den_eps) are
  exposed as CLI parameters to allow sensitivity testing.
- The script returns deterministic arrays (no NaNs in final SM) by
  clamping and filling invalid intermediate results conservatively.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

# Optional georef dependencies; non-fatal if absent (we store tuples)
try:
    from affine import Affine
except Exception:
    Affine = None


# -----------------------------
# Helpers: IO / parsing
# -----------------------------

def npz_load(path: Path) -> Dict:
    """Load an NPZ as a dict and raise if missing.

    Uses allow_pickle=True to be robust to object-typed metadata (e.g. WKT).
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return dict(np.load(path, allow_pickle=True))


def guess_date_from_name(name: str) -> Optional[str]:
    """Try to extract YYYYMMDD from a filename for use in default outputs."""
    m = re.search(r"(19|20)\d{6}", name)  # YYYYMMDD
    return m.group(0) if m else None


def find_key(d: Dict, candidates) -> Optional[str]:
    """Return the first key in `candidates` that exists in dict `d` or None."""
    for k in candidates:
        if k in d:
            return k
    return None


def find_tb_key(d: Dict) -> str:
    """Heuristic to locate the TB array key in an NPZ file.

    Looks for common canonical keys first and then falls back to the
    first 2D numeric array found. Raises KeyError when no candidate exists.
    """
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
    """Normalize transform to tuple or return None.

    Supports Affine instance, tuple/list, or numpy array with 6 elements.
    Keeps caller code simple when we write the transform back to NPZ.
    """
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
    """Return authoritative height,width for TB using metadata when present.

    If the NPZ includes 'height'/'width' and they match the TB shape, use
    them to avoid ambiguity when TB has been cropped or padded earlier.
    """
    h, w = TB.shape
    if "height" in d and "width" in d:
        hh = int(d["height"])
        ww = int(d["width"])
        if (hh, ww) == (h, w):
            return hh, ww
    return h, w


def load_map_or_const(npz_path: Optional[Path], key: str, shape: Tuple[int, int], const: float) -> np.ndarray:
    """Load a per-pixel map from NPZ or create a constant array of the given shape.

    Validates that the loaded array matches the expected shape and casts to float32.
    """
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
    """Horizontal-pol smooth-surface reflectivity for real ε (vectorized).

    Uses stable form with sqrt(max(eps - sin^2(theta), epsilon)) to avoid
    complex results when numerical rounding or small negative values occur.
    """
    st = np.sin(theta_rad)
    ct = np.cos(theta_rad)
    root = np.sqrt(np.maximum(eps - st * st, 1e-12))
    g = (ct - root) / (ct + root)
    return g * g


def fresnel_rv(eps: np.ndarray, theta_rad: np.ndarray) -> np.ndarray:
    """Vertical-pol smooth-surface reflectivity for real ε (vectorized)."""
    st = np.sin(theta_rad)
    ct = np.cos(theta_rad)
    root = np.sqrt(np.maximum(eps - st * st, 1e-12))
    g = (eps * ct - root) / (eps * ct + root)
    return g * g


def invert_eps_from_r(r_target: np.ndarray, theta_rad: np.ndarray, tb_pol: str, iters: int = 40) -> np.ndarray:
    """Bisection invert reflectivity -> dielectric constant ε for each pixel.

    r_target is clamped to [0, 0.9999] to avoid impossible values. The search
    interval for ε is [1, 80], which comfortably contains realistic soil
    permittivity values for moist soils. The function performs vectorized
    bisection for robustness and avoids Newton-like instabilities.
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
    """Approximate Topp polynomial mapping ε -> volumetric soil moisture (SM).

    The returned SM is clamped to [0, 0.6] to avoid nonphysical extrapolations.
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
    """Analytically invert ATBD Eq.1 for rough reflectivity r_rough per pixel.

    The algebra rearranges Eq.1 to isolate r (rough reflectivity) and returns
    r_rough together with gamma = exp(-tau * sec(theta)). To avoid division
    by very small denominators we clamp/regularize the denominator using
    `den_eps` which stabilizes pixels where the analytic solution is ill-conditioned.
    """
    sec = 1.0 / np.maximum(np.cos(theta_rad), 1e-6)
    gamma = np.exp(-tau * sec).astype(np.float32)

    A = (1.0 - omega) * (1.0 - gamma)  # canopy emission factor

    # Rearranged:
    # TB = Ts*gamma - Ts*gamma*r + Tc*A + Tc*A*r*gamma
    # TB - (Ts*gamma + Tc*A) = r*gamma*(Tc*A - Ts)
    num = (TB - (Ts * gamma + Tc * A)).astype(np.float32)
    den = (gamma * (Tc * A - Ts)).astype(np.float32)

    # Guard against near-zero denominators by replacing small |den| with den_eps
    den_abs = np.abs(den)
    den_safe = np.where(den_abs < float(den_eps),
                        np.sign(den) * float(den_eps) + (den == 0) * float(den_eps),
                        den).astype(np.float32)

    with np.errstate(divide="ignore", invalid="ignore"):
        r = num / den_safe

    return r.astype(np.float32), gamma


def undo_roughness(r_rough: np.ndarray, h: np.ndarray, theta_rad: np.ndarray, x: float = 2.0) -> np.ndarray:
    """Undo the exponential roughness parameterization to estimate r_smooth.

    Based on the ATBD simplified model:
      r_rough ≈ r_smooth * exp(-h * cos(theta)^x)
    so r_smooth = r_rough * exp(+h * cos(theta)^x)
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
    ap.add_argument("--sm-min", type=float, default=0.05, help="Min soil moisture (avoid SM=0 artifacts)")
    ap.add_argument("--sm-max", type=float, default=0.45, help="Max soil moisture cap")
    ap.add_argument("--den-eps", type=float, default=1e-3, help="Min abs(denominator) to avoid blow-ups")

    args = ap.parse_args()

    # --------------------------
    # Load input TB and meta
    # --------------------------
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

    # --------------------------
    # Load or default auxiliary maps
    # --------------------------
    # Temperature maps: Ts/Tc preferred; else Teff is used for both
    Teff = load_map_or_const(args.Teff_npz, "Teff_K", shape, args.Teff_const)
    Ts = load_map_or_const(args.Ts_npz, "Ts_K", shape, args.Teff_const) if args.Ts_npz else Teff.copy()
    Tc = load_map_or_const(args.Tc_npz, "Tc_K", shape, args.Teff_const) if args.Tc_npz else Teff.copy()

    # Theta (incidence)
    theta_deg = load_map_or_const(args.theta_npz, "theta_deg", shape, args.theta_const_deg)
    theta_rad = np.deg2rad(theta_deg.astype(np.float32))

    # Tau, omega, roughness
    tau = load_map_or_const(args.tau_npz, "tau", shape, args.tau_const)
    omega = load_map_or_const(args.omega_npz, "omega", shape, args.omega_const)
    hcoef = load_map_or_const(args.h_npz, "h", shape, args.h_const)

    # --------------------------
    # 1) Invert ATBD Eq.1 for rough reflectivity r_rough
    # --------------------------
    r_rough, gamma = solve_r_rough_from_tb_atbd(
        TB, Ts=Ts, Tc=Tc, tau=tau, omega=omega, theta_rad=theta_rad, den_eps=float(args.den_eps)
    )

    # Keep a copy of raw r_rough for diagnostics; we will regularize below
    r_rough_raw = r_rough.copy()
    invalid = ~np.isfinite(r_rough_raw)

    # --------------------------
    # Regularization: clamp emissivity (prevents SM->0 degeneracy) and bound r
    # --------------------------
    # e_rough_reg is clipped to [e_min, e_max] then converted back to r
    e_rough_reg = np.clip(1.0 - r_rough, float(args.e_min), float(args.e_max)).astype(np.float32)
    r_rough = (1.0 - e_rough_reg).astype(np.float32)

    # Bound reflectivity (rough) to [0, 0.999]
    r_rough = np.clip(r_rough, 0.0, 0.999).astype(np.float32)

    # --------------------------
    # 2) Undo roughness to estimate r_smooth
    # --------------------------
    r_smooth = undo_roughness(r_rough, h=hcoef, theta_rad=theta_rad, x=float(args.roughness_x))
    r_smooth_raw = r_smooth.copy()
    invalid |= ~np.isfinite(r_smooth_raw)
    r_smooth = np.clip(r_smooth, 0.0, 0.999).astype(np.float32)

    # --------------------------
    # 3) Fresnel inversion (r_smooth -> ε)
    # --------------------------
    eps = invert_eps_from_r(r_smooth, theta_rad=theta_rad, tb_pol=tb_pol, iters=40).astype(np.float32)

    # --------------------------
    # 4) ε -> SM via Topp polynomial
    # --------------------------
    if args.sm_method == "topp":
        SM = topp_eps_to_sm(eps)
    else:
        raise ValueError(f"Unknown --sm-method: {args.sm_method}")

    # Final SM clamp to avoid extreme values and ensure no NaNs in output
    SM = np.clip(SM, float(args.sm_min), float(args.sm_max)).astype(np.float32)

    # Also compute emissivities for output/diagnostics
    e_rough = (1.0 - r_rough).astype(np.float32)
    e_smooth = (1.0 - r_smooth).astype(np.float32)

    # --------------------------
    # Write output NPZ including intermediate diagnostics
    # --------------------------
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
