#!/usr/bin/env python3
"""Plot outputs from step6 tau-omega inversion NPZ.

Usage:
  python plot_sm_tauomega.py --npz ../data/processed/SM_fine_20150607_VV_tauomega.npz --out-dir ../data/figures
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def load_npz(path: Path) -> dict:
    if not Path(path).exists():
        raise FileNotFoundError(f"NPZ not found: {path}")
    return dict(np.load(path, allow_pickle=True))


def main():
    p = argparse.ArgumentParser(description="Plot SM and related maps from tau-omega NPZ")
    p.add_argument("--npz", required=True, help="Path to SM NPZ produced by step6")
    p.add_argument("--out-dir", default="../data/figures", help="Directory to save PNG")
    p.add_argument("--vmin-tb", type=float, default=None)
    p.add_argument("--vmax-tb", type=float, default=None)
    args = p.parse_args()

    d = load_npz(Path(args.npz))

    # Prefer canonical names
    SM = d.get("soil_moisture")
    eps = d.get("epsilon")
    e = d.get("emissivity")
    r = d.get("reflectivity")
    TB = d.get("TB")

    if SM is None and "soil_moisture" not in d:
        # try alternative keys
        for k in ("sm", "SM", "soil_moisture_fine"):
            if k in d:
                SM = d[k]
                break

    if SM is None:
        raise KeyError("soil_moisture not found in NPZ")

    # Manual crop: user requested to show only up to index 35 in x (columns) and 28 in y (rows)
    # This means [:28, :35] slice (rows 0-27, columns 0-34)
    def manual_crop(a, max_y=28, max_x=35):
        if a is None:
            return None
        return a[:max_y, :max_x]
    
    TB = manual_crop(TB)
    SM = manual_crop(SM)
    eps = manual_crop(eps)
    e = manual_crop(e)
    r = manual_crop(r)

    os.makedirs(args.out_dir, exist_ok=True)
    base = Path(args.npz).stem
    outpng = Path(args.out_dir) / f"{base}.png"

    # Setup figure: 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()

    # TB
    if TB is not None:
        tb = TB
        vmin = args.vmin_tb if args.vmin_tb is not None else float(np.nanpercentile(tb[np.isfinite(tb)], 2))
        vmax = args.vmax_tb if args.vmax_tb is not None else float(np.nanpercentile(tb[np.isfinite(tb)], 98))
        cmap_tb = plt.get_cmap("viridis").copy()
        cmap_tb.set_bad("white")
        im0 = axes[0].imshow(tb, origin="upper", cmap=cmap_tb, vmin=vmin, vmax=vmax)
        axes[0].set_title(f"TB used (key='{d.get('tb_key','?')}')")
        plt.colorbar(im0, ax=axes[0], fraction=0.046)
    else:
        axes[0].text(0.5, 0.5, "TB (not present)", ha="center", va="center")
        axes[0].set_axis_off()

    # Soil moisture
    cmap_sm = plt.get_cmap("plasma").copy()
    cmap_sm.set_bad("white")
    im1 = axes[1].imshow(SM, origin="upper", cmap=cmap_sm, vmin=0.0, vmax=0.6)
    axes[1].set_title(f"Soil moisture (SM)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # Epsilon
    if eps is not None:
        cmap_eps = plt.get_cmap("YlGn").copy()
        cmap_eps.set_bad("white")
        im2 = axes[2].imshow(eps, origin="upper", cmap=cmap_eps)
        axes[2].set_title("Dielectric permittivity (epsilon)")
        plt.colorbar(im2, ax=axes[2], fraction=0.046)
    else:
        axes[2].text(0.5, 0.5, "epsilon (not present)", ha="center", va="center")
        axes[2].set_axis_off()

    # Emissivity
    if e is not None:
        cmap_e = plt.get_cmap("coolwarm").copy()
        cmap_e.set_bad("white")
        im3 = axes[3].imshow(e, origin="upper", cmap=cmap_e, vmin=0.0, vmax=1.0)
        axes[3].set_title("Emissivity")
        plt.colorbar(im3, ax=axes[3], fraction=0.046)
    else:
        axes[3].text(0.5, 0.5, "emissivity (not present)", ha="center", va="center")
        axes[3].set_axis_off()

    # Reflectivity
    if r is not None:
        cmap_r = plt.get_cmap("inferno").copy()
        cmap_r.set_bad("white")
        im4 = axes[4].imshow(r, origin="upper", cmap=cmap_r, vmin=0.0, vmax=1.0)
        axes[4].set_title("Reflectivity")
        plt.colorbar(im4, ax=axes[4], fraction=0.046)
    else:
        axes[4].text(0.5, 0.5, "reflectivity (not present)", ha="center", va="center")
        axes[4].set_axis_off()

    # Last panel: metadata / histogram of SM
    axes[5].axis('off')
    txt = []
    txt.append(f"{base}")
    if "date" in d:
        txt.append(f"Date: {d['date']}")
    if "pol" in d:
        txt.append(f"Pol: {d['pol']}")
    txt.append(f"Shape: {SM.shape} | Finite: {int(np.isfinite(SM).sum())}")
    try:
        smv = SM[np.isfinite(SM)]
        txt.append(f"SM range: [{smv.min():.3f}, {smv.max():.3f}]")
    except Exception:
        pass
    axes[5].text(0.01, 0.5, "\n".join(txt), va="center", fontsize=10)

    fig.suptitle(f"SM Tau-Omega Results: {base}")
    plt.tight_layout()
    fig.savefig(outpng, dpi=200)
    print(f"[OK] Saved plot: {outpng}")


if __name__ == '__main__':
    main()
