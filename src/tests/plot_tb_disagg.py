#!/usr/bin/env python3
"""Quick plot for tb_disagg NPZ files.

Usage:
  python plot_tb_disagg.py --npz ../data/interim/tb_disagg-20150607-VV.npz --out-dir ../data/figures
"""
from __future__ import annotations

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def load_npz(path: str) -> dict:
    d = dict(np.load(path, allow_pickle=True))
    return d


def main():
    p = argparse.ArgumentParser(description="Plot tb_disagg NPZ (saves PNG to --out-dir)")
    p.add_argument("--npz", required=True, help="Path to tb_disagg-YYYYMMDD-POL.npz")
    p.add_argument("--out-dir", default="../data/figures", help="Directory to save PNG")
    p.add_argument("--vmin", type=float, default=None)
    p.add_argument("--vmax", type=float, default=None)
    args = p.parse_args()

    data = load_npz(args.npz)

    TBc_native = data.get("TBc_native")
    TB_fine = data.get("TB_fine")
    dSigma = data.get("dSigma_copol")
    dSigma_x = data.get("dSigma_xpol")

    arrs = [TBc_native, TB_fine, dSigma]
    titles = ["TBc_native", "TB_fine", "dSigma_copol"]

    # determine vmin/vmax from data if not provided
    if args.vmin is None or args.vmax is None:
        vals = np.hstack([np.ravel(a) for a in arrs if a is not None])
        finite = np.isfinite(vals)
        if finite.any():
            lo = float(np.nanpercentile(vals[finite], 2))
            hi = float(np.nanpercentile(vals[finite], 98))
            vmin = args.vmin if args.vmin is not None else lo
            vmax = args.vmax if args.vmax is not None else hi
        else:
            vmin, vmax = None, None
    else:
        vmin, vmax = args.vmin, args.vmax

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.npz))[0]
    outpng = os.path.join(args.out_dir, f"{base}.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, arr, title in zip(axes, arrs, titles):
        if arr is None:
            ax.text(0.5, 0.5, "(not found)", ha="center", va="center")
            ax.set_axis_off()
            continue
        im = ax.imshow(arr, origin="upper", vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(base)
    plt.tight_layout()
    fig.savefig(outpng, dpi=200)
    print(f"[OK] Saved plot: {outpng}")


if __name__ == "__main__":
    main()
