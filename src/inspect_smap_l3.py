#!/usr/bin/env python3
"""Inspect a SMAP L3 HDF5 file and highlight TB-like datasets.

This script traverses all datasets in a SMAP L3 HDF5 file and prints
their path, shape, dtype, and a marker when they look like brightness
temperature (TB) variables based on their names or attributes.

It also lists the top-level groups with a small sample of their children.
"""

from __future__ import annotations

import argparse
from typing import Any

import h5py

_TB_NAME_HINTS = ("tb", "brightness", "temperature")
_TB_ATTR_HINTS = ("tb", "brightness")


def _looks_like_tb_name(name: str) -> bool:
    """Return True if the dataset name suggests brightness temperature.

    Args:
        name: HDF5 dataset path or name.

    Returns:
        True if the name contains a TB-related hint, False otherwise.
    """
    name_lower = name.lower()
    return any(hint in name_lower for hint in _TB_NAME_HINTS)


def _looks_like_tb_attr(value: str) -> bool:
    """Return True if an attribute value suggests brightness temperature.

    Args:
        value: String representation of an attribute value.

    Returns:
        True if the value contains a TB-related hint, False otherwise.
    """
    value_lower = value.lower()
    return any(hint in value_lower for hint in _TB_ATTR_HINTS)


def visit_all(name: str, obj: Any) -> None:
    """Visitor that prints information about each dataset.

    It highlights datasets that look like brightness temperature (TB)
    based on their name or attributes.

    Args:
        name: HDF5 object name (relative path inside the file).
        obj: HDF5 object (dataset or group).
    """
    if not isinstance(obj, h5py.Dataset):
        return

    shape = obj.shape
    dtype = obj.dtype

    has_tb = _looks_like_tb_name(name)
    attrs_info = ""

    for key, value in obj.attrs.items():
        try:
            str_value = (
                value.decode()
                if isinstance(value, (bytes, bytearray))
                else str(value)
            )
        except Exception:  # pragma: no cover - defensive
            str_value = str(value)

        if _looks_like_tb_attr(str_value):
            attrs_info = f"  [ATTR: {key}={str_value[:50]}]"
            has_tb = True

    marker = ">>> TB? <<<" if has_tb else ""
    print(f"{name:70s} {str(shape):20s} {dtype} {marker} {attrs_info}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a SMAP L3 HDF5 file and highlight brightness "
            "temperature (TB)-like datasets."
        )
    )
    parser.add_argument(
        "file",
        help="Path to the SMAP L3 HDF5 file (e.g., SMAP_L3_SM_P_*.h5).",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for SMAP L3 inspection."""
    args = parse_args()
    path = args.file

    with h5py.File(path, "r") as h5_file:
        print("=" * 80)
        print(f"File: {path}")
        print("=" * 80)

        print("\n--- All datasets ---")
        h5_file.visititems(visit_all)

        print("\n--- Top-level groups ---")
        for key in h5_file.keys():
            print(f"  {key}")
            group = h5_file[key]
            if hasattr(group, "keys"):
                # Show only a subset of children to avoid flooding output.
                for subkey in list(group.keys())[:10]:
                    print(f"    - {subkey}")


if __name__ == "__main__":
    main()
