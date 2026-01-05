"""RMSE vs time plot for fine-resolution soil moisture validation (per pixel).

This script computes the per-date RMSE for two collocated fine-resolution pixels
independently (no combination/averaging across pixels) and plots both time series
together with the SMAP expected accuracy threshold.
"""

from datetime import datetime
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


SMAP_EXPECTED_RMSE = 0.04  # m3/m3
DATE_FORMAT = "%Y%m%d"


def parse_dates(date_strings: List[str]) -> List[datetime]:
    """Converts date strings to datetime objects.

    Args:
        date_strings: List of dates in YYYYMMDD format.

    Returns:
        List of datetime objects.
    """
    return [datetime.strptime(date, DATE_FORMAT) for date in date_strings]


def compute_rmse_per_date_single_pixel(
    sm_est: np.ndarray,
    sm_ref: np.ndarray,
) -> np.ndarray:
    """Computes per-date RMSE for a single pixel.

    Note: With one observation per date, RMSE reduces to absolute error per date.

    Args:
        sm_est: Estimated soil moisture for the pixel (%).
        sm_ref: Reference soil moisture for the pixel (%).

    Returns:
        Per-date RMSE values in m3/m3.
    """
    if sm_est.shape != sm_ref.shape:
        raise ValueError("sm_est and sm_ref must have the same shape.")

    rmse_percent = np.sqrt((sm_est - sm_ref) ** 2)
    return rmse_percent / 100.0  # Convert % to m3/m3


def compute_two_pixel_rmse_time_series(
    sm1_est: np.ndarray,
    sm1_ref: np.ndarray,
    sm2_est: np.ndarray,
    sm2_ref: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes per-date RMSE time series independently for two pixels.

    Args:
        sm1_est: Estimated soil moisture for pixel 1 (%).
        sm1_ref: Reference soil moisture for pixel 1 (%).
        sm2_est: Estimated soil moisture for pixel 2 (%).
        sm2_ref: Reference soil moisture for pixel 2 (%).

    Returns:
        Tuple (rmse_pixel_1, rmse_pixel_2) in m3/m3.
    """
    rmse_p1 = compute_rmse_per_date_single_pixel(sm1_est, sm1_ref)
    rmse_p2 = compute_rmse_per_date_single_pixel(sm2_est, sm2_ref)
    return rmse_p1, rmse_p2


def plot_two_pixel_rmse_time_series(
    dates: List[datetime],
    rmse_p1: np.ndarray,
    rmse_p2: np.ndarray,
    output_path: str,
) -> None:
    """Plots two RMSE time series (one per pixel) with SMAP reference threshold.

    Args:
        dates: List of datetime objects.
        rmse_p1: Per-date RMSE values for pixel 1 in m3/m3.
        rmse_p2: Per-date RMSE values for pixel 2 in m3/m3.
        output_path: Path to save the output figure.
    """
    if len(dates) != rmse_p1.size or len(dates) != rmse_p2.size:
        raise ValueError("dates, rmse_p1, and rmse_p2 must have matching lengths.")

    plt.figure()
    plt.plot(dates, rmse_p1, marker="o", label="Pixel 1 RMSE (per date)")
    plt.plot(dates, rmse_p2, marker="o", label="Pixel 2 RMSE (per date)")

    plt.axhline(
        SMAP_EXPECTED_RMSE,
        linestyle="--",
        label="SMAP expected RMSE (0.04 m³/m³)",
    )

    plt.xlabel("Date")
    plt.ylabel("RMSE (m³/m³)")
    plt.title("RMSE of Fine-Resolution Soil Moisture vs Time (Per Pixel)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.show()


def main() -> None:
    """Main execution function."""
    date_strings = ["20150607", "20150610", "20150615", "20150618", "20150620"]
    dates = parse_dates(date_strings)

    # Pixel 1 soil moisture values (%)
    sm1_est = np.array([10.8, 8.7, 9.0, 10.0, 8.0])
    sm1_ref = np.array([13.16, 8.5, 5.6, 14.0, 5.0])

    # Pixel 2 soil moisture values (%)
    sm2_est = np.array([22.0, 14.8, 16.8, 13.5, 12.3])
    sm2_ref = np.array([26.0, 14.0, 18.3, 16.0, 15.0])

    rmse_p1, rmse_p2 = compute_two_pixel_rmse_time_series(
        sm1_est,
        sm1_ref,
        sm2_est,
        sm2_ref,
    )

    plot_two_pixel_rmse_time_series(
        dates,
        rmse_p1,
        rmse_p2,
        output_path="rmse_groundtruth_pixel1_pixel2.png",
    )


if __name__ == "__main__":
    main()
