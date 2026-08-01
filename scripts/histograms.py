"""Per-point feature histograms, for designing Day 5's binning scheme.

Unlike eda.py (which validated per-instance/per-scan features via track_id
grouping), this operates on raw per-point attributes only - the classifier is
point-wise (see EDA.md/TODO.md discussion), so track_id, n_points, extent, and
vr_std/mad are not available at inference and aren't considered here.
Uses results/train_points.parquet only, same split discipline as eda.py.
"""
import numpy as np

import matplotlib
import pandas as pd

matplotlib.use("Agg")  # headless: this script only ever saves plots, never shows them
import matplotlib.pyplot as plt  # noqa: E402

from dataloader import GROUP_COLORS, RESULTS_DIR  # noqa: E402

HIST_DIR = RESULTS_DIR / "histograms"

CLASSES = ["car", "large_vehicle", "pedestrian", "pedestrian_group", "two_wheeler"]

FEATURES = {
    "rcs": "RCS [dBsm]",
    "vr_compensated": "radial velocity, ego-compensated [m/s]",
    "range_sc": "range [m]",
    "azimuth_sc": "azimuth [rad]",
    "x_cc": "x position, sensor-centered [m]",
    "y_cc": "y position, sensor-centered [m]",
}


def plot_feature_histogram(df: pd.DataFrame, col: str, xlabel: str) -> None:
    """One overlaid density histogram per class, percentile-based x-range so
    outliers (e.g. Doppler aliasing) don't squash the bulk of the distribution."""
    class_data = df.loc[df["class_name"].isin(CLASSES), col]
    x_range = (class_data.quantile(0.005), class_data.quantile(0.995))

    fig, ax = plt.subplots(figsize=(9, 5))
    for name in CLASSES:
        color = GROUP_COLORS[name]
        sub = df.loc[df["class_name"] == name, col]
        ax.hist(sub, bins=50, range=x_range, density=True, histtype="step",
                 linewidth=1.8, color=color, label=name)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.set_title(f"{col} by final class (per point)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = HIST_DIR / f"{col}.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")


if __name__ == "__main__":
    HIST_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(RESULTS_DIR / "train_points.parquet")
    print(f"Loaded {len(df)} points from train_points.parquet")

    for col, xlabel in FEATURES.items():
        plot_feature_histogram(df, col, xlabel)
