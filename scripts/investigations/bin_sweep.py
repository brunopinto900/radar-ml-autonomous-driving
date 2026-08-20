"""Bin-count sweep for the paper-faithful feature set (Day 6 - see TODO.md).

For each feature, one figure with a grid of subplots - the same per-class
overlaid density histogram at increasing bin counts, side by side. Answers
"where does finer resolution stop revealing new real class-separating
structure and start just showing sampling noise" - the natural resolution
ceiling of the feature itself.

Deliberately population-level (all points, all instances, pooled), not
per-instance: individual instances are often single digits of points (see
EDA.md item #4 - pedestrian is >54% single-point scans) and can't meaningfully
support many bins on their own - that's a sample-size question, answered
separately by how many points a given instance actually has. This script
answers a different question: how finely can the *feature itself* usefully be
binned, given the whole dataset's worth of data.

Uses results/train_points.parquet only, same split discipline as eda.py.
"""
import numpy as np

import matplotlib
import pandas as pd

matplotlib.use("Agg")  # headless: this script only ever saves plots, never shows them
import matplotlib.pyplot as plt  # noqa: E402

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/ core modules

from dataloader import GROUP_COLORS, RESULTS_DIR  # noqa: E402

BIN_SWEEP_DIR = RESULTS_DIR / "bin_sweep"

CLASSES = ["car", "large_vehicle", "pedestrian", "pedestrian_group", "two_wheeler"]

# The paper's own feature set (Tatarchenko & Rambach, see TODO.md Day 6/8):
# radial distance, ego-motion-compensated Doppler velocity, RCS, and
# object-centered Cartesian x/y/z (2D here - sensor #2 only, per
# DESIGN_DECISIONS.md). x/y are relative to each instance's own centroid to
# match their choice - see FEATURE_MAP.md for why (shape independent of range).
FEATURES = {
    "rcs": "RCS [dBsm]",
    "vr_compensated": "radial velocity, ego-compensated [m/s]",
    "range_sc": "range [m]",
    "x_rel": "x relative to instance centroid (object-centered) [m]",
    "y_rel": "y relative to instance centroid (object-centered) [m]",
}

BIN_COUNTS = [8, 16, 32, 64]


def plot_bin_sweep(df: pd.DataFrame, col: str, xlabel: str) -> None:
    class_data = df.loc[df["class_name"].isin(CLASSES), col]
    x_range = (class_data.quantile(0.005), class_data.quantile(0.995))

    ncols = 2
    nrows = -(-len(BIN_COUNTS) // ncols)  # ceil
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.5 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, n_bins in zip(axes, BIN_COUNTS):
        for name in CLASSES:
            color = GROUP_COLORS[name]
            sub = df.loc[df["class_name"] == name, col]
            ax.hist(sub, bins=n_bins, range=x_range, density=True, histtype="step",
                     linewidth=1.5, color=color, label=name)
        ax.set_title(f"{n_bins} bins", fontsize=10)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("density", fontsize=8)
        ax.tick_params(labelsize=7)

    for ax in axes[len(BIN_COUNTS):]:
        ax.axis("off")

    axes[0].legend(fontsize=7)
    fig.suptitle(f"Bin-count sweep: {col} (per point, pooled, by class)")
    fig.tight_layout()
    path = BIN_SWEEP_DIR / f"{col}_bin_sweep.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")


if __name__ == "__main__":
    BIN_SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(RESULTS_DIR / "train_points.parquet")
    print(f"Loaded {len(df)} points from train_points.parquet")

    group_key = ["sequence_name", "timestamp", "track_id"]
    df["x_rel"] = df["x_cc"] - df.groupby(group_key)["x_cc"].transform("mean")
    df["y_rel"] = df["y_cc"] - df.groupby(group_key)["y_cc"].transform("mean")

    for col, xlabel in FEATURES.items():
        plot_bin_sweep(df, col, xlabel)
