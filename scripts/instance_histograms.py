"""Per-instance feature histograms (Day 5) - the actual histogram-encoding scheme
from Tatarchenko & Rambach (see TODO.md Day 8): for each object instance, compute
one histogram per feature over that instance's own points (raw point counts per
bin, not density - matches the paper, and lets bin counts implicitly encode
instance size/point count, not just shape).

Uses ground-truth track_id grouping to build training examples - this does not
solve the real-world clustering gap (see TODO.md's v2 future work), it only
answers "does this histogram representation look separable" for a model trained
on already-associated instances, same scope as the paper itself.
"""
import numpy as np

import matplotlib
import pandas as pd

matplotlib.use("Agg")  # headless: this script only ever saves plots, never shows them
import matplotlib.pyplot as plt  # noqa: E402

from dataloader import GROUP_COLORS, RESULTS_DIR  # noqa: E402

HIST_DIR = RESULTS_DIR / "instance_histograms"

CLASSES = ["car", "large_vehicle", "pedestrian", "pedestrian_group", "two_wheeler"]

FEATURES = {
    "rcs": "RCS [dBsm]",
    "vr_compensated": "radial velocity, ego-compensated [m/s]",
    "range_sc": "range [m]",
    "azimuth_sc": "azimuth [rad]",
    "x_cc": "x position, sensor-centered [m]",
    "y_cc": "y position, sensor-centered [m]",
}

# Extension: same features, but relative to each instance's own mean instead of
# raw/absolute. Fixes two separate problems the raw histograms above have: (1)
# range_sc/azimuth_sc/x_cc/y_cc raw values mostly encode where in the world the
# object was recorded (environment), not its shape - de-meaning per instance
# recenters on the object itself, same fix Tatarchenko & Rambach use (object-
# centered Cartesian coords); (2) vr_compensated raw is dominated by bulk/
# behavioral velocity (item #2's "redundant, dropped" verdict) - de-meaning
# recovers Doppler *spread*, the micro-Doppler feature from EDA.md item #4, now
# as a full histogram shape instead of a single std/MAD scalar. rcs_rel and
# range_rel are bonus additions in the same spirit (RCS spread ties to item #1's
# bus/specular-glint observation; range spread could hint at object orientation).
RELATIVE_FEATURES = {
    "rcs_rel": "RCS relative to instance mean (RCS spread) [dB]",
    "vr_rel": "radial velocity relative to instance median (Doppler spread) [m/s]",
    "range_rel": "range relative to instance mean (range spread) [m]",
    "azimuth_rel": "azimuth relative to instance mean (azimuth spread) [rad]",
    "x_rel": "x relative to instance centroid (spatial extent) [m]",
    "y_rel": "y relative to instance centroid (spatial extent) [m]",
    "extent_rel": "distance from instance centroid (rotation-invariant spatial extent) [m]",
}
RAW_TO_REL = {
    "rcs_rel": "rcs",
    "vr_rel": "vr_compensated",
    "range_rel": "range_sc",
    "azimuth_rel": "azimuth_sc",
    "x_rel": "x_cc",
    "y_rel": "y_cc",
}
# Centering method per relative feature - median for vr_rel since a single
# aliased/wrapped-around point (item #4) pulls a mean far more than a median,
# same reasoning as preferring MAD over std. Everything else uses mean - no
# similar single-point-dominated failure mode established for those features.
CENTERING = {"vr_rel": "median"}

# Explicit (lo, hi) override for features where the global 0.5-99.5 percentile
# range is too wide to resolve the real signal - vr_rel's actual class gap is
# sub-1 m/s (item #4: car median 0.08, pedestrian median 0.52), but the
# dataset's genuine wide-spread outliers stretch the percentile range enough
# that 8 bins over it are ~5.6 m/s wide, burying the real difference in one
# bin. Clip to +-3 m/s (same cutoff used for the vr_std <= 3.0 scatter, 02g)
# so bin width actually matches the scale of the signal.
EXPLICIT_RANGES = {"vr_rel": (-3.0, 3.0)}

N_BINS = 8
N_EXAMPLES = 8
MIN_POINTS = 5  # for visualization only - most instances are sparser than this (EDA.md item #4)


def compute_bin_edges(df: pd.DataFrame, col: str, n_bins: int) -> np.ndarray:
    """Global (dataset-wide) bin edges - percentile-based by default so
    aliasing/RCS outliers don't stretch the range, or an explicit override
    (EXPLICIT_RANGES) where percentile-based bins would be too coarse to
    resolve the real signal. Same edges used for every instance's histogram
    so they're directly comparable to each other."""
    if col in EXPLICIT_RANGES:
        lo, hi = EXPLICIT_RANGES[col]
    else:
        lo, hi = df[col].quantile(0.005), df[col].quantile(0.995)
    return np.linspace(lo, hi, n_bins + 1)


def plot_feature_examples(examples: dict, edges: np.ndarray, col: str, xlabel: str) -> None:
    """Mean histogram (± std across example instances) as bars, one column set
    per class - overlaying each instance as its own step outline (the previous
    version) produced a cluttered stack of many small overlapping rectangles
    instead of a readable shape. Averaging across the sampled instances gives
    one clean bar chart per class while still showing instance-to-instance
    variability via the error bars."""
    bin_mids = (edges[:-1] + edges[1:]) / 2
    bin_width = (edges[1] - edges[0]) * 0.85

    fig, axes = plt.subplots(1, len(CLASSES), figsize=(4 * len(CLASSES), 4), sharey=True)
    for ax, name in zip(axes, CLASSES):
        color = GROUP_COLORS[name]
        all_counts = np.stack([np.histogram(g[col], bins=edges)[0] for g in examples[name]])
        mean_counts = all_counts.mean(axis=0)
        std_counts = all_counts.std(axis=0)
        ax.bar(bin_mids, mean_counts, width=bin_width, yerr=std_counts, color=color,
               alpha=0.7, edgecolor="black", linewidth=0.8, error_kw={"elinewidth": 1, "capsize": 2})
        ax.set_title(f"{name}\n(mean of {len(examples[name])} instances)", fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.tick_params(labelsize=7)

    axes[0].set_ylabel("mean point count per bin (± std across instances)")
    fig.suptitle(f"Per-instance histogram of {col} - mean shape across example instances ({N_BINS} bins)")
    fig.tight_layout()
    path = HIST_DIR / f"{col}_instance_examples.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")


if __name__ == "__main__":
    HIST_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(RESULTS_DIR / "train_points.parquet")
    print(f"Loaded {len(df)} points from train_points.parquet")

    group_key = ["sequence_name", "timestamp", "track_id"]
    for rel_col, raw_col in RAW_TO_REL.items():
        method = CENTERING.get(rel_col, "mean")
        df[rel_col] = df[raw_col] - df.groupby(group_key)[raw_col].transform(method)
    df["extent_rel"] = np.hypot(df["x_rel"], df["y_rel"])

    rng = np.random.default_rng(42)
    groups = list(df.groupby(group_key))
    print(f"{len(groups)} instances")

    examples: dict[str, list[pd.DataFrame]] = {name: [] for name in CLASSES}
    for _, g in groups:
        name = g["class_name"].iloc[0]
        if name in examples and len(g) >= MIN_POINTS:
            examples[name].append(g)

    for name in CLASSES:
        pool = examples[name]
        idx = rng.choice(len(pool), size=min(N_EXAMPLES, len(pool)), replace=False)
        examples[name] = [pool[i] for i in idx]
        print(f"{name}: {len(pool)} instances with n_points >= {MIN_POINTS}, showing {len(examples[name])}")

    for col, xlabel in FEATURES.items():
        edges = compute_bin_edges(df, col, N_BINS)
        plot_feature_examples(examples, edges, col, xlabel)

    for col, xlabel in RELATIVE_FEATURES.items():
        edges = compute_bin_edges(df, col, N_BINS)
        plot_feature_examples(examples, edges, col, xlabel)
