"""Shared per-instance histogram feature extraction (Day 6 - see MLP_DESIGN.md).

Turns an instance's raw points into the concatenated M*K histogram feature
vector used as MLP input. Bin edges are fit on the train split only (no
val/test leakage) and reused everywhere else. This is the real pipeline code -
scripts/instance_histograms.py and scripts/bin_sweep.py are diagnostic only,
not this.
"""
import numpy as np
import pandas as pd

GROUP_KEY = ["sequence_name", "timestamp", "track_id"]

# M=5 features, matching the paper's own set (see DESIGN_DECISIONS.md decision 3
# and MLP_DESIGN.md) - rcs/vr_compensated/range_sc raw, x/y object-centered
# (relative to each instance's own mean, not median - see DESIGN_DECISIONS.md).
FEATURES = ["rcs", "vr_compensated", "range_sc", "x_rel", "y_rel"]
N_BINS = 16

# vr_rel: vr_compensated relative to each instance's own median (Doppler spread /
# micro-Doppler) - deferred from the v1 baseline (MLP_DESIGN.md), real signal
# validated in FEATURE_MAP.md, opt-in via include_vr_rel below (MLP_FINDINGS.md).
# Not in FEATURES - kept separate so every existing caller (len(FEATURES)*N_BINS
# for a saved 80-dim checkpoint) is unaffected unless it explicitly opts in.
VR_REL_RANGE = (-3.0, 3.0)  # explicit override, not fit from data (FEATURE_MAP.md "vr_rel:
                            # two real bugs caught building it", bug #2) - the percentile-fit
                            # range other features use gets blown out to ~-15..+30 m/s by rare
                            # fast-crossing traffic/aliasing elsewhere in the dataset, which
                            # would wash out the real sub-1-m/s class gap into "zero, dead
                            # center" for every class against bins that wide.


def add_relative_xy(df: pd.DataFrame) -> pd.DataFrame:
    """Adds x_rel/y_rel: x_cc/y_cc relative to each instance's own mean position
    (object-centered, matches the paper - see DESIGN_DECISIONS.md decision 3)."""
    df = df.copy()
    df["x_rel"] = df["x_cc"] - df.groupby(GROUP_KEY)["x_cc"].transform("mean")
    df["y_rel"] = df["y_cc"] - df.groupby(GROUP_KEY)["y_cc"].transform("mean")
    return df


def add_relative_vr(df: pd.DataFrame) -> pd.DataFrame:
    """Adds vr_rel: vr_compensated relative to each instance's own MEDIAN (not
    mean - median-centering avoids one aliased/outlier point dragging the whole
    instance's reference off and spuriously widening every other point's
    relative value too, FEATURE_MAP.md "vr_rel: two real bugs", bug #1)."""
    df = df.copy()
    df["vr_rel"] = df["vr_compensated"] - df.groupby(GROUP_KEY)["vr_compensated"].transform("median")
    return df


def fit_bin_edges(train_df: pd.DataFrame, include_vr_rel: bool = False) -> dict[str, np.ndarray]:
    """Percentile-based (0.5-99.5%) global bin edges per feature, fit on the
    train split only. Reused as-is for val/test and for any train subset -
    never refit on them (see MLP_DESIGN.md section 0). vr_rel is the one
    exception - an explicit range (VR_REL_RANGE), not fit from data."""
    edges = {}
    for col in FEATURES:
        lo, hi = train_df[col].quantile(0.005), train_df[col].quantile(0.995)
        edges[col] = np.linspace(lo, hi, N_BINS + 1)
    if include_vr_rel:
        edges["vr_rel"] = np.linspace(VR_REL_RANGE[0], VR_REL_RANGE[1], N_BINS + 1)
    return edges


def instance_vector(points: pd.DataFrame, bin_edges: dict[str, np.ndarray],
                     include_n_points: bool = False, include_vr_rel: bool = False,
                     replace_vr_with_vr_rel: bool = False) -> np.ndarray:
    """One instance's points -> concatenated histogram vector (raw counts, not
    density - see MLP_DESIGN.md). Layout depends on the vr_rel flags:
    include_vr_rel=True alone APPENDS a 6th vr_rel block after the 5 FEATURES
    blocks (81/96-dim, tests vr_rel as added information); with
    replace_vr_with_vr_rel=True too, vr_rel instead SWAPS IN for
    vr_compensated in place (still 5 blocks, 80-dim, tests vr_rel as a
    replacement rather than an addition - MLP_FINDINGS.md section 9). A scalar
    n_points, if included, is always appended last. All three flags are off by
    default since they change the vector's dimensionality."""
    if include_vr_rel and replace_vr_with_vr_rel:
        cols = [c if c != "vr_compensated" else "vr_rel" for c in FEATURES]
    elif include_vr_rel:
        cols = list(FEATURES) + ["vr_rel"]
    else:
        cols = list(FEATURES)
    vec = np.concatenate(
        [np.histogram(points[col], bins=bin_edges[col])[0] for col in cols]
    ).astype(np.float32)
    if include_n_points:
        vec = np.append(vec, len(points)).astype(np.float32)
    return vec


def build_dataset(df: pd.DataFrame, bin_edges: dict[str, np.ndarray], classes: list[str],
                   include_n_points: bool = False, include_vr_rel: bool = False,
                   replace_vr_with_vr_rel: bool = False) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Full (X, y, keys) for every instance in df. y is an integer class index
    into `classes` (same order as MLP_DESIGN.md's output layer). keys is a
    DataFrame (GROUP_KEY columns, same row order as X/y) so a prediction can be
    traced back to that instance's raw points later - see dataloader.py's
    plot_predictions_grid."""
    class_to_idx = {name: i for i, name in enumerate(classes)}
    X, y, keys = [], [], []
    for key, points in df.groupby(GROUP_KEY):
        X.append(instance_vector(points, bin_edges, include_n_points, include_vr_rel,
                                  replace_vr_with_vr_rel))
        y.append(class_to_idx[points["class_name"].iloc[0]])
        keys.append(key)
    keys_df = pd.DataFrame(keys, columns=GROUP_KEY)
    return np.stack(X), np.array(y, dtype=np.int64), keys_df
