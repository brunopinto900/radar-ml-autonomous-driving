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


def add_relative_xy(df: pd.DataFrame) -> pd.DataFrame:
    """Adds x_rel/y_rel: x_cc/y_cc relative to each instance's own mean position
    (object-centered, matches the paper - see DESIGN_DECISIONS.md decision 3)."""
    df = df.copy()
    df["x_rel"] = df["x_cc"] - df.groupby(GROUP_KEY)["x_cc"].transform("mean")
    df["y_rel"] = df["y_cc"] - df.groupby(GROUP_KEY)["y_cc"].transform("mean")
    return df


def fit_bin_edges(train_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Percentile-based (0.5-99.5%) global bin edges per feature, fit on the
    train split only. Reused as-is for val/test and for any train subset -
    never refit on them (see MLP_DESIGN.md section 0)."""
    edges = {}
    for col in FEATURES:
        lo, hi = train_df[col].quantile(0.005), train_df[col].quantile(0.995)
        edges[col] = np.linspace(lo, hi, N_BINS + 1)
    return edges


def instance_vector(points: pd.DataFrame, bin_edges: dict[str, np.ndarray]) -> np.ndarray:
    """One instance's points -> concatenated M*K histogram vector (raw counts,
    not density - see MLP_DESIGN.md)."""
    return np.concatenate(
        [np.histogram(points[col], bins=bin_edges[col])[0] for col in FEATURES]
    ).astype(np.float32)


def build_dataset(df: pd.DataFrame, bin_edges: dict[str, np.ndarray],
                   classes: list[str]) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Full (X, y, keys) for every instance in df. y is an integer class index
    into `classes` (same order as MLP_DESIGN.md's output layer). keys is a
    DataFrame (GROUP_KEY columns, same row order as X/y) so a prediction can be
    traced back to that instance's raw points later - see dataloader.py's
    plot_predictions_grid."""
    class_to_idx = {name: i for i, name in enumerate(classes)}
    X, y, keys = [], [], []
    for key, points in df.groupby(GROUP_KEY):
        X.append(instance_vector(points, bin_edges))
        y.append(class_to_idx[points["class_name"].iloc[0]])
        keys.append(key)
    keys_df = pd.DataFrame(keys, columns=GROUP_KEY)
    return np.stack(X), np.array(y, dtype=np.int64), keys_df
