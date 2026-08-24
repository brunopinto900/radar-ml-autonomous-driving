"""large_vehicle / truck / bus taxonomy check: is merging them (Design_Decisions.md decision 1)
actually justified? Builds per-instance features (rcs, vr_compensated, x_extent, y_extent,
doppler_spread) and runs the separability_probe.run_probe class-weighted LR + RF probe on a
sequence-grouped held-out split, with per-class one-vs-rest and pairwise ROC-AUC.
"""
import pandas as pd

from dataloader import LABELS, RESULTS_DIR
from separability_probe import run_probe, run_probe_cv

TAXONOMY_CLASSES = ["large_vehicle", "truck", "bus"]
PROBE_FEATURES = ["rcs", "vr_compensated", "x_extent", "y_extent", "doppler_spread"]
NAME_TO_COLOR = {name: color for name, color in LABELS.values()}


INSTANCE_COLS = ["sequence_name", "timestamp", "track_id"]
TAXONOMY_DIR = RESULTS_DIR / "taxonomy"
DOPPLER_SPREAD_CACHE = RESULTS_DIR / "data" / "doppler_spread_cache.parquet"


def add_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add x_rel/y_rel (point position relative to its instance's centroid, mean-centered),
    n_points (instance's raw point count, broadcast to every one of its rows - cheap groupby
    size, no caching needed unlike doppler_spread below), spatial_extent (instance's bounding-box
    diagonal, sqrt(x_extent^2 + y_extent^2) where extent is max-min per axis - unlike x_rel/y_rel,
    which are per-point and orientation-dependent (a car seen broadside vs head-on spreads very
    differently across the x/y axes even though it's the same physical object), this is one
    orientation-robust scalar per instance: rotating the point cloud changes x_extent/y_extent
    individually but their combined diagonal is far more stable), and doppler_spread (per-instance
    median absolute deviation of vr_compensated). doppler_spread is the expensive part (~2-3 min
    at full scale, one Python lambda per instance group), so it's cached to disk keyed by
    sequence_name and reused if the cache covers the same sequences - same skip-if-covered
    pattern as build_points_table.py's cache."""
    df = df.copy()
    group = df.groupby(INSTANCE_COLS)
    df["x_rel"] = df["x_cc"] - group["x_cc"].transform("mean")
    df["y_rel"] = df["y_cc"] - group["y_cc"].transform("mean")
    df["n_points"] = group["x_cc"].transform("size")
    x_extent = group["x_cc"].transform(lambda s: s.max() - s.min())
    y_extent = group["y_cc"].transform(lambda s: s.max() - s.min())
    df["spatial_extent"] = (x_extent**2 + y_extent**2) ** 0.5

    sequences = set(df["sequence_name"].unique())
    if DOPPLER_SPREAD_CACHE.exists():
        cached = pd.read_parquet(DOPPLER_SPREAD_CACHE)
        if set(cached["sequence_name"].unique()) == sequences:
            return df.merge(cached, on=INSTANCE_COLS, how="left")
        print(f"{DOPPLER_SPREAD_CACHE} covers different sequences than requested, recomputing doppler_spread")

    df["doppler_spread"] = group["vr_compensated"].transform(lambda vr: (vr - vr.median()).abs().median())

    DOPPLER_SPREAD_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.drop_duplicates(INSTANCE_COLS)[INSTANCE_COLS + ["doppler_spread"]].to_parquet(DOPPLER_SPREAD_CACHE)
    print(f"Saved doppler_spread cache to {DOPPLER_SPREAD_CACHE}")
    return df


def doppler_spread_diagnostics(df: pd.DataFrame, classes: list[str] = TAXONOMY_CLASSES) -> pd.DataFrame:
    """raw doppler_spread histogram is unreadable due to extreme values: how much
    of the zero-spike is single-point instances (spread is exactly 0 for a lone point by
    construction, not an outlier being suppressed)"""
    df = add_relative_features(df)
    sizes = df.groupby(["sequence_name", "timestamp", "track_id"]).size()
    sizes.name = "n_points"
    instances = df.drop_duplicates(["sequence_name", "timestamp", "track_id"])
    instances = instances.merge(sizes, on=["sequence_name", "timestamp", "track_id"])

    rows = []
    for cls in classes:
        sub = instances.loc[instances["label_name"] == cls]
        rows.append({
            "class": cls,
            "n": len(sub),
            "single_point_pct": 100 * (sub["n_points"] == 1).mean(),
            "zero_spread_pct": 100 * (sub["doppler_spread"] == 0).mean(),
            "max": sub["doppler_spread"].max(),
            "p99": sub["doppler_spread"].quantile(0.99),
        })

    summary = pd.DataFrame(rows).set_index("class")
    print(summary.round(2).to_string())
    return summary


def build_instance_features(df: pd.DataFrame, classes: list[str] = TAXONOMY_CLASSES) -> pd.DataFrame:
    """One feature row per instance: rcs/vr_compensated aggregated to their per-instance
    median (central tendency), x_rel/y_rel aggregated to their per-instance extent
    (max - min, i.e. footprint size) - their per-instance mean is trivially ~0 by
    construction since x_rel/y_rel are already mean-centered, so extent is used instead -
    plus doppler_spread (already one value per instance)."""
    df = df.loc[df["label_name"].isin(classes)]
    df = add_relative_features(df)

    group = df.groupby(["sequence_name", "timestamp", "track_id"])
    features = group.agg(
        rcs=("rcs", "median"),
        vr_compensated=("vr_compensated", "median"),
        x_extent=("x_rel", lambda s: s.max() - s.min()),
        y_extent=("y_rel", lambda s: s.max() - s.min()),
        doppler_spread=("doppler_spread", "first"),
        label_name=("label_name", "first"),
    ).reset_index()
    return features


def run_separability_probe(
    df: pd.DataFrame, classes: list[str] = TAXONOMY_CLASSES, n_splits: int = 5, random_state: int = 0
):
    """Class-count-weighted logistic regression AND random forest on [rcs, vr_compensated,
    x_extent, y_extent, doppler_spread], sequence-grouped split. See run_probe for the
    shared split/train/eval mechanics."""
    features = build_instance_features(df, classes)
    X = features[PROBE_FEATURES].to_numpy(dtype="float64")
    y = features["label_name"].to_numpy(dtype=str)
    groups = features["sequence_name"].to_numpy(dtype=str)
    return run_probe(
        X, y, groups, classes, n_splits=n_splits, random_state=random_state, tag="taxonomy", results_dir=TAXONOMY_DIR
    )


def plot_taxonomy_class_balance(df: pd.DataFrame, classes: list[str] = TAXONOMY_CLASSES):
    """Instance counts vs. sequence coverage for the taxonomy classes, restricted to the fixed
    split's train+val sequences (test excluded, same scope as run_separability_probe_cv) - the
    two aren't the same thing. large_vehicle can look only moderately rare by instance count
    while being far rarer by sequence coverage (only 23 of 130 train+val sequences ever contain
    one), which is what actually drives its noisy CV fold-to-fold variance. Saves
    results/taxonomy_class_balance.png. Returns (summary, fig)."""
    import matplotlib.pyplot as plt

    from sequence_split import load_split

    splits = load_split()
    trainval_sequences = set(splits["train"]) | set(splits["val"])

    instances = df.loc[df["label_name"].isin(classes) & df["sequence_name"].isin(trainval_sequences)]
    instances = instances.drop_duplicates(INSTANCE_COLS)

    rows = []
    for cls in classes:
        sub = instances.loc[instances["label_name"] == cls]
        rows.append({"class": cls, "n_instances": len(sub), "n_sequences": sub["sequence_name"].nunique()})
    summary = pd.DataFrame(rows).set_index("class")
    print(f"{len(trainval_sequences)} train+val sequences total")
    print(summary.to_string())

    fig, (ax_inst, ax_seq) = plt.subplots(1, 2, figsize=(10, 4.5))
    colors = [NAME_TO_COLOR[cls] for cls in classes]

    ax_inst.bar(classes, summary["n_instances"], color=colors, edgecolor="k")
    ax_inst.set_yscale("log")
    ax_inst.set_ylabel("instance count (log scale)")
    ax_inst.set_title("instance count")

    ax_seq.bar(classes, summary["n_sequences"], color=colors, edgecolor="k")
    ax_seq.axhline(len(trainval_sequences), color="gray", linestyle="--", linewidth=1, label="train+val total")
    ax_seq.set_ylabel("sequences containing >=1 instance")
    ax_seq.set_title("sequence coverage")
    ax_seq.legend(fontsize=8)

    fig.suptitle("large_vehicle / truck / bus - train+val scope")
    fig.tight_layout()

    TAXONOMY_DIR.mkdir(parents=True, exist_ok=True)
    path = TAXONOMY_DIR / "taxonomy_class_balance.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")

    return summary, fig


def plot_cv_fold_class_counts(
    df: pd.DataFrame, classes: list[str] = TAXONOMY_CLASSES, n_splits: int = 5, random_state: int = 0
):
    """Per-fold instance counts in the held-out test portion of each of the 5 CV folds used by
    run_separability_probe_cv (same train+val scope, same StratifiedGroupKFold(n_splits,
    random_state), so these are the exact folds that produced its numbers). This is the concrete
    picture behind the CV std: e.g. large_vehicle's count swinging from fold to fold directly
    explains why its recall/AUC std was so large in Design_Decisions.md decision 1's CV
    confirmation, in a way the train+val-total chart (plot_taxonomy_class_balance) can't show.
    Saves results/taxonomy_cv_fold_counts.png. Returns (summary, fig)."""
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.model_selection import StratifiedGroupKFold

    from sequence_split import load_split

    splits = load_split()
    trainval_sequences = set(splits["train"]) | set(splits["val"])

    features = build_instance_features(df, classes)
    features = features.loc[features["sequence_name"].isin(trainval_sequences)]
    y = features["label_name"].to_numpy(dtype=str)
    groups = features["sequence_name"].to_numpy(dtype=str)

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows = []
    for fold, (_, test_idx) in enumerate(splitter.split(np.zeros(len(y)), y, groups)):
        y_test = y[test_idx]
        rows.append({"fold": fold + 1, **{cls: int((y_test == cls).sum()) for cls in classes}})
    summary = pd.DataFrame(rows).set_index("fold")
    print("per-fold test-set instance counts:")
    print(summary.to_string())

    fig, ax = plt.subplots(figsize=(8, 5))
    summary[classes].plot(kind="bar", ax=ax, color=[NAME_TO_COLOR[c] for c in classes], edgecolor="k")
    ax.set_xlabel("CV fold (held-out test portion)")
    ax.set_ylabel("instance count")
    ax.set_title("large_vehicle / truck / bus - per-fold test-set counts (5-fold CV)")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()

    TAXONOMY_DIR.mkdir(parents=True, exist_ok=True)
    path = TAXONOMY_DIR / "taxonomy_cv_fold_counts.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")

    return summary, fig


def run_separability_probe_cv(
    df: pd.DataFrame, classes: list[str] = TAXONOMY_CLASSES, n_splits: int = 5, random_state: int = 0
):
    """Same features as run_separability_probe, but restricted to the fixed split's train+val
    sequences (results/sequence_split.json via sequence_split.load_split - test stays untouched)
    and scored with separability_probe.run_probe_cv's proper n_splits-fold averaging instead of
    a single fold. This is the version whose numbers should actually be compared/reported -
    run_separability_probe's single-fold numbers (still in Design_Decisions.md decision 1 as the
    original evidence) are noisier, per decision 5. Imports sequence_split lazily to avoid a
    circular import (sequence_split -> feature_distributions -> taxonomy_separability)."""
    from sequence_split import load_split

    splits = load_split()
    trainval_sequences = set(splits["train"]) | set(splits["val"])

    features = build_instance_features(df, classes)
    features = features.loc[features["sequence_name"].isin(trainval_sequences)]
    print(f"{len(features)} instances across {features['sequence_name'].nunique()} sequences (train+val only)")

    X = features[PROBE_FEATURES].to_numpy(dtype="float64")
    y = features["label_name"].to_numpy(dtype=str)
    groups = features["sequence_name"].to_numpy(dtype=str)
    return run_probe_cv(X, y, groups, classes, n_splits=n_splits, random_state=random_state)


if __name__ == "__main__":
    from build_points_table import build_and_save_points_table

    df = build_and_save_points_table()
    run_separability_probe_cv(df)
