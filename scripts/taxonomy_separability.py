"""large_vehicle / truck / bus taxonomy check: is merging them (Design_Decisions.md decision 1)
actually justified? Builds per-instance features (rcs, vr_compensated, x_extent, y_extent,
doppler_spread) and runs the separability_probe.run_probe class-weighted LR + RF probe on a
sequence-grouped held-out split, with per-class one-vs-rest and pairwise ROC-AUC.
"""
import pandas as pd

from dataloader import LABELS, RESULTS_DIR
from separability_probe import run_probe

TAXONOMY_CLASSES = ["large_vehicle", "truck", "bus"]
PROBE_FEATURES = ["rcs", "vr_compensated", "x_extent", "y_extent", "doppler_spread"]
NAME_TO_COLOR = {name: color for name, color in LABELS.values()}


INSTANCE_COLS = ["sequence_name", "timestamp", "track_id"]
DOPPLER_SPREAD_CACHE = RESULTS_DIR / "doppler_spread_cache.parquet"


def add_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add x_rel/y_rel (point position relative to its instance's centroid, mean-centered)
    and doppler_spread (per-instance median absolute deviation of vr_compensated). doppler_spread
    is the expensive part (~2-3 min at full scale, one Python lambda per instance group), so it's
    cached to disk keyed by sequence_name and reused if the cache covers the same sequences -
    same skip-if-covered pattern as build_points_table.py's cache."""
    df = df.copy()
    group = df.groupby(INSTANCE_COLS)
    df["x_rel"] = df["x_cc"] - group["x_cc"].transform("mean")
    df["y_rel"] = df["y_cc"] - group["y_cc"].transform("mean")

    sequences = set(df["sequence_name"].unique())
    if DOPPLER_SPREAD_CACHE.exists():
        cached = pd.read_parquet(DOPPLER_SPREAD_CACHE)
        if set(cached["sequence_name"].unique()) == sequences:
            return df.merge(cached, on=INSTANCE_COLS, how="left")
        print(f"{DOPPLER_SPREAD_CACHE} covers different sequences than requested, recomputing doppler_spread")

    df["doppler_spread"] = group["vr_compensated"].transform(lambda vr: (vr - vr.median()).abs().median())

    RESULTS_DIR.mkdir(exist_ok=True)
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
    return run_probe(X, y, groups, classes, n_splits=n_splits, random_state=random_state, tag="taxonomy")


if __name__ == "__main__":
    from build_points_table import build_and_save_points_table

    df = build_and_save_points_table()
    run_separability_probe(df)
