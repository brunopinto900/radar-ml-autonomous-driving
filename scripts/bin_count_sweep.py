"""Day 5 (cont'd): decide the per-instance histogram encoding's bin count empirically, by
running it through the same sequence-grouped LR+RF separability probe used for the taxonomy
decision - not by eyeballing the pooled bin-sweep plots in feature_distributions.py. Point-level
features average only ~2.9 points/instance (1,443,816 points / 503,759 instances), so "which
plot looks best" doesn't reflect what a typical instance's histogram actually looks like; bin
count is a hyperparameter of the encoding, scored the same way any other one is.
"""
import numpy as np
import pandas as pd

from feature_distributions import FINAL_CLASSES, POINT_LEVEL_FEATURES, apply_class_groups
from taxonomy_separability import add_relative_features, run_probe

INSTANCE_COLS = ["sequence_name", "timestamp", "track_id"]


def build_histogram_features(
    df: pd.DataFrame, n_bins: int, classes: list[str] = FINAL_CLASSES, features: list[str] = POINT_LEVEL_FEATURES
) -> pd.DataFrame:
    """One row per instance: each point-level feature becomes n_bins columns holding the
    fraction of that instance's own points landing in each bin - fraction, not raw count, so a
    busy instance's histogram encodes distribution shape rather than just point count. Bin
    edges are fixed at the pooled [1st, 99th] percentile for that feature (matches the bin-sweep
    plots' range fix), same edges across every n_bins so only resolution changes in the sweep.
    doppler_spread is appended unbinned, since it's already one value per instance."""
    df = df.loc[df["group"].isin(classes)]

    hist_frames = []
    for feature in features:
        lo, hi = df[feature].quantile(0.01), df[feature].quantile(0.99)
        edges = np.linspace(lo, hi, n_bins + 1)
        bin_idx = np.clip(np.digitize(df[feature], edges[1:-1]), 0, n_bins - 1)
        counts = (
            df.assign(_bin=bin_idx)
            .groupby(INSTANCE_COLS + ["_bin"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=range(n_bins), fill_value=0)
        )
        counts.columns = [f"{feature}_bin{i}" for i in range(n_bins)]
        row_sums = counts.sum(axis=1)
        hist_frames.append(counts.div(row_sums.where(row_sums > 0, 1), axis=0))

    features_df = pd.concat(hist_frames, axis=1)
    extra = df.drop_duplicates(INSTANCE_COLS).set_index(INSTANCE_COLS, drop=False)
    extra = extra[["doppler_spread", "group", "sequence_name"]]
    return features_df.join(extra).reset_index(drop=True)


def run_bin_sweep(
    df: pd.DataFrame,
    bin_counts: tuple[int, ...] = (4, 8, 16, 32),
    classes: list[str] = FINAL_CLASSES,
    n_splits: int = 5,
    random_state: int = 0,
) -> pd.DataFrame:
    """Builds histogram-encoded features at each candidate bin count and runs both models
    (see taxonomy_separability.run_probe) at each, scoring by macro-average per-class
    one-vs-rest ROC-AUC - the number that should decide the bin count. Prints and returns one
    row per (bin_count, model)."""
    rows = []
    for n_bins in bin_counts:
        features = build_histogram_features(df, n_bins, classes)
        feature_cols = [c for c in features.columns if c not in ("sequence_name", "group")]
        X = features[feature_cols].to_numpy(dtype="float64")
        y = features["group"].to_numpy(dtype=str)
        groups = features["sequence_name"].to_numpy(dtype=str)

        results = run_probe(
            X, y, groups, classes, n_splits=n_splits, random_state=random_state,
            verbose=False, save_confusion=False, tag=f"bins{n_bins}",
        )
        for model_name, (_, auc_per_class, _) in results.items():
            row = {"n_bins": n_bins, "model": model_name, "macro_auc": np.mean(list(auc_per_class.values()))}
            row.update({f"auc_{cls}": auc for cls, auc in auc_per_class.items()})
            rows.append(row)
        print(f"n_bins={n_bins} done ({len(feature_cols)} features)")

    summary = pd.DataFrame(rows)
    print(summary.round(3).to_string(index=False))
    return summary


if __name__ == "__main__":
    from build_points_table import build_and_save_points_table

    df = build_and_save_points_table()
    df = add_relative_features(df)
    df = apply_class_groups(df)

    run_bin_sweep(df)
