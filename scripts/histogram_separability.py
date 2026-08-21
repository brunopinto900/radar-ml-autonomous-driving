"""Day 5 (cont'd): decide the per-instance histogram encoding's bin count empirically, by
running it through the same sequence-grouped LR+RF separability probe used for the taxonomy
decision - not by eyeballing the pooled bin-sweep plots in feature_distributions.py. Point-level
features average only ~2.9 points/instance (1,443,816 points / 503,759 instances), so "which
plot looks best" doesn't reflect what a typical instance's histogram actually looks like; bin
count is a hyperparameter of the encoding, scored the same way any other one is.
"""
import numpy as np
import pandas as pd

from dataloader import RESULTS_DIR
from feature_distributions import FINAL_CLASSES, POINT_LEVEL_FEATURES, apply_class_groups
from separability_probe import run_probe
from taxonomy_separability import add_relative_features

INSTANCE_COLS = ["sequence_name", "timestamp", "track_id"]
SWEEP_CACHE = RESULTS_DIR / "bin_sweep_results.parquet"


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
    (see separability_probe.run_probe) at each. Reports macro-average ROC-AUC *and*
    precision/recall/f1 - AUC is threshold-independent and doesn't reflect the actual argmax
    operating point once class weighting shifts the decision boundary, so a bin count picked on
    AUC alone isn't verified against the metrics that describe real deployed behavior. Cached to
    results/bin_sweep_results.parquet keyed by the exact bin_counts requested (each RF fit here
    takes minutes); skips the sweep entirely if the cache already covers the same bin_counts.
    Prints a per-class precision/recall/f1 table for every (bin_count, model) and returns one
    summary row per (bin_count, model)."""
    if SWEEP_CACHE.exists():
        cached = pd.read_parquet(SWEEP_CACHE)
        if set(cached["n_bins"].unique()) == set(bin_counts) and "macro_f1" in cached.columns:
            print(f"{SWEEP_CACHE} already covers bin_counts={bin_counts}, skipping sweep")
            print(cached.round(3).to_string(index=False))
            return cached
        print(f"{SWEEP_CACHE} covers different bin_counts (or an older schema) than requested, rebuilding")

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
        for model_name, (_, auc_per_class, metrics_per_class, _) in results.items():
            per_class = pd.DataFrame(metrics_per_class).T
            print(f"\nn_bins={n_bins}, {model_name} - precision/recall/f1 per class:")
            print(per_class.round(3).to_string())

            row = {
                "n_bins": n_bins,
                "model": model_name,
                "macro_auc": np.mean(list(auc_per_class.values())),
                "macro_precision": per_class["precision"].mean(),
                "macro_recall": per_class["recall"].mean(),
                "macro_f1": per_class["f1"].mean(),
            }
            row.update({f"auc_{cls}": auc for cls, auc in auc_per_class.items()})
            for cls, m in metrics_per_class.items():
                row[f"precision_{cls}"] = m["precision"]
                row[f"recall_{cls}"] = m["recall"]
                row[f"f1_{cls}"] = m["f1"]
            rows.append(row)
        print(f"n_bins={n_bins} done ({len(feature_cols)} features)")

    summary = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    summary.to_parquet(SWEEP_CACHE)
    print(f"Saved {SWEEP_CACHE}")
    print(summary[["n_bins", "model", "macro_auc", "macro_precision", "macro_recall", "macro_f1"]].round(3).to_string(index=False))
    return summary


if __name__ == "__main__":
    from build_points_table import build_and_save_points_table

    df = build_and_save_points_table()
    df = add_relative_features(df)
    df = apply_class_groups(df)

    run_bin_sweep(df)
