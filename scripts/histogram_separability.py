"""Day 5 (cont'd): decide the per-instance histogram encoding's bin count empirically, by
running it through the same sequence-grouped LR+RF separability probe used for the taxonomy
decision - not by eyeballing the pooled bin-sweep plots in feature_distributions.py. Point-level
features average only ~2.9 points/instance (1,443,816 points / 503,759 instances), so "which
plot looks best" doesn't reflect what a typical instance's histogram actually looks like; bin
count is a hyperparameter of the encoding, scored the same way any other one is - via
separability_probe.run_probe_cv's 5-fold-averaged AUC/precision/recall/f1 on the fixed split's
train+val sequences (test excluded), not a single fold.
"""
import numpy as np
import pandas as pd

from dataloader import RESULTS_DIR
from feature_distributions import FINAL_CLASSES, INSTANCE_LEVEL_FEATURES, POINT_LEVEL_FEATURES, apply_class_groups
from separability_probe import run_probe_cv
from taxonomy_separability import add_relative_features

INSTANCE_COLS = ["sequence_name", "timestamp", "track_id"]
HISTOGRAM_SEPARABILITY_DIR = RESULTS_DIR / "histogram_separability"
SWEEP_CACHE = HISTOGRAM_SEPARABILITY_DIR / "bin_sweep_results.parquet"


def fit_bin_edges(
    df: pd.DataFrame, n_bins: int, features: list[str] = POINT_LEVEL_FEATURES, range_method: str = "percentile"
) -> dict[str, np.ndarray]:
    """Bin edges fit from `df` alone (pass a train-only df and reuse the returned edges for
    val/test via build_histogram_features's `edges` param - bin boundaries are a fitted
    preprocessing parameter, so refitting them per split would leak val/test's distribution into
    where the bins fall). `range_method` picks how the edges are placed: "percentile" (default,
    decision 2) and "gaussian" both split n_bins EQUAL-WIDTH bins across a clipped outer range,
    [1st, 99th] percentile vs [mean - 2*std, mean + 2*std] - the two agree closely for a roughly
    normal distribution but diverge for a skewed/heavy-tailed one, which is exactly what this
    project's RCS/Doppler features look like, 2 std devs can clip a different (and
    outlier-sensitive, since std itself is outlier-sensitive) fraction of the data than the
    1st/99th percentile does. "quantile" is a different kind of change, not just a different
    range but UNEQUAL-width bins: edges are placed so each bin holds roughly the same fraction of
    training points (np.quantile at n_bins+1 evenly spaced probabilities over the full [0, 1]
    range, no separate outlier clipping needed - the outermost bins naturally widen to absorb the
    sparse tails). That gives more resolution where the data is actually dense and less where
    it's sparse, instead of n_bins spent uniformly across the range regardless of where the points
    are. Can produce duplicate/non-increasing edges if a feature has enough exactly-repeated
    values concentrated at one quantile (not checked for here - watch for a degenerate/empty bin
    if a feature turns out to be very spiky)."""
    if range_method == "percentile":
        return {feature: np.linspace(df[feature].quantile(0.01), df[feature].quantile(0.99), n_bins + 1) for feature in features}
    if range_method == "gaussian":
        return {
            feature: np.linspace(
                df[feature].mean() - 2 * df[feature].std(), df[feature].mean() + 2 * df[feature].std(), n_bins + 1
            )
            for feature in features
        }
    if range_method == "quantile":
        return {feature: df[feature].quantile(np.linspace(0, 1, n_bins + 1)).to_numpy() for feature in features}
    raise ValueError(f"unknown range_method: {range_method!r}")


def build_histogram_features(
    df: pd.DataFrame,
    n_bins: int,
    classes: list[str] = FINAL_CLASSES,
    features: list[str] = POINT_LEVEL_FEATURES,
    edges: dict[str, np.ndarray] | None = None,
    extra_features: list[str] = INSTANCE_LEVEL_FEATURES,
    normalize: bool = True,
) -> pd.DataFrame:
    """One row per instance: each point-level feature becomes n_bins columns holding, by
    default (`normalize=True`), the fraction of that instance's own points landing in each bin -
    fraction, not raw count, so a busy instance's histogram encodes distribution shape rather
    than just point count. That normalization is exactly what makes point count irrecoverable
    from the encoding (a 1-point and an N-point instance landing in the same bin are
    indistinguishable) - `normalize=False` skips it and leaves raw per-bin counts, which
    embeds point count back in implicitly (the bins for one feature sum to that instance's
    n_points) at the cost of putting sparse and busy instances on different numeric scales for
    the same underlying shape. `edges` (see fit_bin_edges), if given, are used as-is instead of
    being recomputed from `df` - needed to encode val/test with edges fit on train only.
    Otherwise edges are fit from `df` itself (fine when df is the only/whole pool being encoded,
    e.g. the CV probes). `extra_features` (default doppler_spread) are appended unbinned, since
    they're already one value per instance - pass [] to skip (e.g. a feature-swap variant that
    bins everything)."""
    df = df.loc[df["group"].isin(classes)]
    if edges is None:
        edges = fit_bin_edges(df, n_bins, features)

    hist_frames = []
    for feature in features:
        e = edges[feature]
        bin_idx = np.clip(np.digitize(df[feature], e[1:-1]), 0, n_bins - 1)
        counts = (
            df.assign(_bin=bin_idx)
            .groupby(INSTANCE_COLS + ["_bin"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=range(n_bins), fill_value=0)
        )
        counts.columns = [f"{feature}_bin{i}" for i in range(n_bins)]
        if normalize:
            row_sums = counts.sum(axis=1)
            counts = counts.div(row_sums.where(row_sums > 0, 1), axis=0)
        hist_frames.append(counts)

    features_df = pd.concat(hist_frames, axis=1)
    extra = df.drop_duplicates(INSTANCE_COLS).set_index(INSTANCE_COLS, drop=False)
    extra = extra[extra_features + ["group", "sequence_name"]]
    return features_df.join(extra).reset_index(drop=True)


def build_stat_features(
    df: pd.DataFrame,
    classes: list[str] = FINAL_CLASSES,
    feature_stats: dict[str, list[str]] | None = None,
    extra_features: list[str] = INSTANCE_LEVEL_FEATURES,
) -> pd.DataFrame:
    """One row per instance: each point-level feature in `feature_stats` (e.g. {"rcs":
    ["mean","median","std"], "radial": ["std"]}) becomes one column per requested statistic,
    aggregated directly from that instance's own points - no bin edges to fit, unlike
    build_histogram_features, since each instance's mean/median/std depends only on its own
    points, nothing to fit on train alone and reuse on val/test. "std" uses ddof=0 (population,
    not sample) so a 1-point instance gets a well-defined 0 instead of NaN (ddof=1 divides by
    n-1, undefined at n=1). `extra_features` (default doppler_spread) are appended as-is via
    .first(), since they're already one value per instance, broadcast to every point row -
    aggregating them the same way as a point-level feature would be degenerate (every point in an
    instance shares the identical value, so e.g. std would always come out 0)."""
    df = df.loc[df["group"].isin(classes)]
    group = df.groupby(INSTANCE_COLS)

    stat_cols = []
    for feature, stats in feature_stats.items():
        for stat in stats:
            col = group[feature].std(ddof=0) if stat == "std" else getattr(group[feature], stat)()
            col.name = f"{feature}_{stat}"
            stat_cols.append(col)
    stat_df = pd.concat(stat_cols, axis=1)

    extra = df.drop_duplicates(INSTANCE_COLS).set_index(INSTANCE_COLS, drop=False)
    extra = extra[extra_features + ["group", "sequence_name"]]
    return stat_df.join(extra).reset_index(drop=True)


def run_bin_sweep(
    df: pd.DataFrame,
    bin_counts: tuple[int, ...] = (4, 8, 16, 32),
    classes: list[str] = FINAL_CLASSES,
    n_splits: int = 5,
    random_state: int = 0,
) -> pd.DataFrame:
    """Builds histogram-encoded features at each candidate bin count and scores them with
    separability_probe.run_probe_cv's proper n_splits-fold averaging (mean +/- std), restricted
    to the fixed split's train+val sequences (results/sequence_split.json - test stays untouched).
    A single fold isn't trustworthy enough to rank bin counts against (see Design_Decisions.md
    decision 3's "Confirmed with 5-fold CV" subsection - this replaced an earlier single-fold
    version of this sweep). Cached to results/bin_sweep_results.parquet keyed by the exact
    bin_counts requested (each bin count trains LR+RF n_splits times - this takes a while);
    skips the sweep entirely if the cache already covers the same bin_counts. run_probe_cv prints
    its own per-class mean+/-std table for every (bin_count, model); returns one summary row per
    (bin_count, model)."""
    from sequence_split import load_split

    splits = load_split()
    trainval_sequences = set(splits["train"]) | set(splits["val"])
    df = df.loc[df["sequence_name"].isin(trainval_sequences)]

    if SWEEP_CACHE.exists():
        cached = pd.read_parquet(SWEEP_CACHE)
        if set(cached["n_bins"].unique()) == set(bin_counts) and "macro_f1_mean" in cached.columns:
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

        print(f"\nn_bins={n_bins} ({len(feature_cols)} features):")
        cv_summary = run_probe_cv(X, y, groups, classes, n_splits=n_splits, random_state=random_state)
        for model_name, metrics in cv_summary.items():
            row = {"n_bins": n_bins, "model": model_name}
            for metric in ("auc", "precision", "recall", "f1"):
                means = [metrics[metric][cls]["mean"] for cls in classes]
                row[f"macro_{metric}_mean"] = float(np.mean(means))
                for cls in classes:
                    row[f"{metric}_{cls}_mean"] = metrics[metric][cls]["mean"]
                    row[f"{metric}_{cls}_std"] = metrics[metric][cls]["std"]
            rows.append(row)
        print(f"n_bins={n_bins} done")

    summary = pd.DataFrame(rows)
    SWEEP_CACHE.parent.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(SWEEP_CACHE)
    print(f"Saved {SWEEP_CACHE}")
    macro_cols = ["n_bins", "model", "macro_auc_mean", "macro_precision_mean", "macro_recall_mean", "macro_f1_mean"]
    print(summary[macro_cols].round(3).to_string(index=False))
    return summary


if __name__ == "__main__":
    from build_points_table import build_and_save_points_table

    df = build_and_save_points_table()
    df = add_relative_features(df)
    df = apply_class_groups(df)

    run_bin_sweep(df)
