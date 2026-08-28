"""two_wheeler vs large_vehicle fold-to-fold stability: per-instance point-level feature
distributions across the same 6 folds split_sensitivity.py's noise floor check uses, checking
whether two_wheeler's larger F1 spread (MLP_Decisions_and_Findings.md Summary item 3) traces to
a genuinely less stable underlying feature distribution across folds, comparable sample size to
large_vehicle rules out "just less data" as the explanation.

Per-instance aggregation (mean of each instance's own points, not pooled points), so a busy
instance doesn't outweigh a sparse one. p1/p99 range fixed once per class/feature across all
folds combined, not per fold, so subplot axes are directly comparable. Density-normalized so
differing per-fold instance counts don't distort the visual comparison. Pairwise KS statistic
between every pair of folds (not fold vs. a pooled reference the fold is itself part of)
quantifies the same thing numerically, and shows which specific folds differ from which, not
just an average."""
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ks_2samp

from dataloader import FINAL_CLASS_COLORS, RESULTS_DIR
from feature_distributions import MLP_CLASSES, POINT_LEVEL_FEATURES
from mlp_classifier import apply_mlp_class_groups
from sequence_split import INSTANCE_COLS, select_best_split

COMPARE_CLASSES = ["two_wheeler", "large_vehicle"]
FOLD_STABILITY_DIR = RESULTS_DIR / "fold_stability"


def compute_ks_matrix(fold_values: dict[int, pd.Series]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pairwise KS statistic and p-value between every pair of folds (symmetric; diagonal 0 for
    the statistic, 1 for the p-value, a sample compared with itself is maximally "not
    different"). Richer than a single fold-vs-the-rest number: shows which specific folds differ
    from which, rather than just how far each fold sits from an average. The p-value answers the
    question the raw statistic can't on its own: given these sample sizes, is this gap larger
    than sampling noise alone would produce."""
    folds = sorted(fold_values)
    stat_mat = pd.DataFrame(0.0, index=folds, columns=folds)
    p_mat = pd.DataFrame(1.0, index=folds, columns=folds)
    for i in folds:
        for j in folds:
            if i < j:
                result = ks_2samp(fold_values[i], fold_values[j])
                stat_mat.loc[i, j] = stat_mat.loc[j, i] = result.statistic
                p_mat.loc[i, j] = p_mat.loc[j, i] = result.pvalue
    return stat_mat, p_mat


def plot_ks_matrix(ks_matrix: pd.DataFrame, title: str):
    """Heatmap of a fold x fold KS matrix, annotated with each pairwise value."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    values = ks_matrix.to_numpy()
    im = ax.imshow(values, cmap="viridis", vmin=0)
    ax.set_xticks(range(len(ks_matrix)))
    ax.set_yticks(range(len(ks_matrix)))
    ax.set_xticklabels(ks_matrix.columns)
    ax.set_yticklabels(ks_matrix.index)
    ax.set_xlabel("fold")
    ax.set_ylabel("fold")
    vmax = values.max() if values.max() > 0 else 1.0
    for i in range(len(ks_matrix)):
        for j in range(len(ks_matrix)):
            color = "white" if values[i, j] > vmax / 2 else "black"
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(im, ax=ax, label="KS statistic")
    ax.set_title(title, fontsize=10, wrap=True)
    fig.tight_layout()
    return fig


def build_per_instance_means(
    df: pd.DataFrame, features: list[str] = POINT_LEVEL_FEATURES, classes: list[str] = COMPARE_CLASSES
) -> pd.DataFrame:
    """One row per instance: mean of that instance's own points for each feature, not pooled
    across instances, so a 20-point instance doesn't outweigh a 1-point one when later averaged
    or histogrammed across a fold. Requires df to already have x_rel/y_rel (taxonomy_
    separability.add_relative_features) and `group` (mlp_classifier.apply_mlp_class_groups)."""
    sub = df.loc[df["group"].isin(classes)]
    agg = {feature: "mean" for feature in features}
    agg["group"] = "first"
    return sub.groupby(INSTANCE_COLS).agg(agg).reset_index()


def compare_fold_distributions(
    df: pd.DataFrame,
    features: list[str] = POINT_LEVEL_FEATURES,
    classes: list[str] = COMPARE_CLASSES,
    stratify_classes: list[str] = MLP_CLASSES,
    n_bins: int = 32,
    n_seeds: int = 10,
    base_random_state: int = 0,
    per_instance: bool = True,
):
    """For each (class, feature), one figure with 6 subplots (one per fold, the same 6 folds
    split_sensitivity.py trains on), each a density-normalized histogram of that fold's values,
    shared p1/p99 x-range across all 6 (computed once, pooled over every fold, not per fold,
    otherwise the axes wouldn't be comparable). Fold membership is each candidate's
    val_sequences (sequence_split.select_best_split), the same slice split_sensitivity.py
    evaluates macro F1 on.

    `per_instance` (default True): aggregate to one value per instance first (build_per_
    instance_means), the correct way to run this. False pools every point directly instead,
    reproducing the weighting bias per-instance aggregation exists to avoid, kept only to
    demonstrate that failure mode concretely, not for real analysis.

    Returns (figs, ks_matrices, ks_summary): figs is {(class, feature): histogram fig};
    ks_matrices is {(class, feature): 6x6 pairwise-KS heatmap fig}; ks_summary is the same
    matrices flattened to one row per class/feature/fold_i/fold_j."""
    if per_instance:
        pool = build_per_instance_means(df, features=features, classes=classes)
    else:
        pool = df.loc[df["group"].isin(classes), [*INSTANCE_COLS, "group", *features]]

    candidates = (
        select_best_split(df, classes=stratify_classes, features=POINT_LEVEL_FEATURES,
                           n_seeds=n_seeds, base_random_state=base_random_state)
        .drop_duplicates("fold")
        .sort_values("fold")
    )

    fold_pool = {}
    for _, row in candidates.iterrows():
        fold_pool[row["fold"]] = pool.loc[pool["sequence_name"].isin(row["val_sequences"])]

    unit = "instance mean" if per_instance else "raw point (pooled, biased)"
    figs = {}
    ks_matrices = {}
    ks_rows = []
    for cls in classes:
        cls_all = pool.loc[pool["group"] == cls]
        color = FINAL_CLASS_COLORS[cls]

        for feature in features:
            reference = cls_all[feature].dropna()
            lo, hi = reference.quantile(0.01), reference.quantile(0.99)

            fold_values = {
                fold: fold_pool[fold].loc[fold_pool[fold]["group"] == cls, feature].dropna()
                for fold in sorted(fold_pool)
            }

            fig, axes = plt.subplots(2, 3, figsize=(15, 7))
            axes = axes.flatten()
            for i, fold in enumerate(sorted(fold_values)):
                ax = axes[i]
                ax.hist(fold_values[fold], bins=n_bins, range=(lo, hi), density=True,
                         color=color, edgecolor="k", linewidth=0.3)
                ax.set_title(f"fold {fold} (n={len(fold_values[fold])})")
                ax.set_xlabel(feature)
            fig.suptitle(f"{cls}: {feature} per {unit}, by fold (density, range=[p1,p99] pooled)")
            fig.tight_layout()
            figs[(cls, feature)] = fig

            ks_matrix, p_matrix = compute_ks_matrix(fold_values)
            ks_matrices[(cls, feature)] = plot_ks_matrix(
                ks_matrix, f"{cls}: {feature} per {unit}, pairwise KS between folds"
            )
            for fold_i in ks_matrix.index:
                for fold_j in ks_matrix.columns:
                    ks_rows.append({
                        "class": cls, "feature": feature, "fold_i": fold_i, "fold_j": fold_j,
                        "ks": ks_matrix.loc[fold_i, fold_j], "p_value": p_matrix.loc[fold_i, fold_j],
                        "per_instance": per_instance,
                    })

    ks_summary = pd.DataFrame(ks_rows)
    return figs, ks_matrices, ks_summary


def summarize_ks(ks_summary: pd.DataFrame) -> pd.DataFrame:
    """Mean/max pairwise KS and the count of significant pairs (p < 0.05) per class/feature
    (diagonal excluded, always 0 stat/1 p-value by construction, would just dilute the mean),
    the compact numeric answer to "how much does this feature's distribution actually move fold
    to fold, and is that movement more than sampling noise." Max matters as much as mean, one
    bad pair of folds is what would actually explain an F1 outlier, not the average."""
    off_diagonal = ks_summary.loc[ks_summary["fold_i"] != ks_summary["fold_j"]]
    summary = off_diagonal.groupby(["class", "feature"]).agg(
        ks_mean=("ks", "mean"), ks_max=("ks", "max"),
        n_significant_pairs=("p_value", lambda p: int((p < 0.05).sum())),
        n_pairs=("p_value", "size"),
    ).round(3)
    return summary.sort_values(["feature", "class"])


def _run(df: pd.DataFrame, per_instance: bool, features: list[str] = POINT_LEVEL_FEATURES):
    tag = "by_fold" if per_instance else "by_fold_pointpooled"
    figs, ks_matrices, ks_summary = compare_fold_distributions(df, features=features, per_instance=per_instance)

    FOLD_STABILITY_DIR.mkdir(parents=True, exist_ok=True)
    for (cls, feature), fig in figs.items():
        path = FOLD_STABILITY_DIR / f"{cls}_{feature}_{tag}.png"
        fig.savefig(path, dpi=150)
        print(f"Saved {path}")
    for (cls, feature), fig in ks_matrices.items():
        path = FOLD_STABILITY_DIR / f"{cls}_{feature}_{tag}_ks_matrix.png"
        fig.savefig(path, dpi=150)
        print(f"Saved {path}")

    feature_tag = "_".join(features)
    csv_path = FOLD_STABILITY_DIR / f"ks_matrix_{tag}_{feature_tag}.csv"
    ks_summary.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    label = "per-instance (correct)" if per_instance else "point-pooled (biased, for comparison only)"
    print(f"\n=== {label}: pairwise KS matrices (statistic, then p-value) ===")
    for (cls, feature) in figs:
        matrix = ks_summary.loc[(ks_summary["class"] == cls) & (ks_summary["feature"] == feature)]
        ks_pivot = matrix.pivot(index="fold_i", columns="fold_j", values="ks")
        p_pivot = matrix.pivot(index="fold_i", columns="fold_j", values="p_value")
        print(f"\n{cls} / {feature}, KS statistic")
        print(ks_pivot.round(3).to_string())
        print(f"{cls} / {feature}, p-value")
        print(p_pivot.round(3).to_string())
    print(f"\n=== {label}: summary (off-diagonal) ===")
    print(summarize_ks(ks_summary).to_string())
    return ks_summary


def _compare_and_print(per_instance_ks: pd.DataFrame, pointpooled_ks: pd.DataFrame):
    compare = per_instance_ks.merge(
        pointpooled_ks, on=["class", "feature", "fold_i", "fold_j"], suffixes=("_per_instance", "_pointpooled")
    )
    compare = compare.loc[compare["fold_i"] != compare["fold_j"]]
    compare["ks_delta"] = compare["ks_pointpooled"] - compare["ks_per_instance"]
    print("\n=== per-instance vs point-pooled pairwise KS ===")
    print(compare[[
        "class", "feature", "fold_i", "fold_j",
        "ks_per_instance", "p_value_per_instance", "ks_pointpooled", "p_value_pointpooled", "ks_delta",
    ]].to_string(index=False))


if __name__ == "__main__":
    from build_points_table import build_and_save_points_table
    from taxonomy_separability import add_relative_features

    df = build_and_save_points_table()
    df = add_relative_features(df)
    df = apply_mlp_class_groups(df)

    per_instance_ks = _run(df, per_instance=True)
    pointpooled_ks = _run(df, per_instance=False)
    _compare_and_print(per_instance_ks, pointpooled_ks)

    # spatial_extent/doppler_spread are already one value per instance, .transform()-broadcast
    # across that instance's points in df, so "point-pooled" here means pooling that repeated
    # value once per point instead of once per instance, same weighting bias, different source
    extent_features = ["spatial_extent", "doppler_spread"]
    extent_per_instance_ks = _run(df, per_instance=True, features=extent_features)
    extent_pointpooled_ks = _run(df, per_instance=False, features=extent_features)
    _compare_and_print(extent_per_instance_ks, extent_pointpooled_ks)
