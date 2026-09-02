"""two_wheeler bimodality check: bicycle vs motorized_two_wheeler, the two raw classes merged
into two_wheeler (dataloader.CLASS_GROUPS). Follow-up to the no_vr_compensated causal test
(MLP_Decisions_and_Findings.md section 15): vr_compensated is the model's single biggest F1
contributor for two_wheeler, despite the class spanning two different physical velocity
regimes (a slow bicycle, a much faster moped/scooter), worth checking directly rather than
assuming that reliance is on a coherent signal.

Three checks:
- Histogram + 1 vs 2-component GMM BIC on per-instance vr_compensated: is the pooled
  two_wheeler distribution genuinely bimodal, and does that trace to the sub-label split.
- Separability probe (LR + RF, same PROBE_FEATURES/run_probe machinery as the bus/large_
  vehicle/truck taxonomy decision): can bicycle and motorized_two_wheeler actually be told
  apart from each other, the direct question underneath the taxonomy choice to merge them.
- Fold composition check: a surgical, no-retraining test of the composition-shift mechanism
  the two checks above couldn't rule out, does a fold with a heavier motorized_two_wheeler
  mix among its two_wheeler val instances read as a vr_compensated cross-fold outlier
  (fold_stability.py's cached pairwise KS) or an F1 outlier (baseline's split-search folds)."""
import json

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.mixture import GaussianMixture

from dataloader import RESULTS_DIR
from feature_distributions import MLP_CLASSES, POINT_LEVEL_FEATURES
from fold_stability import FOLD_STABILITY_DIR
from mlp_classifier import MLP_DIR, apply_mlp_class_groups
from separability_probe import run_probe
from sequence_split import INSTANCE_COLS, select_best_split
from taxonomy_separability import NAME_TO_COLOR, PROBE_FEATURES, add_relative_features, build_instance_features

SUBCLASSES = ["bicycle", "motorized_two_wheeler"]
TWO_WHEELER_DIR = RESULTS_DIR / "two_wheeler_bimodality"


def plot_vr_compensated_bimodality(df: pd.DataFrame, classes: list[str] = SUBCLASSES):
    """Per-instance (median) vr_compensated, density-normalized panels sharing one p1/p99
    range: pooled two_wheeler, then each sub-label. Small multiples, not overlaid, same
    convention as fold_stability.py."""
    features = build_instance_features(df, classes)
    pooled = features["vr_compensated"].dropna()
    lo, hi = pooled.quantile(0.01), pooled.quantile(0.99)

    panels = [("two_wheeler (pooled)", pooled, "tab:gray")]
    for cls in classes:
        values = features.loc[features["label_name"] == cls, "vr_compensated"].dropna()
        panels.append((cls, values, NAME_TO_COLOR[cls]))

    fig, axes = plt.subplots(1, len(panels), figsize=(4.3 * len(panels), 4), sharey=True)
    for ax, (title, values, color) in zip(axes, panels):
        ax.hist(values, bins=32, range=(lo, hi), density=True, color=color, edgecolor="k", linewidth=0.3)
        ax.set_title(f"{title} (n={len(values)})")
        ax.set_xlabel("vr_compensated [m/s]")
    fig.suptitle("two_wheeler vr_compensated: pooled vs bicycle vs motorized_two_wheeler")
    fig.tight_layout()

    # Finding: the pooled panel stays dominated by bicycle's own right-skewed shape (peak
    # ~6-7 m/s), motorized_two_wheeler is only 4.7% of instances (1,617/34,498), too small a
    # share to be the pooled shape's main driver. motorized_two_wheeler's own panel is
    # strikingly multimodal in isolation though: a sharp spike at 0 m/s (idling/stopped), a
    # separate hump ~3-7 m/s (moving), a smaller tail ~10 m/s.
    TWO_WHEELER_DIR.mkdir(parents=True, exist_ok=True)
    path = TWO_WHEELER_DIR / "vr_compensated_bimodality.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    return fig


def gmm_bic_check(df: pd.DataFrame, classes: list[str] = SUBCLASSES) -> pd.DataFrame:
    """1 vs 2-component Gaussian mixture on pooled two_wheeler vr_compensated (per-instance
    median). BIC prefers the model that explains the data best net of extra-parameter
    complexity, lower is better. A 2-component win, with cluster means that roughly separate
    the two sub-labels, is numeric evidence for the bimodality read of the histogram above,
    not just a visual impression."""
    features = build_instance_features(df, classes)
    X = features["vr_compensated"].dropna().to_numpy().reshape(-1, 1)

    rows = []
    for n_components in (1, 2):
        gmm = GaussianMixture(n_components=n_components, random_state=0).fit(X)
        rows.append({
            "n_components": n_components,
            "bic": gmm.bic(X),
            "means": sorted(round(m, 3) for m in gmm.means_.ravel().tolist()),
        })
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    # Finding: 2 components wins on BIC (199,072 vs 201,321 for 1), cluster means 1.19 and
    # 3.75 m/s. Doesn't by itself confirm a bicycle/moped split though, see the histogram
    # finding above, motorized_two_wheeler is too rare to be the driver of this pooled result.
    return summary


def run_bicycle_motorized_probe(df: pd.DataFrame, classes: list[str] = SUBCLASSES, n_splits: int = 5, random_state: int = 0):
    """Same PROBE_FEATURES/run_probe machinery as the bus/large_vehicle/truck taxonomy
    decision (taxonomy_separability.run_separability_probe): the direct question underneath
    merging bicycle and motorized_two_wheeler into two_wheeler, can a probe actually tell
    them apart. High separability here would say the merge trades away real signal, the
    opposite conclusion from the bus/large_vehicle case, where poor separability supported
    merging."""
    features = build_instance_features(df, classes)
    X = features[PROBE_FEATURES].to_numpy(dtype="float64")
    y = features["label_name"].to_numpy(dtype=str)
    groups = features["sequence_name"].to_numpy(dtype=str)
    # Finding: pairwise AUC 0.665 (LR), 0.698 (RF), well short of the strong separability a
    # "two incompatible velocity regimes" story would predict. Caveat: severe class imbalance
    # in the held-out split (298 motorized_two_wheeler vs 6,494 bicycle, ~22:1), a small and
    # likely noisy sample to trust a precise AUC from.
    return run_probe(
        X, y, groups, classes, n_splits=n_splits, random_state=random_state,
        tag="bicycle_motorized_two_wheeler", results_dir=TWO_WHEELER_DIR,
    )


def fold_composition_vs_instability(
    df: pd.DataFrame,
    stratify_classes: list[str] = MLP_CLASSES,
    n_seeds: int = 10,
    base_random_state: int = 0,
) -> pd.DataFrame:
    """For each of the 6 split-sensitivity/fold_stability folds (same select_best_split call,
    so identical fold definitions), the motorized_two_wheeler share of that fold's two_wheeler
    val instances, against that fold's vr_compensated KS-to-other-folds (fold_stability.py's
    cached pairwise matrix, results/fold_stability/) and baseline two_wheeler F1
    (results/mlp/split_search/fold_<n>/). If motorized_two_wheeler-heavy folds are also the
    vr_compensated/F1 outliers, that's direct evidence the composition-shift mechanism
    contributes; if the two are uncorrelated, sparsity/averaging noise alone (section 15/16)
    remains the better-supported explanation."""
    df = apply_mlp_class_groups(add_relative_features(df))
    candidates = (
        select_best_split(df, classes=stratify_classes, features=POINT_LEVEL_FEATURES,
                           n_seeds=n_seeds, base_random_state=base_random_state)
        .drop_duplicates("fold").sort_values("fold")
    )

    instances = df.loc[df["group"] == "two_wheeler"].drop_duplicates(INSTANCE_COLS)
    rows = []
    for _, row in candidates.iterrows():
        fold_instances = instances.loc[instances["sequence_name"].isin(row["val_sequences"])]
        counts = fold_instances["label_name"].value_counts()
        n_bicycle, n_motorized = counts.get("bicycle", 0), counts.get("motorized_two_wheeler", 0)
        rows.append({
            "fold": row["fold"], "n_bicycle": n_bicycle, "n_motorized": n_motorized,
            "motorized_share": n_motorized / (n_bicycle + n_motorized) if (n_bicycle + n_motorized) else float("nan"),
        })
    composition = pd.DataFrame(rows).set_index("fold")

    ks = pd.read_csv(FOLD_STABILITY_DIR / "ks_matrix_by_fold_rcs_vr_compensated_x_rel_y_rel.csv")
    ks = ks.loc[(ks["class"] == "two_wheeler") & (ks["feature"] == "vr_compensated") & (ks["fold_i"] != ks["fold_j"])]
    ks_by_fold = ks.groupby("fold_i")["ks"].mean().rename("mean_ks_to_other_folds")

    f1_by_fold = {}
    for fold in composition.index:
        metrics = json.loads((MLP_DIR / "split_search" / f"fold_{fold}" / "mlp_val_metrics.json").read_text())
        f1_by_fold[fold] = metrics["two_wheeler"]["f1"]

    result = composition.join(ks_by_fold).join(pd.Series(f1_by_fold, name="two_wheeler_f1"))
    print(result.round(3).to_string())

    corr = result[["motorized_share", "mean_ks_to_other_folds", "two_wheeler_f1"]].corr(method="spearman")
    print("\nSpearman correlation (n=6 folds, read directionally not as a significance test):")
    print(corr.round(3).to_string())

    TWO_WHEELER_DIR.mkdir(parents=True, exist_ok=True)
    path = TWO_WHEELER_DIR / "fold_composition_vs_instability.csv"
    result.to_csv(path)
    print(f"\nSaved {path}")
    return result


if __name__ == "__main__":
    from build_points_table import build_and_save_points_table

    df = build_and_save_points_table()

    plot_vr_compensated_bimodality(df)
    gmm_bic_check(df)
    run_bicycle_motorized_probe(df)
    fold_composition_vs_instability(df)

    # Overall finding: mixed, not a clean confirmation. The pooled GMM technically prefers 2
    # components, but that's not well explained by a bicycle/moped split (motorized_two_wheeler
    # is too rare to dominate the pooled shape), and the separability probe found only modest
    # AUC, not the strong separation the "two incompatible velocity regimes" story would
    # predict. motorized_two_wheeler's own distribution is genuinely multimodal in isolation,
    # so it could still contribute somewhat to section 15's cross-fold KS via composition
    # shifts, just not shown here to be the primary driver. Full writeup:
    # MLP_Decisions_and_Findings.md section 17.
