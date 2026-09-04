"""pedestrian/two_wheeler confusion check:
- point-level azimuth x RCS scatter.
- separability_probe run twice, on sparse (n_points <= SPARSE_MAX) and dense
  (n_points >= DENSE_MIN) instances, checking whether the confusion is capped by sparsity
  (same mechanism as section 10's point-count finding) or persists regardless. Two feature-set
  variants of this same regime split: PROBE_FEATURES' 5 hand-built stats
  (run_sparse_regime_probe) and the MLP's actual 65-dim histogram encoding
  (run_encoded_regime_probe), the latter answers separability in what the trained model itself
  sees rather than in a hand-built proxy for it.

Diagnostic only, no MLP retraining, reuses taxonomy_separability's add_relative_features/
PROBE_FEATURES, histogram_separability's build_histogram_features, and separability_probe's
run_probe."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, pearsonr, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from dataloader import FINAL_CLASS_COLORS, RESULTS_DIR
from feature_distributions import INSTANCE_LEVEL_FEATURES, POINT_LEVEL_FEATURES
from histogram_separability import build_histogram_features
from mlp_classifier import N_BINS, apply_mlp_class_groups
from separability_probe import class_weights, run_probe
from taxonomy_separability import INSTANCE_COLS, PROBE_FEATURES, add_relative_features

CONFUSION_CLASSES = ["pedestrian", "two_wheeler"]
NAME_TO_COLOR = FINAL_CLASS_COLORS
PEDESTRIAN_DIR = RESULTS_DIR / "pedestrian_two_wheeler"

SPARSE_MAX = 2
DENSE_MIN = 5


def _final_classes(df: pd.DataFrame) -> pd.DataFrame:
    """two_wheeler isn't a raw label_name (it's bicycle/motorized_two_wheeler merged, per
    dataloader.MLP_CLASS_GROUPS), maps to the final training class and swaps it in as label_name
    so the rest of this module (and taxonomy_separability's helpers, which key off label_name)
    doesn't need to know the difference."""
    df = apply_mlp_class_groups(df)
    return df.drop(columns=["label_name"]).rename(columns={"group": "label_name"})


def plot_azimuth_rcs_scatter(df: pd.DataFrame, classes: list[str] = CONFUSION_CLASSES):
    """Point-level azimuth_sc vs rcs, one point per detection (not per instance), colored by
    class. Checks whether a joint angle/RCS relationship separates the two classes in a way
    neither feature does alone. Saves results/pedestrian_two_wheeler/azimuth_rcs_scatter.png."""
    df = _final_classes(df)
    sub = df.loc[df["label_name"].isin(classes)]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for cls in classes:
        s = sub.loc[sub["label_name"] == cls]
        ax.scatter(s["azimuth_sc"], s["rcs"], s=4, alpha=0.3, color=NAME_TO_COLOR[cls], label=cls)
    ax.set_xlabel("azimuth_sc [rad]")
    ax.set_ylabel("rcs [dBsm]")
    ax.set_title("pedestrian vs two_wheeler: point-level azimuth x RCS")
    ax.legend(markerscale=4)
    fig.tight_layout()

    PEDESTRIAN_DIR.mkdir(parents=True, exist_ok=True)
    path = PEDESTRIAN_DIR / "azimuth_rcs_scatter.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    return fig


def build_instance_features(df: pd.DataFrame, classes: list[str] = CONFUSION_CLASSES) -> pd.DataFrame:
    """Same per-instance aggregation as taxonomy_separability.build_instance_features
    (rcs/vr_compensated median, x/y extent, doppler_spread), with n_points kept alongside for
    the sparse/dense split below."""
    df = _final_classes(df)
    df = df.loc[df["label_name"].isin(classes)]
    df = add_relative_features(df)

    group = df.groupby(INSTANCE_COLS)
    features = group.agg(
        rcs=("rcs", "median"),
        vr_compensated=("vr_compensated", "median"),
        x_extent=("x_rel", lambda s: s.max() - s.min()),
        y_extent=("y_rel", lambda s: s.max() - s.min()),
        doppler_spread=("doppler_spread", "first"),
        n_points=("n_points", "first"),
        label_name=("label_name", "first"),
    ).reset_index()
    return features


def run_sparse_regime_probe(
    df: pd.DataFrame,
    classes: list[str] = CONFUSION_CLASSES,
    n_splits: int = 5,
    random_state: int = 0,
    verbose: bool = True,
):
    """run_probe on PROBE_FEATURES, once for n_points <= SPARSE_MAX and once for
    n_points >= DENSE_MIN. Prints class counts per regime before each probe (small regimes can
    starve StratifiedGroupKFold of a class in some folds, worth seeing directly, not just
    inferring from a failure)."""
    features = build_instance_features(df, classes)

    results = {}
    regimes = {
        "sparse": features["n_points"] <= SPARSE_MAX,
        "dense": features["n_points"] >= DENSE_MIN,
    }
    for regime, mask in regimes.items():
        sub = features.loc[mask]
        print(f"\n=== {regime} regime ({'n_points <= ' + str(SPARSE_MAX) if regime == 'sparse' else 'n_points >= ' + str(DENSE_MIN)}) ===")
        print(sub["label_name"].value_counts().to_string())

        X = sub[PROBE_FEATURES].to_numpy(dtype="float64")
        y = sub["label_name"].to_numpy(dtype=str)
        groups = sub["sequence_name"].to_numpy(dtype=str)
        results[regime] = run_probe(
            X,
            y,
            groups,
            classes,
            n_splits=n_splits,
            random_state=random_state,
            verbose=verbose,
            tag=f"pedestrian_two_wheeler_{regime}",
            results_dir=PEDESTRIAN_DIR,
        )
    return results


def build_encoded_regime_data(df: pd.DataFrame, classes: list[str] = CONFUSION_CLASSES) -> dict[str, tuple]:
    """The MLP's actual 65-dim histogram encoding (histogram_separability.
    build_histogram_features, N_BINS=16 per POINT_LEVEL_FEATURES + unbinned doppler_spread),
    built once per sparse (n_points<=2)/dense (n_points>=5) regime, shared by
    run_encoded_regime_probe and run_encoded_permutation_importance so both operate on
    identical data. Bin edges fit per regime from that regime's own two-class pool (edges=None
    in build_histogram_features), same discipline as histogram_separability.run_bin_sweep's own
    CV probes, there's no separate train split to fit edges on here.

    Returns {regime: (X, y, groups, feature_cols)}."""
    df = apply_mlp_class_groups(df)
    df = df.loc[df["group"].isin(classes)]
    df = add_relative_features(df)

    data = {}
    regimes = {
        "sparse": df["n_points"] <= SPARSE_MAX,
        "dense": df["n_points"] >= DENSE_MIN,
    }
    for regime, mask in regimes.items():
        hist = build_histogram_features(
            df.loc[mask], N_BINS, classes, POINT_LEVEL_FEATURES, extra_features=INSTANCE_LEVEL_FEATURES, normalize=True
        )
        feature_cols = [c for c in hist.columns if c not in ("sequence_name", "group")]
        data[regime] = (
            hist[feature_cols].to_numpy(dtype="float64"),
            hist["group"].to_numpy(dtype=str),
            hist["sequence_name"].to_numpy(dtype=str),
            feature_cols,
        )
    return data


def run_encoded_regime_probe(
    df: pd.DataFrame,
    classes: list[str] = CONFUSION_CLASSES,
    n_splits: int = 5,
    random_state: int = 0,
    verbose: bool = True,
):
    """Same sparse/dense comparison as run_sparse_regime_probe, but on the MLP's actual 65-dim
    histogram encoding instead of PROBE_FEATURES' 5 hand-built stats. Section 13's finding (raw
    physical signal survives at n=1-2) used the hand-built set; this checks whether that
    survives once the data goes through the same binning/normalization the trained MLP actually
    sees, not just whether separating signal exists in the raw data."""
    data = build_encoded_regime_data(df, classes)

    results = {}
    for regime, (X, y, groups, _feature_cols) in data.items():
        print(f"\n=== {regime} regime, encoded ({'n_points <= ' + str(SPARSE_MAX) if regime == 'sparse' else 'n_points >= ' + str(DENSE_MIN)}) ===")
        print(pd.Series(y).value_counts().to_string())

        results[regime] = run_probe(
            X,
            y,
            groups,
            classes,
            n_splits=n_splits,
            random_state=random_state,
            verbose=verbose,
            tag=f"pedestrian_two_wheeler_encoded_{regime}",
            results_dir=PEDESTRIAN_DIR,
        )
    return results


FEATURE_GROUPS = {feature: [f"{feature}_bin{i}" for i in range(N_BINS)] for feature in POINT_LEVEL_FEATURES}
FEATURE_GROUPS["doppler_spread"] = ["doppler_spread"]


def run_encoded_permutation_importance(
    df: pd.DataFrame,
    classes: list[str] = CONFUSION_CLASSES,
    n_splits: int = 5,
    random_state: int = 0,
    n_repeats: int = 20,
) -> pd.DataFrame:
    """Section 19's open question: is the sparse regime's worse pairwise AUC driven by
    x_rel/y_rel becoming informative once there are more points (near-degenerate at
    n_points=1-2), or by rcs/vr_compensated simply getting less noisy at higher point counts
    (the mechanism sections 15-16 established for two_wheeler generally)? Recall alone can't
    distinguish these, this can: grouped permutation importance on the same sequence-grouped
    held-out split run_encoded_regime_probe uses (same n_splits/random_state, so baseline_auc
    here reproduces that probe's AUC), for both models, per regime.

    Grouped, not sklearn's per-column permutation_importance: all N_BINS bin columns of one
    point-level feature (e.g. every rcs_bin*) are shuffled together with the same row
    permutation, not independently, since shuffling bins independently would destroy the
    within-feature bin correlation (they're one histogram) without answering "what if the model
    never saw this feature at all," which is the actual question. doppler_spread is its own
    1-column group, already unbinned.

    Returns a (regime, model, feature)-indexed DataFrame with baseline_auc, mean_drop, std_drop
    (mean_drop = baseline_auc - permuted_auc, averaged over n_repeats; more positive = more
    important)."""
    rng = np.random.default_rng(random_state)
    data = build_encoded_regime_data(df, classes)

    rows = []
    for regime, (X, y, groups, feature_cols) in data.items():
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        train_idx, test_idx = next(splitter.split(X, y, groups))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pos_class = classes[0]
        y_test_binary = (y_test == pos_class).astype(int)

        scaler = StandardScaler().fit(X_train)
        weights = class_weights(pd.Series(y_train))
        models = {
            "logistic_regression": (LogisticRegression(max_iter=1000, class_weight=weights), True),
            "random_forest": (
                RandomForestClassifier(n_estimators=300, class_weight=weights, random_state=random_state),
                False,
            ),
        }

        for model_name, (clf, needs_scaling) in models.items():
            X_tr = scaler.transform(X_train) if needs_scaling else X_train
            X_te = scaler.transform(X_test) if needs_scaling else X_test
            clf.fit(X_tr, y_train)
            pos_idx = list(clf.classes_).index(pos_class)

            baseline_auc = roc_auc_score(y_test_binary, clf.predict_proba(X_te)[:, pos_idx])

            for group_name, cols in FEATURE_GROUPS.items():
                col_idx = [feature_cols.index(c) for c in cols]
                drops = []
                for _ in range(n_repeats):
                    X_perm = X_te.copy()
                    perm = rng.permutation(X_perm.shape[0])
                    X_perm[:, col_idx] = X_perm[perm][:, col_idx]
                    auc = roc_auc_score(y_test_binary, clf.predict_proba(X_perm)[:, pos_idx])
                    drops.append(baseline_auc - auc)
                rows.append({
                    "regime": regime,
                    "model": model_name,
                    "feature": group_name,
                    "baseline_auc": baseline_auc,
                    "mean_drop": np.mean(drops),
                    "std_drop": np.std(drops),
                })

    return pd.DataFrame(rows).set_index(["regime", "model", "feature"])


def feature_distribution_ks(df: pd.DataFrame, classes: list[str] = CONFUSION_CLASSES) -> pd.DataFrame:
    """Model-free check on whether `rcs`'s near-zero permutation importance (section 19) means
    `pedestrian`/`two_wheeler` genuinely overlap in raw `rcs`, or the model just doesn't need it
    because something else already covers the same ground. Two-sample KS test + ECDF, per
    regime, for every POINT_LEVEL_FEATURE, pooling all points from instances in that regime (not
    per-instance medians), matching exactly what build_histogram_features bins per feature.
    Saves results/pedestrian_two_wheeler/feature_ecdfs.png."""
    df = apply_mlp_class_groups(df)
    df = df.loc[df["group"].isin(classes)]
    df = add_relative_features(df)

    regimes = {
        "sparse": df["n_points"] <= SPARSE_MAX,
        "dense": df["n_points"] >= DENSE_MIN,
    }

    rows = []
    fig, axes = plt.subplots(len(regimes), len(POINT_LEVEL_FEATURES), figsize=(4.2 * len(POINT_LEVEL_FEATURES), 4 * len(regimes)))
    for ri, (regime, mask) in enumerate(regimes.items()):
        sub = df.loc[mask]
        for fi, feature in enumerate(POINT_LEVEL_FEATURES):
            a = sub.loc[sub["group"] == classes[0], feature].dropna().to_numpy()
            b = sub.loc[sub["group"] == classes[1], feature].dropna().to_numpy()
            stat, pvalue = ks_2samp(a, b)
            rows.append({
                "regime": regime, "feature": feature, "ks_stat": stat, "p_value": pvalue,
                f"n_{classes[0]}": len(a), f"n_{classes[1]}": len(b),
            })

            ax = axes[ri, fi]
            for cls, vals in ((classes[0], a), (classes[1], b)):
                xs = np.sort(vals)
                ys = np.arange(1, len(xs) + 1) / len(xs)
                ax.plot(xs, ys, label=cls, color=NAME_TO_COLOR[cls])
            ax.set_title(f"{regime}: {feature}\nKS={stat:.3f}")
            ax.set_ylabel("ECDF")
            if ri == 0 and fi == 0:
                ax.legend()
    fig.tight_layout()

    PEDESTRIAN_DIR.mkdir(parents=True, exist_ok=True)
    path = PEDESTRIAN_DIR / "feature_ecdfs.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")

    return pd.DataFrame(rows).set_index(["regime", "feature"])


RCS_CORRELATION_FEATURES = ["x_rel", "y_rel", "vr_compensated", "doppler_spread", "azimuth_sc"]


def rcs_position_correlation(df: pd.DataFrame, classes: list[str] = CONFUSION_CLASSES) -> pd.DataFrame:
    """The redundancy half of the same question: does `rcs` correlate with another feature the
    model already has, physically plausible since `rcs` depends on aspect angle and `x_rel`/
    `y_rel` encode where on the object a point sits (`azimuth_sc` included too, the scene-level
    angle rather than the intra-object one). `build_histogram_features` bins each point-level
    feature separately, so a real correlation here means the model could recover
    `rcs`-equivalent information from a correlated feature's histogram even while barely relying
    on `rcs`'s own bins, explaining a near-zero permutation drop without requiring the classes to
    overlap in raw `rcs` (see `feature_distribution_ks` for that direct check). Point-level
    Pearson (linear) and Spearman (monotonic, more robust to a non-linear physical relationship)
    correlation against `rcs`, pooled across all points, per class."""
    df = apply_mlp_class_groups(df)
    df = df.loc[df["group"].isin(classes)]
    df = add_relative_features(df)

    rows = []
    for cls in classes:
        sub = df.loc[df["group"] == cls]
        for feature in RCS_CORRELATION_FEATURES:
            valid = sub[["rcs", feature]].dropna()
            pearson_r, pearson_p = pearsonr(valid["rcs"], valid[feature])
            spearman_r, spearman_p = spearmanr(valid["rcs"], valid[feature])
            rows.append({
                "class": cls, "feature": feature,
                "pearson_r": pearson_r, "pearson_p": pearson_p,
                "spearman_r": spearman_r, "spearman_p": spearman_p,
                "n": len(valid),
            })
    return pd.DataFrame(rows).set_index(["class", "feature"])


def summarize_probe_results(results: dict) -> pd.DataFrame:
    """Compact regime x model table from run_sparse_regime_probe's output: macro F1 (mean of
    per-class f1 in metrics_per_class) and pairwise AUC (same value for both classes in binary
    classification, per separability_probe.per_class_auc, only one needs showing)."""
    rows = []
    for regime, models in results.items():
        for model_name, (_report, auc_per_class, metrics_per_class, _fig) in models.items():
            macro_f1 = sum(m["f1"] for m in metrics_per_class.values()) / len(metrics_per_class)
            rows.append({
                "regime": regime,
                "model": model_name,
                "macro_f1": macro_f1,
                "pairwise_auc": next(iter(auc_per_class.values())),
            })
    return pd.DataFrame(rows).set_index(["regime", "model"])


if __name__ == "__main__":
    from build_points_table import build_and_save_points_table

    df = build_and_save_points_table()
    run_sparse_regime_probe(df)

    encoded_results = run_encoded_regime_probe(df)
    print("\n=== encoded (65-dim histogram) probe summary ===")
    print(summarize_probe_results(encoded_results).round(3).to_string())

    print("\n=== grouped permutation importance (encoded, per regime) ===")
    print(run_encoded_permutation_importance(df).round(3).to_string())
