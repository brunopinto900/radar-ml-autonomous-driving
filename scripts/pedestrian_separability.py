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
import torch
from scipy.stats import ks_2samp, pearsonr, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from dataloader import FINAL_CLASS_COLORS, RESULTS_DIR
from feature_distributions import HISTOGRAM_FEATURES, INSTANCE_LEVEL_FEATURES, MLP_CLASSES, POINT_LEVEL_FEATURES
from histogram_separability import build_histogram_features, fit_bin_edges
from mlp_classifier import DEVICE, HIDDEN_DIM, MLP, MLP_DIR, MODEL_FILENAME, N_BINS, apply_mlp_class_groups
from separability_probe import class_weights, run_probe
from sequence_split import select_best_split
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


VR_BANDS = (0.5, 1.0, 2.0)


def vr_local_density_ratio(df: pd.DataFrame, classes: list[str] = CONFUSION_CLASSES) -> pd.DataFrame:
    """Direct mechanism check for the confusion's direction (`two_wheeler`->`pedestrian` common,
    the reverse rare): not "do the distributions overlap" (`feature_distribution_ks`) but "which
    class actually outnumbers the other inside the shared near-zero `vr_compensated` band,"
    per regime. Raw point counts within `abs(vr_compensated) < threshold`, no density
    estimation, reported both as a share of each class's own regime total (within-class) and as
    a direct between-class ratio (`local_ratio` = classes[0] count / classes[1] count), the
    number that approximates what a classifier implicitly weighs when deciding what an
    ambiguous near-zero-velocity point actually is."""
    df = apply_mlp_class_groups(df)
    df = df.loc[df["group"].isin(classes)]
    df = add_relative_features(df)

    regimes = {
        "sparse": df["n_points"] <= SPARSE_MAX,
        "dense": df["n_points"] >= DENSE_MIN,
    }
    rows = []
    for regime, mask in regimes.items():
        sub = df.loc[mask]
        totals = sub["group"].value_counts()
        for thresh in VR_BANDS:
            band = sub.loc[sub["vr_compensated"].abs() < thresh]
            counts = band["group"].value_counts()
            row = {"regime": regime, "vr_band": thresh}
            for cls in classes:
                row[f"{cls}_n"] = int(counts.get(cls, 0))
                row[f"{cls}_pct_of_total"] = 100 * counts.get(cls, 0) / totals[cls]
            row["local_ratio"] = counts.get(classes[0], 0) / max(counts.get(classes[1], 0), 1)
            rows.append(row)
    return pd.DataFrame(rows).set_index(["regime", "vr_band"])


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


def _histogram_features_with_keys(
    df: pd.DataFrame,
    n_bins: int,
    classes: list[str],
    features: list[str],
    edges: dict,
    extra_features: list[str],
    normalize: bool = True,
) -> pd.DataFrame:
    """Identical construction to histogram_separability.build_histogram_features, except
    INSTANCE_COLS (sequence_name/timestamp/track_id) are kept as explicit columns instead of
    being discarded by reset_index(drop=True). mlp_probe_agreement needs to align two
    separately built encodings (the MLP's 5-class one, the probe's 2-class one) on the same
    physical instance, relying on implicit row-order matching across two different
    constructions would be a silent-failure risk not worth taking for a key this cheap to
    keep."""
    df = df.loc[df["group"].isin(classes)]
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
    extra = extra[extra_features + ["group"] + INSTANCE_COLS]
    return features_df.join(extra).reset_index(drop=True)


def mlp_probe_agreement(
    df: pd.DataFrame,
    n_seeds: int = 10,
    base_random_state: int = 0,
) -> pd.DataFrame:
    """Test B: does the MLP's actual two_wheeler val errors look like a genuine data ceiling (an
    independent probe finds the same instances hard too) or an MLP-specific gap (the probe
    confidently gets them right, the MLP doesn't)? Pooled across the 6 split-sensitivity folds
    (split_sensitivity.py's select_best_split), not the single standing split, two_wheeler has
    this project's largest fold-to-fold instability (F1 spread 0.386, Summary item 3), so one
    split's answer isn't trustworthy alone here.

    Per fold: loads that fold's already-cached baseline MLP (results/mlp/split_search/fold_<n>,
    no retraining), gets its actual argmax predictions on that fold's val two_wheeler instances,
    same MLP_CLASSES bin edges (fit on that fold's train, all 5 classes) the model was actually
    trained with. Separately refits an LR/RF probe restricted to pedestrian/two_wheeler on that
    same fold's train sequences (no leakage against the fold's own val set), with its own bin
    edges fit on that 2-class train subset, a deliberate choice, not reusing the MLP's 5-class
    edges, the probe is meant to be an independently-reasonable encoding of the same raw data,
    not yoked to the MLP pipeline's own preprocessing.

    Restricts to MLP val two_wheeler instances the MLP called either two_wheeler (correct) or
    pedestrian (the confusion under study); predictions of car/pedestrian_group are a different
    failure mode, out of scope here. Joins MLP and probe predictions on the shared instance key.

    Returns one row per (fold, instance): mlp_pred, probe_pred_lr/rf, probe_p_pedestrian_lr/rf."""
    candidates = select_best_split(
        df, classes=MLP_CLASSES, features=HISTOGRAM_FEATURES, n_seeds=n_seeds, base_random_state=base_random_state
    ).drop_duplicates("fold").sort_values("fold")

    all_rows = []
    for _, cand in candidates.iterrows():
        fold = cand["fold"]
        splits = {"train": cand["train_sequences"], "val": cand["val_sequences"], "test": cand["test_sequences"]}
        train_df = df.loc[df["sequence_name"].isin(splits["train"])]
        val_df = df.loc[df["sequence_name"].isin(splits["val"])]

        # --- MLP side: real cached model, real 5-class edges ---
        mlp_edges = fit_bin_edges(train_df, N_BINS, POINT_LEVEL_FEATURES, range_method="percentile")
        mlp_val = _histogram_features_with_keys(
            val_df, N_BINS, MLP_CLASSES, POINT_LEVEL_FEATURES, mlp_edges, INSTANCE_LEVEL_FEATURES, normalize=True
        )
        mlp_feature_cols = [c for c in mlp_val.columns if c not in ("group", *INSTANCE_COLS)]
        X_mlp = mlp_val[mlp_feature_cols].to_numpy(dtype="float32")

        model_cache = MLP_DIR / "split_search" / f"fold_{fold}" / MODEL_FILENAME
        model = MLP(
            input_dim=X_mlp.shape[1], hidden_dim=HIDDEN_DIM, num_classes=len(MLP_CLASSES),
            n_hidden_layers=2, dropout=0.0, batch_norm=False,
        ).to(DEVICE)
        model.load_state_dict(torch.load(model_cache, map_location=DEVICE))
        model.eval()
        with torch.no_grad():
            mlp_pred_idx = model(torch.tensor(X_mlp, device=DEVICE)).argmax(dim=1).cpu().numpy()
        mlp_val = mlp_val.assign(mlp_pred=[MLP_CLASSES[i] for i in mlp_pred_idx])

        two_wheeler_mask = mlp_val["group"] == "two_wheeler"
        in_scope = mlp_val["mlp_pred"].isin(CONFUSION_CLASSES)
        mlp_tw = mlp_val.loc[two_wheeler_mask & in_scope, INSTANCE_COLS + ["mlp_pred"]]

        # --- probe side: independently trained, own edges, same fold's train sequences ---
        probe_train_df = train_df.loc[train_df["group"].isin(CONFUSION_CLASSES)]
        probe_edges = fit_bin_edges(probe_train_df, N_BINS, POINT_LEVEL_FEATURES, range_method="percentile")
        probe_train_hist = _histogram_features_with_keys(
            probe_train_df, N_BINS, CONFUSION_CLASSES, POINT_LEVEL_FEATURES, probe_edges, INSTANCE_LEVEL_FEATURES, normalize=True
        )
        probe_val_hist = _histogram_features_with_keys(
            val_df, N_BINS, CONFUSION_CLASSES, POINT_LEVEL_FEATURES, probe_edges, INSTANCE_LEVEL_FEATURES, normalize=True
        )
        probe_feature_cols = [c for c in probe_train_hist.columns if c not in ("group", *INSTANCE_COLS)]

        X_train = probe_train_hist[probe_feature_cols].to_numpy(dtype="float64")
        y_train = probe_train_hist["group"].to_numpy(dtype=str)
        X_val = probe_val_hist[probe_feature_cols].to_numpy(dtype="float64")

        scaler = StandardScaler().fit(X_train)
        weights = class_weights(pd.Series(y_train))
        lr = LogisticRegression(max_iter=1000, class_weight=weights).fit(scaler.transform(X_train), y_train)
        rf = RandomForestClassifier(n_estimators=300, class_weight=weights, random_state=base_random_state).fit(X_train, y_train)

        pos_idx_lr = list(lr.classes_).index("pedestrian")
        pos_idx_rf = list(rf.classes_).index("pedestrian")
        probe_val_hist = probe_val_hist.assign(
            probe_pred_lr=lr.predict(scaler.transform(X_val)),
            probe_pred_rf=rf.predict(X_val),
            probe_p_pedestrian_lr=lr.predict_proba(scaler.transform(X_val))[:, pos_idx_lr],
            probe_p_pedestrian_rf=rf.predict_proba(X_val)[:, pos_idx_rf],
        )

        merged = mlp_tw.merge(
            probe_val_hist[INSTANCE_COLS + ["probe_pred_lr", "probe_pred_rf", "probe_p_pedestrian_lr", "probe_p_pedestrian_rf"]],
            on=INSTANCE_COLS, how="inner",
        )
        merged["fold"] = fold
        all_rows.append(merged)

    return pd.concat(all_rows, ignore_index=True)


def summarize_mlp_probe_agreement(result: pd.DataFrame) -> pd.DataFrame:
    """Of the MLP's actual errors (mlp_pred=='pedestrian'), what fraction does the independent
    RF/LR probe also call 'pedestrian'? High agreement -> genuine data ceiling (an independent
    model finds the same instances ambiguous too). Low agreement -> the probe finds usable
    signal the MLP isn't using, an MLP-specific gap. Reports the same breakdown for the MLP's
    correct calls as a sanity baseline (should show high agreement if the probe is reasonable)."""
    rows = []
    for mlp_pred, sub in result.groupby("mlp_pred"):
        for model in ("lr", "rf"):
            rows.append({
                "mlp_pred": mlp_pred,
                "probe_model": model,
                "n": len(sub),
                "probe_agrees_pct": 100 * (sub[f"probe_pred_{model}"] == mlp_pred).mean(),
                "mean_probe_p_pedestrian": sub[f"probe_p_pedestrian_{model}"].mean(),
            })
    return pd.DataFrame(rows).set_index(["mlp_pred", "probe_model"])


def mlp_vs_rf_multiclass(
    df: pd.DataFrame,
    n_seeds: int = 10,
    base_random_state: int = 0,
) -> pd.DataFrame:
    """Closes mlp_probe_agreement's scope caveat: that comparison used a binary probe (only ever
    choosing two_wheeler vs. pedestrian), an easier task than the MLP's real 5-way job. This
    fits a 5-class RandomForestClassifier on the exact same per-fold 65-dim histogram encoding
    the MLP trains on (same bin edges, same MLP_CLASSES, same fold's train sequences), so the
    comparison is genuinely fair: same features, same 5-way task, different model family (tree
    ensemble vs. this project's small 2x16 MLP), not a different, easier task. Also the first
    time this project has compared a different model family on the same features, sections 6-7
    only ever varied the MLP's own capacity/depth.

    Reuses the exact same per-fold MLP-side construction as mlp_probe_agreement (same edges,
    same feature matrix), so mlp_pred/rf_pred are computed on identical inputs, only the
    classifier differs. Pooled across the same 6 split-sensitivity folds, no leakage (RF is
    trained on that fold's train sequences only).

    Returns one row per (fold, instance), every class, not restricted to two_wheeler: group
    (true class), mlp_pred, rf_pred."""
    candidates = select_best_split(
        df, classes=MLP_CLASSES, features=HISTOGRAM_FEATURES, n_seeds=n_seeds, base_random_state=base_random_state
    ).drop_duplicates("fold").sort_values("fold")

    all_rows = []
    for _, cand in candidates.iterrows():
        fold = cand["fold"]
        splits = {"train": cand["train_sequences"], "val": cand["val_sequences"], "test": cand["test_sequences"]}
        train_df = df.loc[df["sequence_name"].isin(splits["train"])]
        val_df = df.loc[df["sequence_name"].isin(splits["val"])]

        edges = fit_bin_edges(train_df, N_BINS, POINT_LEVEL_FEATURES, range_method="percentile")
        train_hist = _histogram_features_with_keys(
            train_df, N_BINS, MLP_CLASSES, POINT_LEVEL_FEATURES, edges, INSTANCE_LEVEL_FEATURES, normalize=True
        )
        val_hist = _histogram_features_with_keys(
            val_df, N_BINS, MLP_CLASSES, POINT_LEVEL_FEATURES, edges, INSTANCE_LEVEL_FEATURES, normalize=True
        )
        feature_cols = [c for c in train_hist.columns if c not in ("group", *INSTANCE_COLS)]

        # --- MLP side: real cached model, no retraining ---
        X_val_mlp = val_hist[feature_cols].to_numpy(dtype="float32")
        model_cache = MLP_DIR / "split_search" / f"fold_{fold}" / MODEL_FILENAME
        model = MLP(
            input_dim=X_val_mlp.shape[1], hidden_dim=HIDDEN_DIM, num_classes=len(MLP_CLASSES),
            n_hidden_layers=2, dropout=0.0, batch_norm=False,
        ).to(DEVICE)
        model.load_state_dict(torch.load(model_cache, map_location=DEVICE))
        model.eval()
        with torch.no_grad():
            mlp_pred_idx = model(torch.tensor(X_val_mlp, device=DEVICE)).argmax(dim=1).cpu().numpy()

        # --- RF side: 5-class, same features, same fold's train sequences ---
        X_train = train_hist[feature_cols].to_numpy(dtype="float64")
        y_train = train_hist["group"].to_numpy(dtype=str)
        X_val_rf = val_hist[feature_cols].to_numpy(dtype="float64")

        weights = class_weights(pd.Series(y_train))
        rf = RandomForestClassifier(n_estimators=300, class_weight=weights, random_state=base_random_state).fit(X_train, y_train)
        rf_pred = rf.predict(X_val_rf)

        fold_result = val_hist[INSTANCE_COLS + ["group"]].copy()
        fold_result["mlp_pred"] = [MLP_CLASSES[i] for i in mlp_pred_idx]
        fold_result["rf_pred"] = rf_pred
        fold_result["fold"] = fold
        all_rows.append(fold_result)

    return pd.concat(all_rows, ignore_index=True)


def summarize_mlp_vs_rf_multiclass(result: pd.DataFrame, classes: list[str] = MLP_CLASSES) -> pd.DataFrame:
    """Per-class recall, MLP vs. the 5-class RF, same instances, same features, plus macro
    recall. Answers whether RF's advantage on two_wheeler (mlp_probe_agreement, binary) survives
    once RF also has to handle the other 4 classes, the fair version of that comparison."""
    rows = []
    for cls in classes:
        sub = result.loc[result["group"] == cls]
        rows.append({
            "class": cls,
            "n": len(sub),
            "mlp_recall": (sub["mlp_pred"] == cls).mean(),
            "rf_recall": (sub["rf_pred"] == cls).mean(),
        })
    summary = pd.DataFrame(rows)
    summary.loc["macro_avg"] = {
        "class": "macro_avg", "n": summary["n"].sum(),
        "mlp_recall": summary["mlp_recall"].mean(), "rf_recall": summary["rf_recall"].mean(),
    }
    return summary.set_index("class")


def two_wheeler_confusion_breakdown(result: pd.DataFrame, classes: list[str] = MLP_CLASSES) -> pd.DataFrame:
    """Row-normalized confusion breakdown for true two_wheeler only, MLP vs. RF side by side,
    the direct fair-comparison analog of the original mlp_confusion_matrix.png row this whole
    investigation started from."""
    sub = result.loc[result["group"] == "two_wheeler"]
    rows = []
    for model in ("mlp", "rf"):
        counts = sub[f"{model}_pred"].value_counts(normalize=True)
        rows.append({"model": model, **{cls: counts.get(cls, 0.0) for cls in classes}})
    return pd.DataFrame(rows).set_index("model")


if __name__ == "__main__":
    from build_points_table import build_and_save_points_table

    df = build_and_save_points_table()
    run_sparse_regime_probe(df)

    encoded_results = run_encoded_regime_probe(df)
    print("\n=== encoded (65-dim histogram) probe summary ===")
    print(summarize_probe_results(encoded_results).round(3).to_string())

    print("\n=== grouped permutation importance (encoded, per regime) ===")
    print(run_encoded_permutation_importance(df).round(3).to_string())
