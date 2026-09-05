"""car/large_vehicle confusion check, same machinery as pedestrian_separability.py applied to
the confusion matrix's other vehicle-cluster pair (car 73% correct, 10% confused as
large_vehicle, Summary section 2/72):
- separability_probe run on sparse (n_points<=SPARSE_MAX)/dense (n_points>=DENSE_MIN) instances,
  PROBE_FEATURES and the MLP's own 65-dim histogram encoding.
- grouped permutation importance on the encoded probe.
- model-free KS test + ECDF per point-level feature, per regime.

Section 10 already established that large_vehicle's confusion is primarily a sparsity ceiling
(f1 climbs from 0.037 at 1 point to 0.995 at 11+, a size question unanswerable from few points),
unlike pedestrian/two_wheeler where the raw physical signal survives at n=1-2 (section 13). This
tests whether that expectation actually holds once run through the same probe/KS/importance
machinery, not just inferred from the point-count curve. No MLP-vs-RF fair comparison here,
that's this project's ceiling-vs-fixable machinery (section 20), out of scope for this pair.

Diagnostic only, no MLP retraining, reuses taxonomy_separability's add_relative_features/
PROBE_FEATURES, histogram_separability's build_histogram_features, and separability_probe's
run_probe."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from dataloader import FINAL_CLASS_COLORS, RESULTS_DIR
from feature_distributions import INSTANCE_LEVEL_FEATURES, MLP_CLASSES, POINT_LEVEL_FEATURES
from histogram_separability import build_histogram_features, fit_bin_edges
from mlp_classifier import DEVICE, HIDDEN_DIM, MLP, MLP_DIR, MODEL_FILENAME, N_BINS, apply_mlp_class_groups
from pedestrian_separability import _histogram_features_with_keys
from separability_probe import class_weights, run_probe
from sequence_split import load_split
from taxonomy_separability import INSTANCE_COLS, PROBE_FEATURES, add_relative_features

CONFUSION_CLASSES = ["car", "large_vehicle"]
NAME_TO_COLOR = FINAL_CLASS_COLORS
CAR_LARGE_VEHICLE_DIR = RESULTS_DIR / "car_large_vehicle"

SPARSE_MAX = 2
DENSE_MIN = 5


def _final_classes(df: pd.DataFrame) -> pd.DataFrame:
    """large_vehicle isn't a raw label_name (it's large_vehicle/truck/bus merged, per
    dataloader.MLP_CLASS_GROUPS), maps to the final training class and swaps it in as
    label_name so the rest of this module (and taxonomy_separability's helpers, which key off
    label_name) doesn't need to know the difference."""
    df = apply_mlp_class_groups(df)
    return df.drop(columns=["label_name"]).rename(columns={"group": "label_name"})


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
            tag=f"car_large_vehicle_{regime}",
            results_dir=CAR_LARGE_VEHICLE_DIR,
        )
    return results


def build_encoded_regime_data(df: pd.DataFrame, classes: list[str] = CONFUSION_CLASSES) -> dict[str, tuple]:
    """The MLP's actual 65-dim histogram encoding (histogram_separability.
    build_histogram_features, N_BINS=16 per POINT_LEVEL_FEATURES + unbinned doppler_spread),
    built once per sparse (n_points<=2)/dense (n_points>=5) regime, shared by
    run_encoded_regime_probe and run_encoded_permutation_importance so both operate on
    identical data. Bin edges fit per regime from that regime's own two-class pool (edges=None
    in build_histogram_features), same discipline as pedestrian_separability's equivalent.

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
    histogram encoding instead of PROBE_FEATURES' 5 hand-built stats, checking whether
    separability at low point counts survives once the data goes through the same
    binning/normalization the trained MLP actually sees."""
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
            tag=f"car_large_vehicle_encoded_{regime}",
            results_dir=CAR_LARGE_VEHICLE_DIR,
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
    """Which feature actually drives car/large_vehicle separability, per regime: grouped
    permutation importance (all N_BINS bin columns of one point-level feature shuffled together,
    not per-column, since shuffling bins independently would destroy the within-feature bin
    correlation without answering "what if the model never saw this feature at all"), on the
    same sequence-grouped held-out split run_encoded_regime_probe uses (same n_splits/
    random_state, so baseline_auc here reproduces that probe's AUC), for both models, per
    regime. doppler_spread is its own 1-column group, already unbinned. Section 10's read
    predicts x_rel/y_rel (extent/size) should dominate once points exist to spread across, not
    rcs/vr_compensated the way pedestrian/two_wheeler's dense regime leaned on y_rel.

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
    """Model-free check on whether car/large_vehicle overlap in raw feature values, or the
    model just doesn't need a feature because something else already covers the same ground.
    Two-sample KS test + ECDF, per regime, for every POINT_LEVEL_FEATURE, pooling all points
    from instances in that regime (not per-instance medians), matching exactly what
    build_histogram_features bins per feature. Saves
    results/car_large_vehicle/feature_ecdfs.png.

    KS p-values are meaningless at this sample size (effective n = n1*n2/(n1+n2) runs into the
    tens of thousands here), only ks_stat's magnitude is informative; p_value is still reported
    for completeness, not to be read as evidence of anything beyond "not exactly zero
    difference"."""
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

    CAR_LARGE_VEHICLE_DIR.mkdir(parents=True, exist_ok=True)
    path = CAR_LARGE_VEHICLE_DIR / "feature_ecdfs.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")

    return pd.DataFrame(rows).set_index(["regime", "feature"])


def summarize_probe_results(results: dict) -> pd.DataFrame:
    """Compact regime x model table from run_sparse_regime_probe's/run_encoded_regime_probe's
    output: macro F1 (mean of per-class f1 in metrics_per_class) and pairwise AUC (same value
    for both classes in binary classification, per separability_probe.per_class_auc, only one
    needs showing)."""
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


def _real_model_val_diagnostics_base(df: pd.DataFrame):
    """Shared setup for the three real-model (no retraining) diagnostics below, all answering
    the same question the regime probe can't: not whether separating signal exists in
    principle, but what the actual shared, jointly-trained standing-split MLP does with true
    car/large_vehicle val instances. Loads the cached baseline model (results/mlp/mlp_model.pt),
    builds val's 65-dim histogram encoding with instance keys retained (train-fit edges, same
    discipline as mlp_classifier.prepare_split_features), and runs one forward pass to get both
    the argmax prediction and the full 5-class softmax distribution per instance.

    Returns (sub, X_val, feature_cols, model):
    - sub: one row per true car/large_vehicle val instance, columns INSTANCE_COLS, group (true
      label), mlp_pred, prob_<cls> for cls in MLP_CLASSES, n_points, rcs_median,
      vr_compensated_median, x_extent, y_extent, bucket (car_correct/car_as_large_vehicle/
      large_vehicle_true), regime (sparse/mid/dense), and _row (its positional index into
      X_val, for permutation importance to reuse without rebuilding the encoding).
    - X_val, feature_cols, model: the real model's actual input matrix, column names, and the
      loaded model itself, for anything downstream that needs to run its own forward pass
      (permutation importance)."""
    df = apply_mlp_class_groups(df)
    df = add_relative_features(df)

    splits = load_split()
    train_df = df.loc[df["sequence_name"].isin(splits["train"])]
    val_df = df.loc[df["sequence_name"].isin(splits["val"])]

    edges = fit_bin_edges(train_df, N_BINS, POINT_LEVEL_FEATURES, range_method="percentile")
    val_hist = _histogram_features_with_keys(
        val_df, N_BINS, MLP_CLASSES, POINT_LEVEL_FEATURES, edges, INSTANCE_LEVEL_FEATURES, normalize=True
    )
    feature_cols = [c for c in val_hist.columns if c not in ("group", *INSTANCE_COLS)]
    X_val = val_hist[feature_cols].to_numpy(dtype="float32")
    val_hist = val_hist.assign(_row=np.arange(len(val_hist)))

    model = MLP(
        input_dim=X_val.shape[1], hidden_dim=HIDDEN_DIM, num_classes=len(MLP_CLASSES),
        n_hidden_layers=2, dropout=0.0, batch_norm=False,
    ).to(DEVICE)
    model.load_state_dict(torch.load(MLP_DIR / MODEL_FILENAME, map_location=DEVICE))
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_val, device=DEVICE))
        probs = F.softmax(logits, dim=1).cpu().numpy()
    pred_idx = probs.argmax(axis=1)
    val_hist = val_hist.assign(mlp_pred=[MLP_CLASSES[i] for i in pred_idx])
    prob_cols = [f"prob_{cls}" for cls in MLP_CLASSES]
    for i, cls in enumerate(MLP_CLASSES):
        val_hist[f"prob_{cls}"] = probs[:, i]

    instance_stats = val_df.groupby(INSTANCE_COLS).agg(
        rcs_median=("rcs", "median"),
        vr_compensated_median=("vr_compensated", "median"),
        x_extent=("x_rel", lambda s: s.max() - s.min()),
        y_extent=("y_rel", lambda s: s.max() - s.min()),
        spatial_extent=("spatial_extent", "first"),
        n_points=("n_points", "first"),
    ).reset_index()

    merged = val_hist[INSTANCE_COLS + ["group", "mlp_pred", "_row"] + prob_cols].merge(
        instance_stats, on=INSTANCE_COLS, how="inner"
    )

    sub = merged.loc[merged["group"].isin(CONFUSION_CLASSES)].copy()
    sub["bucket"] = np.select(
        [
            (sub["group"] == "car") & (sub["mlp_pred"] == "car"),
            (sub["group"] == "car") & (sub["mlp_pred"] == "large_vehicle"),
            sub["group"] == "large_vehicle",
        ],
        ["car_correct", "car_as_large_vehicle", "large_vehicle_true"],
        default="other",
    )
    sub = sub.loc[sub["bucket"] != "other"]
    sub["regime"] = np.select(
        [sub["n_points"] <= SPARSE_MAX, sub["n_points"] >= DENSE_MIN],
        ["sparse", "dense"],
        default="mid",
    )
    return sub, X_val, feature_cols, model


def real_model_pairwise_permutation_importance(
    df: pd.DataFrame, n_repeats: int = 20, random_state: int = 0
) -> pd.DataFrame:
    """What the section 21 write-up on car/large_vehicle actually needs and the regime probe
    can't give: permutation importance run on the real, cached, jointly-trained 5-class MLP
    itself (no retraining, no fresh probe), restricted to true car/large_vehicle val instances,
    per regime. Metric is accuracy on this restricted population (fraction predicted as their
    own true class), not AUC, this is a real 5-way argmax that can land on any of the 5
    classes, not a dedicated 2-class decision. All N_BINS bin columns of one point-level
    feature are shuffled together (same grouped-permutation discipline as
    run_encoded_permutation_importance), among just these restricted rows, not the whole val
    set, the question is what the model relies on for this specific population, not for
    everything at once."""
    rng = np.random.default_rng(random_state)
    sub, X_val, feature_cols, model = _real_model_val_diagnostics_base(df)

    rows = []
    for regime, g in sub.groupby("regime"):
        idx = g["_row"].to_numpy()
        y_true = g["group"].to_numpy()
        X_sub = X_val[idx]

        with torch.no_grad():
            baseline_idx = model(torch.tensor(X_sub, device=DEVICE)).argmax(dim=1).cpu().numpy()
        baseline_pred = np.array(MLP_CLASSES)[baseline_idx]
        baseline_acc = (baseline_pred == y_true).mean()

        for group_name, cols in FEATURE_GROUPS.items():
            col_idx = [feature_cols.index(c) for c in cols]
            drops = []
            for _ in range(n_repeats):
                X_perm = X_sub.copy()
                perm = rng.permutation(X_perm.shape[0])
                X_perm[:, col_idx] = X_perm[perm][:, col_idx]
                with torch.no_grad():
                    pred_idx = model(torch.tensor(X_perm, device=DEVICE)).argmax(dim=1).cpu().numpy()
                pred = np.array(MLP_CLASSES)[pred_idx]
                drops.append(baseline_acc - (pred == y_true).mean())
            rows.append({
                "regime": regime, "feature": group_name,
                "n": len(idx), "baseline_acc": baseline_acc,
                "mean_drop": np.mean(drops), "std_drop": np.std(drops),
            })

    result = pd.DataFrame(rows).set_index(["regime", "feature"])
    return result.reindex(["sparse", "mid", "dense"], level="regime")


RAW_PROFILE_FEATURES = ["rcs_median", "vr_compensated_median", "x_extent", "y_extent", "spatial_extent"]


def real_model_feature_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Extends car_misclassification_rcs_profile beyond rcs alone: raw (unbinned) per-instance
    statistics for every feature the permutation importance above can implicate, across the
    real model's own predicted buckets (car_correct/car_as_large_vehicle/large_vehicle_true),
    per regime. Answers, e.g., whether dense car_as_large_vehicle errors actually have
    unusually large x_extent (closer to true large_vehicle's footprint) the way the
    permutation-importance finding would predict, the same direct check that caught rcs's
    counterintuitive direction in the sparse regime."""
    sub, _X_val, _feature_cols, _model = _real_model_val_diagnostics_base(df)

    rows = []
    for (regime, bucket), g in sub.groupby(["regime", "bucket"]):
        for feature in RAW_PROFILE_FEATURES:
            vals = g[feature]
            rows.append({
                "regime": regime, "bucket": bucket, "feature": feature, "n": len(vals),
                "mean": vals.mean(), "median": vals.median(), "std": vals.std(),
                "p10": vals.quantile(0.10), "p25": vals.quantile(0.25),
                "p75": vals.quantile(0.75), "p90": vals.quantile(0.90),
            })
    result = pd.DataFrame(rows).set_index(["regime", "bucket", "feature"])
    return result.reindex(["sparse", "mid", "dense"], level="regime")


def real_model_confidence_margin(df: pd.DataFrame) -> pd.DataFrame:
    """Top-1 minus top-2 softmax margin for the real model's own predictions on true
    car/large_vehicle val instances, per bucket per regime. A confident-but-wrong error (large
    margin, model is sure and simply wrong) and a genuine toss-up (small margin, near the
    decision boundary) are different failure signatures, boundary/feature overlap predicts the
    second, something else (a shortcut, a mislabeled boundary case, capacity starvation)
    predicts the first."""
    sub, _X_val, _feature_cols, _model = _real_model_val_diagnostics_base(df)

    prob_cols = [f"prob_{cls}" for cls in MLP_CLASSES]
    prob_matrix = sub[prob_cols].to_numpy()
    sorted_probs = np.sort(prob_matrix, axis=1)[:, ::-1]
    sub = sub.assign(margin=sorted_probs[:, 0] - sorted_probs[:, 1])

    rows = []
    for (regime, bucket), g in sub.groupby(["regime", "bucket"]):
        margin = g["margin"]
        rows.append({
            "regime": regime, "bucket": bucket, "n": len(margin),
            "mean": margin.mean(), "median": margin.median(), "std": margin.std(),
            "p10": margin.quantile(0.10), "p25": margin.quantile(0.25),
            "p75": margin.quantile(0.75), "p90": margin.quantile(0.90),
        })
    result = pd.DataFrame(rows).set_index(["regime", "bucket"])
    return result.reindex(["sparse", "mid", "dense"], level="regime")


def car_misclassification_rcs_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Direct test of two pushback points on the sparse-regime probe's story: (1) is car's
    actual val misclassification-as-large_vehicle concentrated in the sparse regime the way the
    x_rel-degenerate-at-n<=2 argument implies, checked against real predictions rather than
    assumed from section 10's point-count curve, and (2) does the misclassified group's raw
    rcs actually sit closer to true large_vehicle's than to correctly-classified car's, the
    literal overlap claim, not just the aggregate KS statistic.

    Loads the real cached standing-split baseline MLP (results/mlp/mlp_model.pt, no
    retraining), gets its actual val predictions (same features/edges/split
    mlp_classifier.evaluate_val_metrics uses), and buckets true car/large_vehicle val instances
    into three groups: car_correct (true car, predicted car), car_as_large_vehicle (true car,
    predicted large_vehicle, the specific error under study), large_vehicle_true (every true
    large_vehicle instance, correctly predicted or not, the reference population). Reports raw
    per-instance median rcs distribution (not the binned encoding) for each group, split by
    n_points regime (sparse<=SPARSE_MAX, mid, dense>=DENSE_MIN).

    Returns a (regime, bucket)-indexed DataFrame: n, mean/median/std/p10/p25/p75/p90 of rcs."""
    from mlp_classifier import DEVICE, HIDDEN_DIM, MLP, MLP_DIR, MODEL_FILENAME
    from pedestrian_separability import _histogram_features_with_keys
    from sequence_split import load_split
    import torch

    df = apply_mlp_class_groups(df)
    df = add_relative_features(df)

    splits = load_split()
    train_df = df.loc[df["sequence_name"].isin(splits["train"])]
    val_df = df.loc[df["sequence_name"].isin(splits["val"])]

    from feature_distributions import MLP_CLASSES
    from histogram_separability import fit_bin_edges

    edges = fit_bin_edges(train_df, N_BINS, POINT_LEVEL_FEATURES, range_method="percentile")
    val_hist = _histogram_features_with_keys(
        val_df, N_BINS, MLP_CLASSES, POINT_LEVEL_FEATURES, edges, INSTANCE_LEVEL_FEATURES, normalize=True
    )
    feature_cols = [c for c in val_hist.columns if c not in ("group", *INSTANCE_COLS)]
    X_val = val_hist[feature_cols].to_numpy(dtype="float32")

    model = MLP(
        input_dim=X_val.shape[1], hidden_dim=HIDDEN_DIM, num_classes=len(MLP_CLASSES),
        n_hidden_layers=2, dropout=0.0, batch_norm=False,
    ).to(DEVICE)
    model.load_state_dict(torch.load(MLP_DIR / MODEL_FILENAME, map_location=DEVICE))
    model.eval()
    with torch.no_grad():
        pred_idx = model(torch.tensor(X_val, device=DEVICE)).argmax(dim=1).cpu().numpy()
    val_hist = val_hist.assign(mlp_pred=[MLP_CLASSES[i] for i in pred_idx])

    instance_rcs = val_df.groupby(INSTANCE_COLS).agg(
        rcs_median=("rcs", "median"), n_points=("n_points", "first")
    ).reset_index()
    merged = val_hist[INSTANCE_COLS + ["group", "mlp_pred"]].merge(instance_rcs, on=INSTANCE_COLS, how="inner")

    sub = merged.loc[merged["group"].isin(["car", "large_vehicle"])].copy()
    sub["bucket"] = np.select(
        [
            (sub["group"] == "car") & (sub["mlp_pred"] == "car"),
            (sub["group"] == "car") & (sub["mlp_pred"] == "large_vehicle"),
            sub["group"] == "large_vehicle",
        ],
        ["car_correct", "car_as_large_vehicle", "large_vehicle_true"],
        default="other",
    )
    sub = sub.loc[sub["bucket"] != "other"]
    sub["regime"] = np.select(
        [sub["n_points"] <= SPARSE_MAX, sub["n_points"] >= DENSE_MIN],
        ["sparse", "dense"],
        default="mid",
    )

    rows = []
    for (regime, bucket), g in sub.groupby(["regime", "bucket"]):
        rcs = g["rcs_median"]
        rows.append({
            "regime": regime, "bucket": bucket, "n": len(rcs),
            "mean": rcs.mean(), "median": rcs.median(), "std": rcs.std(),
            "p10": rcs.quantile(0.10), "p25": rcs.quantile(0.25),
            "p75": rcs.quantile(0.75), "p90": rcs.quantile(0.90),
        })
    result = pd.DataFrame(rows).set_index(["regime", "bucket"])
    return result.reindex(["sparse", "mid", "dense"], level="regime")


if __name__ == "__main__":
    from build_points_table import build_and_save_points_table

    df = build_and_save_points_table()

    hand_built_results = run_sparse_regime_probe(df)
    print("\n=== hand-built (PROBE_FEATURES) probe summary ===")
    print(summarize_probe_results(hand_built_results).round(3).to_string())

    encoded_results = run_encoded_regime_probe(df)
    print("\n=== encoded (65-dim histogram) probe summary ===")
    print(summarize_probe_results(encoded_results).round(3).to_string())

    print("\n=== grouped permutation importance (encoded, per regime) ===")
    print(run_encoded_permutation_importance(df).round(3).to_string())

    print("\n=== feature distribution KS (per regime) ===")
    print(feature_distribution_ks(df).round(4).to_string())
