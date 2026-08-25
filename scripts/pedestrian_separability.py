"""pedestrian / two_wheeler confusion check: point-level azimuth x RCS scatter, plus the
separability_probe run twice, restricted to sparse instances (n_points <= SPARSE_MAX) and to
denser ones (n_points >= DENSE_MIN) - checks whether their confusion is capped by sparsity (same
mechanism as section 10's point-count finding) or persists even with more points to work with.
Diagnostic only, no MLP retraining - reuses taxonomy_separability's add_relative_features/
PROBE_FEATURES and separability_probe's run_probe.
"""
import matplotlib.pyplot as plt
import pandas as pd

from dataloader import FINAL_CLASS_COLORS, RESULTS_DIR
from mlp_classifier import apply_mlp_class_groups
from separability_probe import run_probe
from taxonomy_separability import INSTANCE_COLS, PROBE_FEATURES, add_relative_features

CONFUSION_CLASSES = ["pedestrian", "two_wheeler"]
NAME_TO_COLOR = FINAL_CLASS_COLORS
PEDESTRIAN_DIR = RESULTS_DIR / "pedestrian_two_wheeler"

SPARSE_MAX = 2
DENSE_MIN = 5


def _final_classes(df: pd.DataFrame) -> pd.DataFrame:
    """two_wheeler isn't a raw label_name (it's bicycle/motorized_two_wheeler merged, per
    dataloader.MLP_CLASS_GROUPS) - map to the final training class and swap it in as label_name
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
    ax.set_title("pedestrian vs two_wheeler - point-level azimuth x RCS")
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
    starve StratifiedGroupKFold of a class in some folds - worth seeing directly, not just
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


def summarize_probe_results(results: dict) -> pd.DataFrame:
    """Compact regime x model table from run_sparse_regime_probe's output: macro F1 (mean of
    per-class f1 in metrics_per_class) and pairwise AUC (same value for both classes in binary
    classification, per separability_probe.per_class_auc - only one needs showing)."""
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
    plot_azimuth_rcs_scatter(df)
    run_sparse_regime_probe(df)
