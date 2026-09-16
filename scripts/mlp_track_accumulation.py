"""track-accumulation branch: the quantile_bins_5features MLP (histogram-encoded
baseline architecture, mlp_classifier.py/quantile_bins_5features_split_sensitivity.py)
trained on the same windowed point sets DeepReflecs uses (deepreflecs_track_
accumulation.build_windowed_point_sets), instead of DeepReflecs' point-set network.

Same accumulation (per-scan x_seq/y_seq recentering, sliding window, range_sc mode),
same 5 features, only the encoder/architecture differs. Lets "does accumulation help"
be checked independently of DeepReflecs' specific point-set architecture: if a simple
histogram+MLP also improves from more pooled points, that favors accumulation helping
via denser/less noisy per-instance statistics (which a histogram can capture) over
something specific to DeepReflecs' order-blind max-pool learning fine per-point
patterns a histogram would flatten away.

A window doesn't correspond to one (sequence_name, timestamp, track_id) row group like
a single-scan instance does (it pools several), so histogram_separability.
build_histogram_features can't be reused directly (it groups by that key). Reimplements
the same quantile-bin-edges + per-instance-fraction encoding directly on build_windowed_
point_sets' (point_sets, labels) output instead."""
import numpy as np
import pandas as pd
import torch

from dataloader import RESULTS_DIR
from deepreflecs_track_accumulation import ACCUMULATION_BASELINE_N, STRIDE, build_windowed_point_sets
from feature_distributions import MLP_CLASSES
from mlp_classifier import DEVICE, train_mlp
from sequence_split import load_split

FEATURES = ["x_rel", "y_rel", "vr_compensated", "range_sc", "rcs"]
N_BINS = 16
EPOCHS = 200
TRACK_ACC_MLP_DIR = RESULTS_DIR / "track_accumulation_mlp"


def fit_quantile_edges(
    point_sets: list[np.ndarray], features: list[str] = FEATURES, n_bins: int = N_BINS
) -> dict[str, np.ndarray]:
    """Same recipe as histogram_separability.fit_bin_edges(range_method="quantile"):
    edges placed so each bin holds roughly the same fraction of pooled training
    points. Fit on train windows only, reused as-is for val/test (see
    build_windowed_histogram_features's `edges` param)."""
    all_points = np.concatenate(point_sets, axis=0)
    return {
        feature: np.quantile(all_points[:, i], np.linspace(0, 1, n_bins + 1))
        for i, feature in enumerate(features)
    }


def build_windowed_histogram_features(
    point_sets: list[np.ndarray],
    features: list[str] = FEATURES,
    n_bins: int = N_BINS,
    edges: dict[str, np.ndarray] | None = None,
    normalize: bool = True,
    vr_col: str = "vr_compensated",
    include_doppler_spread: bool = True,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """One row per window: each feature becomes n_bins columns (fraction of that
    window's own points landing in each quantile bin, mirrors histogram_separability.
    build_histogram_features). `include_doppler_spread` (default True, matching
    quantile_bins_5features's own config) appends one extra column, median absolute
    deviation of vr_compensated across the window's own pooled points (taxonomy_
    separability.add_relative_features's exact definition, median(|vr - median(vr)|)),
    computed per window rather than reused from the single-scan cache whose instance
    definition doesn't match a pooled window. Set False for a clean same-5-features
    comparison against DeepReflecs, which has no such extra dispersion feature at all."""
    if edges is None:
        edges = fit_quantile_edges(point_sets, features, n_bins)

    vr_idx = features.index(vr_col)
    n = len(point_sets)
    n_cols = n_bins * len(features) + (1 if include_doppler_spread else 0)
    X = np.zeros((n, n_cols), dtype="float32")
    for w, points in enumerate(point_sets):
        col = 0
        for i, feature in enumerate(features):
            e = edges[feature]
            bin_idx = np.clip(np.digitize(points[:, i], e[1:-1]), 0, n_bins - 1)
            counts = np.bincount(bin_idx, minlength=n_bins).astype("float32")
            if normalize:
                total = counts.sum()
                if total > 0:
                    counts = counts / total
            X[w, col : col + n_bins] = counts
            col += n_bins
        if include_doppler_spread:
            vr = points[:, vr_idx]
            X[w, col] = np.median(np.abs(vr - np.median(vr)))
    return X, edges


def run_windowed_mlp(
    df: pd.DataFrame,
    n: int = ACCUMULATION_BASELINE_N,
    stride: int = STRIDE,
    range_sc_mode: str = "raw",
    classes: list[str] = MLP_CLASSES,
    features: list[str] = FEATURES,
    n_bins: int = N_BINS,
    epochs: int = EPOCHS,
    include_doppler_spread: bool = True,
    splits: dict[str, list[str]] | None = None,
    output_dir=None,
):
    """Single train/test run (no fold sweep, matching the one fixed canonical split
    the comparable DeepReflecs N=5/raw run used) of quantile_bins_5features' own
    config (bin_range="quantile", epochs=200, same 5 features) on windowed
    (accumulated) point sets instead of single-scan instances. include_doppler_spread
    defaults to True (matching quantile_bins_5features), set False for a clean same-
    5-features comparison against DeepReflecs, which never gets a dispersion feature."""
    if output_dir is None:
        suffix = "" if include_doppler_spread else "_nodopplerspread"
        output_dir = TRACK_ACC_MLP_DIR / f"N{n}_stride{stride}_range{range_sc_mode}_quantile{suffix}"
    if splits is None:
        splits = load_split()

    train_df = df.loc[df["sequence_name"].isin(splits["train"])]
    val_df = df.loc[df["sequence_name"].isin(splits["val"])]
    test_df = df.loc[df["sequence_name"].isin(splits["test"])]

    train_sets, y_train = build_windowed_point_sets(train_df, classes, features, n, stride, range_sc_mode)
    val_sets, y_val = build_windowed_point_sets(val_df, classes, features, n, stride, range_sc_mode)
    test_sets, y_test = build_windowed_point_sets(test_df, classes, features, n, stride, range_sc_mode)

    X_train, edges = build_windowed_histogram_features(train_sets, features, n_bins, include_doppler_spread=include_doppler_spread)
    X_val, _ = build_windowed_histogram_features(val_sets, features, n_bins, edges=edges, include_doppler_spread=include_doppler_spread)
    X_test, _ = build_windowed_histogram_features(test_sets, features, n_bins, edges=edges, include_doppler_spread=include_doppler_spread)

    print(
        f"windowed MLP N={n} stride={stride} range_sc_mode={range_sc_mode} include_doppler_spread={include_doppler_spread}: "
        f"train={len(train_sets)} val={len(val_sets)} test={len(test_sets)}, X_dim={X_train.shape[1]}"
    )

    model, history = train_mlp(X_train, y_train, X_val, y_val, classes=classes, epochs=epochs)

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "mlp_model.pt")

    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_recall_fscore_support

    model.eval()
    with torch.no_grad():
        y_pred = model(torch.tensor(X_test, device=DEVICE)).argmax(dim=1).cpu().numpy()
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=range(len(classes)), zero_division=0
    )
    metrics_df = pd.DataFrame(
        {"precision": precision, "recall": recall, "f1": f1, "support": support}, index=classes
    )
    metrics_df.to_json(output_dir / "mlp_test_metrics.json", orient="index", indent=2)
    print(f"per-class precision/recall/f1 (test, windowed MLP N={n}, range_sc_mode={range_sc_mode}):")
    print(metrics_df.round(3).to_string())
    print(f"macro F1: {metrics_df['f1'].mean():.4f}")

    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_test, y_pred, labels=range(len(classes)), normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, colorbar=False, values_format=".2f")
    ax.set_title(f"windowed MLP (N={n}): test confusion matrix (row-normalized)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(output_dir / "mlp_test_confusion_matrix.png", dpi=150)
    print(f"Saved {output_dir / 'mlp_test_confusion_matrix.png'}")

    return model, history, metrics_df
