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
from deepreflecs_track_accumulation import ACCUMULATION_BASELINE_N, STRIDE, TRACK_COLS, build_windowed_point_sets
from feature_distributions import MLP_CLASSES
from mlp_classifier import DEVICE, train_mlp
from sequence_split import load_split
from taxonomy_separability import INSTANCE_COLS

FEATURES = ["x_rel", "y_rel", "vr_compensated", "range_sc", "rcs"]
TEMPORAL_FEATURES = ["rcs", "vr_compensated"]
N_BINS = 16
TRANSITION_BINS = 4
EPOCHS = 200
TRACK_ACC_MLP_DIR = RESULTS_DIR / "track_accumulation_mlp"


def build_windowed_temporal_features(
    df: pd.DataFrame,
    classes: list[str] = MLP_CLASSES,
    temporal_features: list[str] = TEMPORAL_FEATURES,
    n: int = ACCUMULATION_BASELINE_N,
    stride: int = STRIDE,
) -> np.ndarray:
    """One row per window, one column per feature in temporal_features: mean absolute
    consecutive diff across that window's own per-scan medians (median of that scan's
    own points, not the raw pooled points, see track_accumulation.md's capacity
    ablation follow-up discussion), divided by total elapsed real time spanned by the
    window (last scan's timestamp minus first, in seconds; 0 for a single-scan window,
    where there's no diff to take).

    Normalizing by elapsed time rather than by number of gaps matters once a window
    can mix gaps of different real duration, which happens for the all-sensor
    (cross-sensor handoff) points table: two sensors briefly overlapping produce a
    short gap, sensor-2-alone stretches produce a long one, and "1 gap" doesn't mean
    the same amount of real motion in each case. For a single-sensor points table
    gaps are uniform (~0.074s for sensor 2), so this is just a constant rescaling
    there and doesn't change any sensor-2-only ranking (see track_accumulation.md).

    Mirrors build_windowed_point_sets' own filtering/sorting/window-slicing exactly
    (same classes mask, same TRACK_COLS + timestamp sort, same stride slicing) so row
    i here lines up with point_sets[i]/labels[i] from build_windowed_point_sets(df,
    classes, ..., n, stride, ...) called on this same df."""
    mask = df["group"].isin(classes)
    filtered = df.loc[mask]
    feat_matrix = filtered[temporal_features].to_numpy(dtype="float64")

    scan_positions = filtered.groupby(INSTANCE_COLS, sort=False).indices
    scan_arrays = list(scan_positions.values())
    scan_keys = pd.DataFrame(list(scan_positions.keys()), columns=INSTANCE_COLS)
    scan_timestamps = scan_keys["timestamp"].to_numpy(dtype="float64")
    scan_keys["_scan_idx"] = np.arange(len(scan_keys))
    scan_keys = scan_keys.sort_values(TRACK_COLS + ["timestamp"])

    scan_medians = np.stack([np.median(feat_matrix[positions], axis=0) for positions in scan_arrays])

    rows = []
    for _, track_scans in scan_keys.groupby(TRACK_COLS, sort=False):
        ordered = track_scans["_scan_idx"].to_numpy()
        for i in range(0, len(ordered), stride):
            window_idx = ordered[max(0, i - n + 1) : i + 1]
            medians = scan_medians[window_idx]
            if len(window_idx) > 1:
                elapsed_s = (scan_timestamps[window_idx[-1]] - scan_timestamps[window_idx[0]]) / 1e6
                variation = np.abs(np.diff(medians, axis=0)).sum(axis=0) / elapsed_s
            else:
                variation = np.zeros(len(temporal_features))
            rows.append(variation)

    return np.array(rows, dtype="float32")


def build_windowed_diff_vector_features(
    df: pd.DataFrame,
    classes: list[str] = MLP_CLASSES,
    temporal_features: list[str] = TEMPORAL_FEATURES,
    n: int = ACCUMULATION_BASELINE_N,
    stride: int = STRIDE,
) -> np.ndarray:
    """One row per window, (n-1) columns per feature in temporal_features: the signed
    (not absolute) consecutive diff of that window's own per-scan medians, one column
    per gap, kept separate rather than summed. Sign is kept on purpose, build_windowed_
    temporal_features' summed absolute version was shown (track_accumulation.md) to
    conflate a real oscillation (regular sign alternation) with a single hump (same
    step magnitudes, no alternation), exactly because absolute value throws sign away.

    Right-aligned to the target (most recent) scan: column -1 is always "diff between
    the two most recent scans", column -2 "the gap before that", etc, regardless of
    how many scans the window actually has, so a fixed MLP input column always means
    the same relative gap. Windows shorter than n zero-pad the older (left) columns,
    there's no real diff to put there. Flattened as [gap0_feat0, gap0_feat1, ...,
    gap(n-2)_feat0, gap(n-2)_feat1, ...].

    Same filtering/sorting/window-slicing as build_windowed_point_sets, see build_
    windowed_temporal_features' docstring for why that keeps row order aligned."""
    mask = df["group"].isin(classes)
    filtered = df.loc[mask]
    feat_matrix = filtered[temporal_features].to_numpy(dtype="float64")

    scan_positions = filtered.groupby(INSTANCE_COLS, sort=False).indices
    scan_arrays = list(scan_positions.values())
    scan_keys = pd.DataFrame(list(scan_positions.keys()), columns=INSTANCE_COLS)
    scan_keys["_scan_idx"] = np.arange(len(scan_keys))
    scan_keys = scan_keys.sort_values(TRACK_COLS + ["timestamp"])

    scan_medians = np.stack([np.median(feat_matrix[positions], axis=0) for positions in scan_arrays])

    n_feat = len(temporal_features)
    n_gaps = n - 1
    rows = []
    for _, track_scans in scan_keys.groupby(TRACK_COLS, sort=False):
        ordered = track_scans["_scan_idx"].to_numpy()
        for i in range(0, len(ordered), stride):
            window_idx = ordered[max(0, i - n + 1) : i + 1]
            row = np.zeros((n_gaps, n_feat), dtype="float32")
            if len(window_idx) > 1:
                diffs = np.diff(scan_medians[window_idx], axis=0)
                row[n_gaps - diffs.shape[0]:] = diffs
            rows.append(row.reshape(-1))

    return np.array(rows, dtype="float32")


def build_windowed_window_length_feature(
    df: pd.DataFrame,
    classes: list[str] = MLP_CLASSES,
    n: int = ACCUMULATION_BASELINE_N,
    stride: int = STRIDE,
) -> np.ndarray:
    """One row per window, one column: how many scans the window actually has (1 to
    n). Since build_windowed_diff_vector_features' padding is right-aligned and
    deterministic, this single number fully determines which of its columns are real
    vs zero-padded, cheaper than a per-gap flag and carries the same information.
    Same filtering/sorting/window-slicing as build_windowed_point_sets."""
    mask = df["group"].isin(classes)
    filtered = df.loc[mask]

    scan_positions = filtered.groupby(INSTANCE_COLS, sort=False).indices
    scan_keys = pd.DataFrame(list(scan_positions.keys()), columns=INSTANCE_COLS)
    scan_keys["_scan_idx"] = np.arange(len(scan_keys))
    scan_keys = scan_keys.sort_values(TRACK_COLS + ["timestamp"])

    rows = []
    for _, track_scans in scan_keys.groupby(TRACK_COLS, sort=False):
        ordered = track_scans["_scan_idx"].to_numpy()
        for i in range(0, len(ordered), stride):
            window_idx = ordered[max(0, i - n + 1) : i + 1]
            rows.append(len(window_idx))

    return np.array(rows, dtype="float32").reshape(-1, 1)


def fit_transition_edges(
    df: pd.DataFrame,
    classes: list[str] = MLP_CLASSES,
    temporal_features: list[str] = TEMPORAL_FEATURES,
    n_bins: int = TRANSITION_BINS,
) -> dict[str, np.ndarray]:
    """Quantile bin edges fit on every scan's own per-feature median (not raw points,
    not window-pooled points, see build_windowed_temporal_features' per-scan-median
    rationale), used to digitize scan-level medians before counting transitions.
    Fit on train scans only, reused as-is for val/test."""
    mask = df["group"].isin(classes)
    filtered = df.loc[mask]
    feat_matrix = filtered[temporal_features].to_numpy(dtype="float64")
    scan_positions = filtered.groupby(INSTANCE_COLS, sort=False).indices
    scan_medians = np.stack([np.median(feat_matrix[positions], axis=0) for positions in scan_positions.values()])
    return {
        feature: np.quantile(scan_medians[:, i], np.linspace(0, 1, n_bins + 1))
        for i, feature in enumerate(temporal_features)
    }


def build_windowed_transition_matrix_features(
    df: pd.DataFrame,
    classes: list[str] = MLP_CLASSES,
    temporal_features: list[str] = TEMPORAL_FEATURES,
    n: int = ACCUMULATION_BASELINE_N,
    stride: int = STRIDE,
    n_bins: int = TRANSITION_BINS,
    edges: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """One row per window, n_bins**2 columns per feature in temporal_features: counts
    of (bin at scan t-1, bin at scan t) across every consecutive pair of scans in the
    window, normalized by number of transitions present. Handles variable window
    length the same clean way build_windowed_temporal_features does, no padding, no
    length flag, that whole confound (see track_accumulation.md's diff-vector
    discussion) doesn't apply here since the matrix shape never depends on window
    length, only its bin count does.

    Keeps sign implicitly, unlike the summed absolute diff: a low-to-high transition
    and a high-to-low transition land in different cells. n_bins defaults coarser
    (4, not the main histogram's 16) since a window only has up to n-1 transitions to
    spread across n_bins**2 cells, 16 bins would leave almost every cell empty.

    Same filtering/sorting/window-slicing as build_windowed_point_sets."""
    if edges is None:
        edges = fit_transition_edges(df, classes, temporal_features, n_bins)

    mask = df["group"].isin(classes)
    filtered = df.loc[mask]
    feat_matrix = filtered[temporal_features].to_numpy(dtype="float64")

    scan_positions = filtered.groupby(INSTANCE_COLS, sort=False).indices
    scan_arrays = list(scan_positions.values())
    scan_keys = pd.DataFrame(list(scan_positions.keys()), columns=INSTANCE_COLS)
    scan_keys["_scan_idx"] = np.arange(len(scan_keys))
    scan_keys = scan_keys.sort_values(TRACK_COLS + ["timestamp"])

    scan_medians = np.stack([np.median(feat_matrix[positions], axis=0) for positions in scan_arrays])
    scan_bins = np.stack(
        [
            np.clip(np.digitize(scan_medians[:, i], edges[feature][1:-1]), 0, n_bins - 1)
            for i, feature in enumerate(temporal_features)
        ],
        axis=1,
    )

    n_feat = len(temporal_features)
    rows = []
    for _, track_scans in scan_keys.groupby(TRACK_COLS, sort=False):
        ordered = track_scans["_scan_idx"].to_numpy()
        for i in range(0, len(ordered), stride):
            window_idx = ordered[max(0, i - n + 1) : i + 1]
            row = np.zeros((n_feat, n_bins, n_bins), dtype="float32")
            if len(window_idx) > 1:
                bins_seq = scan_bins[window_idx]
                prev, nxt = bins_seq[:-1], bins_seq[1:]
                n_trans = prev.shape[0]
                for f in range(n_feat):
                    counts = np.zeros((n_bins, n_bins), dtype="float32")
                    np.add.at(counts, (prev[:, f], nxt[:, f]), 1.0)
                    row[f] = counts / n_trans
            rows.append(row.reshape(-1))

    return np.array(rows, dtype="float32"), edges


def build_windowed_window_ids(
    df: pd.DataFrame,
    classes: list[str] = MLP_CLASSES,
    n: int = ACCUMULATION_BASELINE_N,
    stride: int = STRIDE,
) -> pd.DataFrame:
    """One row per window, in the same order as build_windowed_point_sets(df, classes,
    ..., n, stride, ...): the window's own target (most recent) scan identity
    (sequence_name, track_id, timestamp). Lets post-hoc, per-track prediction
    smoothing group and time-order windows the same way they were built, without
    threading identity columns through every other builder in this module."""
    mask = df["group"].isin(classes)
    filtered = df.loc[mask]

    scan_positions = filtered.groupby(INSTANCE_COLS, sort=False).indices
    scan_keys = pd.DataFrame(list(scan_positions.keys()), columns=INSTANCE_COLS)
    scan_keys["_scan_idx"] = np.arange(len(scan_keys))
    scan_keys = scan_keys.sort_values(TRACK_COLS + ["timestamp"])

    rows = []
    for _, track_scans in scan_keys.groupby(TRACK_COLS, sort=False):
        for i in range(0, len(track_scans), stride):
            target = track_scans.iloc[i]
            rows.append(
                {"sequence_name": target["sequence_name"], "track_id": target["track_id"], "timestamp": target["timestamp"]}
            )

    return pd.DataFrame(rows)


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
    include_temporal_variation: bool = False,
    include_diff_vector: bool = False,
    include_window_length: bool = False,
    include_transition_matrix: bool = False,
    transition_bins: int = TRANSITION_BINS,
    temporal_features: list[str] = TEMPORAL_FEATURES,
    splits: dict[str, list[str]] | None = None,
    output_dir=None,
    run_tag: str = "",
):
    """Single train/test run (no fold sweep, matching the one fixed canonical split
    the comparable DeepReflecs N=5/raw run used) of quantile_bins_5features' own
    config (bin_range="quantile", epochs=200, same 5 features) on windowed
    (accumulated) point sets instead of single-scan instances. include_doppler_spread
    defaults to True (matching quantile_bins_5features), set False for a clean same-
    5-features comparison against DeepReflecs, which never gets a dispersion feature.
    include_temporal_variation appends build_windowed_temporal_features' columns
    (mean absolute consecutive per-scan-median diff, one per temporal_features entry),
    the cheap first test of whether across-scan dynamics carry anything the flat
    histogram (order-0, no scan boundaries) doesn't already see. include_diff_vector
    appends build_windowed_diff_vector_features' columns instead (signed, per-gap,
    not summed), testing whether the sign pattern discarded by the summed absolute
    version carries anything extra. include_window_length appends build_windowed_
    window_length_feature's single column, letting the model tell which of the diff
    vector's right-aligned columns are real vs zero-padded (see track_accumulation.md's
    padding-confound discussion), meant to accompany include_diff_vector.
    include_transition_matrix appends build_windowed_transition_matrix_features'
    columns instead, a bin-to-bin transition count per feature, keeps sign implicitly
    without needing fixed per-gap columns or padding at all. run_tag (e.g.
    "allsensors") gets appended to the default output_dir: the dir name otherwise
    encodes only N/stride/range mode/feature flags, not which points table (sensor
    2 only vs all sensors) built df, so two runs that differ only in that would
    silently share a directory and overwrite each other's model/metrics/plot (hit
    this directly: an all-sensor run clobbered a sensor-2-only run's saved files
    before this tag existed)."""
    if output_dir is None:
        suffix = "" if include_doppler_spread else "_nodopplerspread"
        suffix += "_temporalvar" if include_temporal_variation else ""
        suffix += "_diffvector" if include_diff_vector else ""
        suffix += "_winlen" if include_window_length else ""
        suffix += "_transitionmatrix" if include_transition_matrix else ""
        suffix += f"_{run_tag}" if run_tag else ""
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

    if include_temporal_variation:
        X_train = np.hstack([X_train, build_windowed_temporal_features(train_df, classes, temporal_features, n, stride)])
        X_val = np.hstack([X_val, build_windowed_temporal_features(val_df, classes, temporal_features, n, stride)])
        X_test = np.hstack([X_test, build_windowed_temporal_features(test_df, classes, temporal_features, n, stride)])

    if include_diff_vector:
        X_train = np.hstack([X_train, build_windowed_diff_vector_features(train_df, classes, temporal_features, n, stride)])
        X_val = np.hstack([X_val, build_windowed_diff_vector_features(val_df, classes, temporal_features, n, stride)])
        X_test = np.hstack([X_test, build_windowed_diff_vector_features(test_df, classes, temporal_features, n, stride)])

    if include_window_length:
        X_train = np.hstack([X_train, build_windowed_window_length_feature(train_df, classes, n, stride)])
        X_val = np.hstack([X_val, build_windowed_window_length_feature(val_df, classes, n, stride)])
        X_test = np.hstack([X_test, build_windowed_window_length_feature(test_df, classes, n, stride)])

    if include_transition_matrix:
        X_train_trans, trans_edges = build_windowed_transition_matrix_features(
            train_df, classes, temporal_features, n, stride, transition_bins
        )
        X_val_trans, _ = build_windowed_transition_matrix_features(
            val_df, classes, temporal_features, n, stride, transition_bins, edges=trans_edges
        )
        X_test_trans, _ = build_windowed_transition_matrix_features(
            test_df, classes, temporal_features, n, stride, transition_bins, edges=trans_edges
        )
        X_train = np.hstack([X_train, X_train_trans])
        X_val = np.hstack([X_val, X_val_trans])
        X_test = np.hstack([X_test, X_test_trans])

    print(
        f"windowed MLP N={n} stride={stride} range_sc_mode={range_sc_mode} include_doppler_spread={include_doppler_spread} "
        f"include_temporal_variation={include_temporal_variation} include_diff_vector={include_diff_vector} "
        f"include_window_length={include_window_length} include_transition_matrix={include_transition_matrix}: "
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
