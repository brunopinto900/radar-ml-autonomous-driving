"""track-accumulation branch (track_accumulation.md): DeepReflecs with sliding-window
track accumulation. Each classification still targets one scan, but pools points from
up to the last N scans of that track (stride=1: every scan is a window target) instead
of only the current scan's own points.

Coordinate frame is x_seq/y_seq (global, fixed orientation), not x_cc/y_cc (car frame,
rotates with ego heading scan to scan and is therefore unusable once points from
different scans get pooled together, see track_accumulation.md's coordinate-frame
section), each scan recentered on its own centroid before pooling (add_relative_
features_seq). Within a pooled window, range_sc is overwritten to the target (most
recent) scan's own value, broadcast to every pooled point: keeps the validated
"distance right now" signal clean instead of mixing in stale ranges from earlier scans
in the window (decided when planning this module, see the plan file).

Reuses DeepReflecs' model and training loop unchanged (deepreflecs_classifier.py):
only how point sets get built differs from the single-scan path there. Because
switching to x_seq/y_seq is itself a frame-convention change independent of
accumulation, N=1 (this module, stride=1) is the correct control to compare N=10
against, not deepreflecs_classifier's cached single-scan (x_cc/y_cc) result."""
import json

import numpy as np
import pandas as pd
import torch

from dataloader import RESULTS_DIR
from deepreflecs_classifier import (
    BATCH_SIZE,
    CONV_DIM,
    DEVICE,
    EPOCHS,
    EVAL_BATCH_SIZE,
    LEARNING_RATE,
    POINT_DIM,
    RANDOM_STATE,
    REFLECTION_FEATURES,
    DeepReflecs,
    _evaluate_metrics,
    pad_to_fixed,
    plot_training_curves,
    train_deepreflecs,
)
from feature_distributions import MLP_CLASSES
from mlp_classifier import apply_mlp_class_groups
from separability_probe import class_weights
from sequence_split import load_split
from taxonomy_separability import INSTANCE_COLS

TRACK_COLS = ["sequence_name", "track_id"]
WINDOW_N = 10
STRIDE = 1

POINTS_TABLE_SEQ_PATH = RESULTS_DIR / "data" / "points_table_seq.parquet"
POINTS_TABLE_SEQ_ALLSENSORS_PATH = RESULTS_DIR / "data" / "points_table_seq_allsensors.parquet"
TRACK_ACC_DIR = RESULTS_DIR / "track_accumulation"

# "the accumulation baseline": N=5 (diminishing returns already set in by N=10, see
# comparison_summary.json), raw range_sc (broadcast vs raw made no real difference at
# N=5, raw is the simpler, assumption-free choice). Named/tracked separately from the
# single-fold N-sweep above once validated across the same 6 folds the single-scan
# model was checked against.
ACCUMULATION_BASELINE_N = 5
ACCUMULATION_BASELINE_RANGE_SC_MODE = "raw"
ACCUMULATION_BASELINE_DIR = TRACK_ACC_DIR / "accumulation_baseline"


def add_relative_features_seq(df: pd.DataFrame) -> pd.DataFrame:
    """Per-scan x_rel/y_rel from x_seq/y_seq, recentered on that scan's own centroid.
    Mirrors taxonomy_separability.add_relative_features but on x_seq/y_seq instead of
    x_cc/y_cc, see module docstring for why."""
    df = df.copy()
    group = df.groupby(INSTANCE_COLS)
    df["x_rel"] = df["x_seq"] - group["x_seq"].transform("mean")
    df["y_rel"] = df["y_seq"] - group["y_seq"].transform("mean")
    return df


def check_label_consistency(df: pd.DataFrame) -> None:
    """Sanity check flagged in track_accumulation.md: confirm label_id is constant
    across every scan of a track before trusting one label per pooled window."""
    n_labels = df.groupby(TRACK_COLS)["label_id"].nunique()
    inconsistent = n_labels[n_labels > 1]
    if len(inconsistent):
        print(f"WARNING: {len(inconsistent)}/{len(n_labels)} tracks have more than one label_id across their scans")
        print(inconsistent)
    else:
        print(f"OK: all {len(n_labels)} tracks have a single label_id across every scan")


def build_windowed_point_sets(
    df: pd.DataFrame,
    classes: list[str] = MLP_CLASSES,
    features: list[str] = REFLECTION_FEATURES,
    n: int = WINDOW_N,
    stride: int = STRIDE,
    range_sc_mode: str = "broadcast",
) -> tuple[list[np.ndarray], np.ndarray]:
    """Groups df (must already have `group` and x_rel/y_rel from add_relative_features_
    seq) by track, and for every stride-th scan in a track, pools that scan's window of
    up to `n` most recent scans (this one included) into one point set. Label is the
    target scan's own class.

    range_sc_mode:
    - "broadcast" (default): every pooled point's range_sc is overwritten to the target
      scan's own value (first point in that scan), keeping the validated "distance right
      now" signal clean, at the cost of collapsing whatever real per-point range spread
      existed within the target scan itself.
    - "raw": range_sc is left untouched, each point keeps its own actual measured range
      regardless of which scan (target or pooled-in older one) it came from. Risks
      diluting the current-distance signal with stale ranges from earlier scans, but
      preserves intra-scan range spread (taxonomy_separability.py's range_extent is this
      same quantity, already used elsewhere in the project as a real feature).

    Vectorized the same way build_point_sets is (deepreflecs_classifier.py): column
    selection done once up front, the per-track/per-window loop below only does plain
    numpy fancy indexing and small-array concatenation, no repeated pandas column
    resolution inside the loop."""
    if range_sc_mode not in ("broadcast", "raw"):
        raise ValueError(f"range_sc_mode must be 'broadcast' or 'raw', got {range_sc_mode!r}")

    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    mask = df["group"].isin(classes)
    filtered = df.loc[mask]

    feat_matrix = filtered[features].to_numpy(dtype="float32")
    class_idx = filtered["group"].map(class_to_idx).to_numpy()
    range_sc_values = filtered["range_sc"].to_numpy(dtype="float32")
    range_col = features.index("range_sc")

    scan_positions = filtered.groupby(INSTANCE_COLS, sort=False).indices
    scan_arrays = list(scan_positions.values())
    scan_keys = pd.DataFrame(list(scan_positions.keys()), columns=INSTANCE_COLS)
    scan_keys["_scan_idx"] = np.arange(len(scan_keys))
    scan_keys = scan_keys.sort_values(TRACK_COLS + ["timestamp"])

    point_sets, labels = [], []
    for _, track_scans in scan_keys.groupby(TRACK_COLS, sort=False):
        ordered = track_scans["_scan_idx"].to_numpy()
        for i in range(0, len(ordered), stride):
            window_idx = ordered[max(0, i - n + 1) : i + 1]
            positions = np.concatenate([scan_arrays[j] for j in window_idx])
            feats = feat_matrix[positions].copy()

            target_positions = scan_arrays[ordered[i]]
            if range_sc_mode == "broadcast":
                feats[:, range_col] = range_sc_values[target_positions[0]]

            point_sets.append(feats)
            labels.append(class_idx[target_positions[0]])

    return point_sets, np.array(labels, dtype="int64")


def prepare_windowed_split_point_sets(
    df: pd.DataFrame,
    classes: list[str] = MLP_CLASSES,
    features: list[str] = REFLECTION_FEATURES,
    n: int = WINDOW_N,
    stride: int = STRIDE,
    range_sc_mode: str = "broadcast",
    splits: dict[str, list[str]] | None = None,
    standardize: bool = True,
):
    """Windowed counterpart of deepreflecs_classifier.prepare_split_point_sets: same
    split-then-build-then-pad-then-standardize structure, built from build_windowed_
    point_sets instead of build_point_sets."""
    if splits is None:
        splits = load_split()

    train_df = df.loc[df["sequence_name"].isin(splits["train"])]
    val_df = df.loc[df["sequence_name"].isin(splits["val"])]
    test_df = df.loc[df["sequence_name"].isin(splits["test"])]

    train_sets, y_train = build_windowed_point_sets(train_df, classes, features, n, stride, range_sc_mode)
    val_sets, y_val = build_windowed_point_sets(val_df, classes, features, n, stride, range_sc_mode)
    test_sets, y_test = build_windowed_point_sets(test_df, classes, features, n, stride, range_sc_mode)

    m_max = max(p.shape[0] for p in (*train_sets, *val_sets, *test_sets))
    X_train, mask_train = pad_to_fixed(train_sets, m_max)
    X_val, mask_val = pad_to_fixed(val_sets, m_max)
    X_test, mask_test = pad_to_fixed(test_sets, m_max)

    if standardize:
        mean = X_train[mask_train].mean(axis=0)
        std = X_train[mask_train].std(axis=0)
        std = np.where(std > 0, std, 1.0)
        for X, mask in ((X_train, mask_train), (X_val, mask_val), (X_test, mask_test)):
            X[mask] = (X[mask] - mean) / std

    print(
        f"N={n} stride={stride} range_sc_mode={range_sc_mode}: windows train={len(train_sets)} "
        f"val={len(val_sets)} test={len(test_sets)}, m_max={m_max}"
    )
    return X_train, mask_train, y_train, X_val, mask_val, y_val, X_test, mask_test, y_test, features


def prepare_windowed_split_point_sets_ragged(
    df: pd.DataFrame,
    classes: list[str] = MLP_CLASSES,
    features: list[str] = REFLECTION_FEATURES,
    n: int = WINDOW_N,
    stride: int = STRIDE,
    range_sc_mode: str = "broadcast",
    splits: dict[str, list[str]] | None = None,
):
    """Same split-then-build structure as prepare_windowed_split_point_sets, but never
    calls pad_to_fixed on a whole split: returns ragged (list[np.ndarray]) point sets
    per split instead of one dense (n_windows, m_max, n_features) array. At large N a
    single outlier window can push m_max into the thousands, and padding every window
    in a 350k+-window split to that one shared value OOMs in plain CPU memory before
    any GPU work starts (see track_accumulation.md, "Pooled DeepReflecs point-set at
    N=50"). train_deepreflecs_ragged pads fresh per mini-batch instead, to that batch's
    own max, the same fix already used for the end-to-end variant's
    collate_scan_sequences."""
    if splits is None:
        splits = load_split()

    train_df = df.loc[df["sequence_name"].isin(splits["train"])]
    val_df = df.loc[df["sequence_name"].isin(splits["val"])]
    test_df = df.loc[df["sequence_name"].isin(splits["test"])]

    train_sets, y_train = build_windowed_point_sets(train_df, classes, features, n, stride, range_sc_mode)
    val_sets, y_val = build_windowed_point_sets(val_df, classes, features, n, stride, range_sc_mode)
    test_sets, y_test = build_windowed_point_sets(test_df, classes, features, n, stride, range_sc_mode)

    all_train_points = np.concatenate(train_sets)
    mean = all_train_points.mean(axis=0).astype("float32")
    std = all_train_points.std(axis=0).astype("float32")
    std = np.where(std > 0, std, 1.0).astype("float32")

    global_m_max = max(p.shape[0] for p in (*train_sets, *val_sets, *test_sets))
    print(
        f"N={n} stride={stride} range_sc_mode={range_sc_mode} (ragged): windows "
        f"train={len(train_sets)} val={len(val_sets)} test={len(test_sets)}, "
        f"global m_max would have been {global_m_max} (never padded to it)"
    )
    return train_sets, y_train, val_sets, y_val, test_sets, y_test, mean, std


def _predict_in_batches_ragged(
    model: DeepReflecs, point_sets: list[np.ndarray], batch_size: int = EVAL_BATCH_SIZE
) -> torch.Tensor:
    """Same idea as deepreflecs_classifier._predict_in_batches, but starting from a
    ragged list instead of an already-padded dense array: pads each mini-batch to that
    batch's own max points right before the forward pass, never a whole-split array."""
    model.eval()
    logits = []
    with torch.no_grad():
        for start in range(0, len(point_sets), batch_size):
            batch_sets = point_sets[start : start + batch_size]
            batch_m_max = max(s.shape[0] for s in batch_sets)
            batch_x, batch_mask = pad_to_fixed(batch_sets, batch_m_max)
            batch_x_t = torch.tensor(batch_x, device=DEVICE)
            batch_mask_t = torch.tensor(batch_mask, device=DEVICE)
            logits.append(model(batch_x_t, batch_mask_t))
    return torch.cat(logits, dim=0)


def train_deepreflecs_ragged(
    train_sets: list[np.ndarray],
    y_train: np.ndarray,
    val_sets: list[np.ndarray],
    y_val: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    classes: list[str] = MLP_CLASSES,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    random_state: int = RANDOM_STATE,
    conv_dim: int = CONV_DIM,
    point_dim: int = POINT_DIM,
):
    """Ragged counterpart of deepreflecs_classifier.train_deepreflecs: never builds one
    dense (n_windows, m_max, n_features) array for a whole split, pads fresh per
    mini-batch instead (see prepare_windowed_split_point_sets_ragged). Standardization
    is applied per-window before padding (mean/std fit on train's real points only,
    passed in), instead of via a boolean mask over a padded array."""
    torch.manual_seed(random_state)
    n_features = train_sets[0].shape[1]

    weights_by_class = class_weights(pd.Series([classes[i] for i in y_train]))
    weight_tensor = torch.tensor(
        [weights_by_class[cls] for cls in classes], dtype=torch.float32, device=DEVICE
    )
    print(f"class weights: {dict(zip(classes, weight_tensor.tolist()))}")

    model = DeepReflecs(n_features, num_classes=len(classes), conv_dim=conv_dim, point_dim=point_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor)

    train_std = [(s - mean) / std for s in train_sets]
    val_std = [(s - mean) / std for s in val_sets]
    y_train_t = torch.tensor(y_train, device=DEVICE)
    y_val_t = torch.tensor(y_val, device=DEVICE)

    n = len(train_std)
    history = []
    for epoch in range(epochs):
        model.train()
        perm = torch.from_numpy(np.random.permutation(n))
        epoch_loss, epoch_correct = 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            batch_sets = [train_std[i] for i in idx.tolist()]
            batch_m_max = max(s.shape[0] for s in batch_sets)
            batch_x, batch_mask = pad_to_fixed(batch_sets, batch_m_max)
            batch_x_t = torch.tensor(batch_x, device=DEVICE)
            batch_mask_t = torch.tensor(batch_mask, device=DEVICE)
            batch_y = y_train_t[idx]

            optimizer.zero_grad()
            logits = model(batch_x_t, batch_mask_t)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(idx)
            epoch_correct += (logits.argmax(dim=1) == batch_y).sum().item()

        train_loss = epoch_loss / n
        train_acc = epoch_correct / n

        val_logits = _predict_in_batches_ragged(model, val_std)
        val_acc = (val_logits.argmax(dim=1) == y_val_t).float().mean().item()

        history.append({"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc, "val_acc": val_acc})
        print(f"epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

    return model, history


def run_windowed_training_ragged(
    df: pd.DataFrame,
    n: int = WINDOW_N,
    stride: int = STRIDE,
    range_sc_mode: str = "broadcast",
    classes: list[str] = MLP_CLASSES,
    features: list[str] = REFLECTION_FEATURES,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    random_state: int = RANDOM_STATE,
    output_dir=None,
    splits: dict[str, list[str]] | None = None,
    conv_dim: int = CONV_DIM,
    point_dim: int = POINT_DIM,
):
    """Ragged counterpart of run_windowed_training: same output_dir/artifact naming
    (deepreflecs_model.pt etc.), so anything downstream that loads an already-trained
    pooled encoder (e.g. deepreflecs_rnn_track_accumulation.compute_pooled_embeddings)
    doesn't need to change. Use this instead of run_windowed_training at N large enough
    that the whole-split dense array doesn't fit in system memory (N=50 sensor2
    measured at ~9.9GB for X_train alone, see track_accumulation.md)."""
    if output_dir is None:
        suffix = "" if range_sc_mode == "broadcast" else f"_range{range_sc_mode}"
        output_dir = TRACK_ACC_DIR / f"N{n}_stride{stride}{suffix}"
    if splits is None:
        splits = load_split()

    train_sets, y_train, val_sets, y_val, test_sets, y_test, mean, std = prepare_windowed_split_point_sets_ragged(
        df, classes=classes, features=features, n=n, stride=stride, range_sc_mode=range_sc_mode, splits=splits,
    )

    model, history = train_deepreflecs_ragged(
        train_sets, y_train, val_sets, y_val, mean, std, classes=classes, epochs=epochs,
        batch_size=batch_size, lr=lr, random_state=random_state, conv_dim=conv_dim, point_dim=point_dim,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "deepreflecs_model.pt")
    print(f"Saved {output_dir / 'deepreflecs_model.pt'}")
    plot_training_curves(history, output_dir=output_dir)
    return model, history, test_sets, y_test, mean, std


def evaluate_windowed_test_metrics_ragged(
    model: DeepReflecs,
    test_sets: list[np.ndarray],
    y_test: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    classes: list[str] = MLP_CLASSES,
    output_dir=None,
    n: int = WINDOW_N,
    stride: int = STRIDE,
    range_sc_mode: str = "broadcast",
):
    """Ragged counterpart of evaluate_windowed_test_metrics: same output filenames/
    format, computed via _predict_in_batches_ragged instead of one dense test array."""
    if output_dir is None:
        suffix = "" if range_sc_mode == "broadcast" else f"_range{range_sc_mode}"
        output_dir = TRACK_ACC_DIR / f"N{n}_stride{stride}{suffix}"

    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_recall_fscore_support

    test_std = [(s - mean) / std for s in test_sets]
    y_pred = _predict_in_batches_ragged(model, test_std).argmax(dim=1).cpu().numpy()

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=range(len(classes)), zero_division=0
    )
    metrics_df = pd.DataFrame({"precision": precision, "recall": recall, "f1": f1, "support": support}, index=classes)
    split_name = f"test (N={n}, stride={stride}, range_sc_mode={range_sc_mode}, ragged)"
    print(f"per-class precision/recall/f1 ({split_name}):")
    print(metrics_df.round(3).to_string())

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_json(output_dir / "deepreflecs_test_metrics.json", orient="index", indent=2)
    print(f"Saved {output_dir / 'deepreflecs_test_metrics.json'}")

    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_test, y_pred, labels=range(len(classes)), normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, colorbar=False, values_format=".2f")
    ax.set_title(f"DeepReflecs: {split_name} confusion matrix (row-normalized)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    confusion_matrix_path = output_dir / "deepreflecs_test_confusion_matrix.png"
    fig.savefig(confusion_matrix_path, dpi=150)
    print(f"Saved {confusion_matrix_path}")

    return metrics_df


def run_windowed_training(
    df: pd.DataFrame,
    n: int = WINDOW_N,
    stride: int = STRIDE,
    range_sc_mode: str = "broadcast",
    classes: list[str] = MLP_CLASSES,
    features: list[str] = REFLECTION_FEATURES,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    random_state: int = RANDOM_STATE,
    output_dir=None,
    splits: dict[str, list[str]] | None = None,
    standardize: bool = True,
    conv_dim: int = CONV_DIM,
    point_dim: int = POINT_DIM,
):
    """Windowed counterpart of deepreflecs_classifier.run_training: builds train/val/
    test windowed point sets, trains (or loads from cache if this exact config was
    already run). Separate output_dir per (N, range_sc_mode) (default TRACK_ACC_DIR/
    N{n}_stride{stride}[_rawrange]) so different configs never clobber each other's
    cache; "broadcast" keeps the original directory name (no suffix) so already-run
    N1/N5/N10 broadcast results stay where they are."""
    if output_dir is None:
        suffix = "" if range_sc_mode == "broadcast" else f"_range{range_sc_mode}"
        output_dir = TRACK_ACC_DIR / f"N{n}_stride{stride}{suffix}"
    if splits is None:
        splits = load_split()

    cache_key = {
        "n": n, "stride": stride, "range_sc_mode": range_sc_mode, "classes": classes, "splits": splits,
        "features": features, "standardize": standardize, "conv_dim": conv_dim, "point_dim": point_dim,
        "epochs": epochs, "batch_size": batch_size, "lr": lr, "random_state": random_state,
    }
    history_cache = output_dir / "deepreflecs_training_history.json"
    model_cache = output_dir / "deepreflecs_model.pt"

    X_train, mask_train, y_train, X_val, mask_val, y_val, X_test, mask_test, y_test, _ = (
        prepare_windowed_split_point_sets(
            df, classes=classes, features=features, n=n, stride=stride, range_sc_mode=range_sc_mode,
            splits=splits, standardize=standardize,
        )
    )

    if history_cache.exists() and model_cache.exists():
        cached = json.loads(history_cache.read_text())
        if cached.get("key") == cache_key:
            print(f"{history_cache} already matches this config, loading cached model + history")
            model = DeepReflecs(len(features), num_classes=len(classes), conv_dim=conv_dim, point_dim=point_dim).to(DEVICE)
            model.load_state_dict(torch.load(model_cache, map_location=DEVICE))
            plot_training_curves(cached["history"], output_dir=output_dir)
            return model, cached["history"], X_test, mask_test, y_test
        print(f"{history_cache} doesn't match this config, retraining")

    model, history = train_deepreflecs(
        X_train, mask_train, y_train, X_val, mask_val, y_val, classes=classes, epochs=epochs,
        batch_size=batch_size, lr=lr, random_state=random_state, conv_dim=conv_dim, point_dim=point_dim,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    history_cache.write_text(json.dumps({"key": cache_key, "history": history}, indent=2))
    torch.save(model.state_dict(), model_cache)
    print(f"Saved {history_cache} and {model_cache}")

    plot_training_curves(history, output_dir=output_dir)
    return model, history, X_test, mask_test, y_test


def evaluate_windowed_test_metrics(
    df: pd.DataFrame,
    n: int = WINDOW_N,
    stride: int = STRIDE,
    range_sc_mode: str = "broadcast",
    classes: list[str] = MLP_CLASSES,
    features: list[str] = REFLECTION_FEATURES,
    output_dir=None,
    splits: dict[str, list[str]] | None = None,
    standardize: bool = True,
    conv_dim: int = CONV_DIM,
    point_dim: int = POINT_DIM,
):
    """The one and only time test should be touched for a given (n, stride, range_sc_
    mode) config, same rule as deepreflecs_classifier.evaluate_test_metrics."""
    if output_dir is None:
        suffix = "" if range_sc_mode == "broadcast" else f"_range{range_sc_mode}"
        output_dir = TRACK_ACC_DIR / f"N{n}_stride{stride}{suffix}"
    model_cache = output_dir / "deepreflecs_model.pt"
    if not model_cache.exists():
        raise FileNotFoundError(f"{model_cache} doesn't exist, run run_windowed_training() first")

    _, _, _, _, _, _, X_test, mask_test, y_test, _ = prepare_windowed_split_point_sets(
        df, classes=classes, features=features, n=n, stride=stride, range_sc_mode=range_sc_mode,
        splits=splits, standardize=standardize,
    )
    model = DeepReflecs(len(features), num_classes=len(classes), conv_dim=conv_dim, point_dim=point_dim).to(DEVICE)
    model.load_state_dict(torch.load(model_cache, map_location=DEVICE))

    return _evaluate_metrics(
        model, X_test, mask_test, y_test, classes,
        metrics_cache=output_dir / "deepreflecs_test_metrics.json", model_cache=model_cache,
        confusion_matrix_path=output_dir / "deepreflecs_test_confusion_matrix.png",
        metrics_bar_path=output_dir / "deepreflecs_test_precision_recall_f1.png",
        split_name=f"test (N={n}, stride={stride}, range_sc_mode={range_sc_mode})",
    )


def evaluate_windowed_val_metrics(
    df: pd.DataFrame,
    n: int = WINDOW_N,
    stride: int = STRIDE,
    range_sc_mode: str = "broadcast",
    classes: list[str] = MLP_CLASSES,
    features: list[str] = REFLECTION_FEATURES,
    output_dir=None,
    splits: dict[str, list[str]] | None = None,
    standardize: bool = True,
    conv_dim: int = CONV_DIM,
    point_dim: int = POINT_DIM,
):
    """val counterpart of evaluate_windowed_test_metrics: never retrains, only loads.
    val is for freely comparing candidates (e.g. across the 6 split-sensitivity folds
    below), test stays touched once, same rule as deepreflecs_classifier.
    evaluate_val_metrics."""
    if output_dir is None:
        suffix = "" if range_sc_mode == "broadcast" else f"_range{range_sc_mode}"
        output_dir = TRACK_ACC_DIR / f"N{n}_stride{stride}{suffix}"
    model_cache = output_dir / "deepreflecs_model.pt"
    if not model_cache.exists():
        raise FileNotFoundError(f"{model_cache} doesn't exist, run run_windowed_training() first")

    _, _, _, X_val, mask_val, y_val, _, _, _, _ = prepare_windowed_split_point_sets(
        df, classes=classes, features=features, n=n, stride=stride, range_sc_mode=range_sc_mode,
        splits=splits, standardize=standardize,
    )
    model = DeepReflecs(len(features), num_classes=len(classes), conv_dim=conv_dim, point_dim=point_dim).to(DEVICE)
    model.load_state_dict(torch.load(model_cache, map_location=DEVICE))

    return _evaluate_metrics(
        model, X_val, mask_val, y_val, classes,
        metrics_cache=output_dir / "deepreflecs_val_metrics.json", model_cache=model_cache,
        confusion_matrix_path=output_dir / "deepreflecs_confusion_matrix.png",
        metrics_bar_path=output_dir / "deepreflecs_precision_recall_f1.png",
        split_name=f"val (N={n}, stride={stride}, range_sc_mode={range_sc_mode})",
    )


def run_windowed_split_sensitivity(
    df: pd.DataFrame,
    n: int = ACCUMULATION_BASELINE_N,
    stride: int = STRIDE,
    range_sc_mode: str = ACCUMULATION_BASELINE_RANGE_SC_MODE,
    n_seeds: int = 10,
    base_random_state: int = 0,
    conv_dim: int = CONV_DIM,
    point_dim: int = POINT_DIM,
    output_dir=ACCUMULATION_BASELINE_DIR,
) -> pd.DataFrame:
    """The accumulation baseline (default N=5, stride=1, raw range_sc), trained/
    evaluated across the same 6 sequence-grouped val carves deepreflecs_split_
    sensitivity.py used for the single-scan model. select_best_split's fold identity
    only depends on classes/df/seeds, not on model or feature set, so passing the same
    classes/n_seeds/base_random_state reproduces those exact same 6 folds, same fixed
    test_sequences throughout, only train/val varies per fold, letting this variant's
    noise floor be compared directly against the single-scan model's.

    Evaluates on val, not test, same touch-test-once rule as deepreflecs_split_
    sensitivity.run_split_sensitivity.

    Passes REFLECTION_FEATURES (this model's own features) to select_best_split
    instead of HISTOGRAM_FEATURES (what the single-scan sensitivity check uses):
    HISTOGRAM_FEATURES includes doppler_spread, an instance-level feature this
    pipeline's df was never given (add_relative_features_seq only adds x_rel/y_rel).
    Only the diagnostic train/val KS-match score depends on which features list is
    passed, the actual fold partitions (train/val/test sequences) depend solely on
    classes/df/seeds, so this doesn't change which 6 folds get reproduced."""
    from sequence_split import select_best_split

    candidates = select_best_split(
        df, classes=MLP_CLASSES, features=REFLECTION_FEATURES, n_seeds=n_seeds, base_random_state=base_random_state
    ).drop_duplicates("fold").sort_values("fold")

    rows = []
    for _, row in candidates.iterrows():
        fold = row["fold"]
        splits = {"train": row["train_sequences"], "val": row["val_sequences"], "test": row["test_sequences"]}
        fold_dir = output_dir / f"fold_{fold}"
        print(f"=== accumulation baseline (N={n}, range_sc_mode={range_sc_mode}, conv_dim={conv_dim}, point_dim={point_dim}) fold {fold} (max_ks={row['max_ks']:.4f}) ===")
        run_windowed_training(
            df, n=n, stride=stride, range_sc_mode=range_sc_mode, output_dir=fold_dir, splits=splits,
            conv_dim=conv_dim, point_dim=point_dim,
        )
        metrics_df, _, _ = evaluate_windowed_val_metrics(
            df, n=n, stride=stride, range_sc_mode=range_sc_mode, output_dir=fold_dir, splits=splits,
            conv_dim=conv_dim, point_dim=point_dim,
        )
        macro_f1 = metrics_df["f1"].mean()
        print(f"fold {fold} macro F1: {macro_f1:.4f}")
        rows.append({"fold": fold, "max_ks": row["max_ks"], "macro_f1": macro_f1})

        # Each fold allocates fresh, large (X_train/X_val, up to ~m_max x n_instances)
        # GPU tensors inside run_windowed_training/evaluate_windowed_val_metrics; once
        # those go out of scope here, PyTorch's caching allocator frees them internally
        # but does not return the memory to the driver, so repeated large alloc/free
        # cycles across 6 sequential folds in one process can fragment the cache until a
        # later fold's allocation fails even though nominal usage is well under the
        # card's limit (observed twice: identical "CUDA error: unknown error" at the
        # same fold boundary on a 6GB GPU, both times with nothing else running).
        # empty_cache() releases those unused cached blocks back to the driver between
        # folds, keeping fragmentation from accumulating.
        torch.cuda.empty_cache()

    summary = pd.DataFrame(rows)
    print()
    print(f"macro F1 range: {summary['macro_f1'].min():.3f} - {summary['macro_f1'].max():.3f}, std: {summary['macro_f1'].std():.3f}")
    print(summary.to_string(index=False))

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "split_sensitivity_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")
    return summary


def compare_control_vs_windowed(
    classes: list[str] = MLP_CLASSES,
    n_control: int = 1,
    n_experiment: int = WINDOW_N,
    stride: int = STRIDE,
    output_dir=TRACK_ACC_DIR,
):
    """Compares the N=1 control against the N-scan experiment, both built through this
    module's own (x_seq/y_seq, frame-consistent) pipeline, not against deepreflecs_
    classifier's cached x_cc/y_cc single-scan result, see module docstring for why that
    comparison would be confounded."""
    import matplotlib.pyplot as plt

    control_dir = TRACK_ACC_DIR / f"N{n_control}_stride{stride}"
    experiment_dir = TRACK_ACC_DIR / f"N{n_experiment}_stride{stride}"
    control_metrics_path = control_dir / "deepreflecs_test_metrics.json"
    experiment_metrics_path = experiment_dir / "deepreflecs_test_metrics.json"

    for path, name in [(control_metrics_path, f"N={n_control} control"), (experiment_metrics_path, f"N={n_experiment} experiment")]:
        if not path.exists():
            raise FileNotFoundError(f"{path} doesn't exist, run evaluate_windowed_test_metrics() for {name} first")

    control_df = pd.read_json(control_metrics_path, orient="index").loc[classes]
    experiment_df = pd.read_json(experiment_metrics_path, orient="index").loc[classes]

    def summarize(metrics_df):
        support = metrics_df["support"]
        accuracy = (metrics_df["recall"] * support).sum() / support.sum()
        return {"accuracy": accuracy, "macro_f1": metrics_df["f1"].mean()}

    summary = {
        f"N{n_control}_control": summarize(control_df),
        f"N{n_experiment}_experiment": summarize(experiment_df),
    }
    print("test set comparison (N=1 control vs windowed experiment):")
    print(pd.DataFrame(summary).round(4).to_string())

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved {summary_path}")

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(classes))
    width = 0.35
    ax.bar(x - width / 2, control_df["f1"], width, label=f"N={n_control} control", edgecolor="k")
    ax.bar(x + width / 2, experiment_df["f1"], width, label=f"N={n_experiment} experiment", edgecolor="k")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 (test)")
    ax.set_title(f"DeepReflecs: N={n_control} control vs N={n_experiment} sliding-window, per-class test F1")
    ax.legend()
    fig.tight_layout()

    bar_path = output_dir / "comparison_f1_bar.png"
    fig.savefig(bar_path, dpi=150)
    print(f"Saved {bar_path}")

    return summary, fig


if __name__ == "__main__":
    from build_points_table import build_and_save_points_table

    df = build_and_save_points_table(table_path=POINTS_TABLE_SEQ_PATH)
    check_label_consistency(df)
    df = add_relative_features_seq(df)
    df = apply_mlp_class_groups(df)

    for n in (1, WINDOW_N):
        run_windowed_training(df, n=n, stride=STRIDE)
        evaluate_windowed_test_metrics(df, n=n, stride=STRIDE)

    compare_control_vs_windowed(n_control=1, n_experiment=WINDOW_N, stride=STRIDE)
