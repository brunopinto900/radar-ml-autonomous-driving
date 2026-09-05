"""Baseline MLP on the histogram-encoded feature vector (Design_Decisions.md decision 6),
trained/evaluated on the fixed sequence-grouped split (decision 5). Bin edges are fit on train
only and applied unchanged to val/test (histogram_separability.fit_bin_edges), unlike the
separability probes, which recomputed edges from whatever pool was passed in each time (a fine
shortcut for a quick probe, wrong for the actual model, edges are a fitted preprocessing
parameter, val/test must never influence them).

Trains on train, tracks train/val loss+accuracy per epoch (not test, which is checked once at
the end via evaluate_test, never watched during training/tuning). Caches training history and
model weights, keyed by the exact architecture/hyperparameter config, so a repeat call with
the same config skips straight to the cached result."""
import json

import numpy as np
import pandas as pd
import torch
from torch import nn

from dataloader import MLP_CLASS_GROUPS, RESULTS_DIR
from feature_distributions import INSTANCE_LEVEL_FEATURES, MLP_CLASSES, POINT_LEVEL_FEATURES
from histogram_separability import build_histogram_features, build_stat_features, fit_bin_edges
from separability_probe import class_weights
from sequence_split import load_split

# --- hyperparameters ---
N_BINS = 16  # Design_Decisions.md decision 3
HIDDEN_DIM = 16
LEARNING_RATE = 4e-5
EPOCHS = 50
BATCH_SIZE = 128
RANDOM_STATE = 0
# ------------------------

MLP_DIR = RESULTS_DIR / "mlp"
HISTORY_FILENAME = "mlp_training_history.json"
MODEL_FILENAME = "mlp_model.pt"
CURVES_FILENAME = "mlp_training_curves.png"
VAL_METRICS_FILENAME = "mlp_val_metrics.json"
CONFUSION_MATRIX_FILENAME = "mlp_confusion_matrix.png"
METRICS_BAR_FILENAME = "mlp_precision_recall_f1.png"
POINT_COUNT_CURVE_FILENAME = "mlp_point_count_curve.png"
TEST_METRICS_FILENAME = "mlp_test_metrics.json"
TEST_CONFUSION_MATRIX_FILENAME = "mlp_test_confusion_matrix.png"
TEST_METRICS_BAR_FILENAME = "mlp_test_precision_recall_f1.png"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MLP(nn.Module):
    """`n_hidden_layers` Linear(+BatchNorm1d)+ReLU blocks (each hidden_dim wide, optionally
    followed by Dropout(dropout)), then a final Linear mapping to one logit per class.
    n_hidden_layers=2, dropout=0.0, batch_norm=False reproduces the original 3-linear-layer
    baseline exactly. batch_norm exists because a plain (unnormalized) stack of 10
    Linear+ReLU+Dropout layers at hidden_dim=16 collapsed to predicting the majority class,
    gradient signal degrading across depth with nothing keeping each layer's activations in a
    well-scaled range (MLP_Decisions_and_Findings.md's depth ablation)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = HIDDEN_DIM,
        num_classes: int = len(MLP_CLASSES),
        n_hidden_layers: int = 2,
        dropout: float = 0.0,
        batch_norm: bool = False,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def apply_mlp_class_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Maps raw label_name to the current working class (dataloader.MLP_CLASS_GROUPS, bus
    merged into large_vehicle), mirroring feature_distributions.apply_class_groups but for the
    taxonomy this module actually trains on by default."""
    df = df.copy()
    df["group"] = df["label_name"].map(MLP_CLASS_GROUPS)
    return df.loc[df["group"].notna()]


def prepare_split_features(
    df: pd.DataFrame,
    classes: list[str] = MLP_CLASSES,
    n_bins: int = N_BINS,
    features: list[str] = POINT_LEVEL_FEATURES,
    extra_features: list[str] = INSTANCE_LEVEL_FEATURES,
    splits: dict[str, list[str]] | None = None,
    normalize: bool = True,
    feature_stats: dict[str, list[str]] | None = None,
    bin_range: str = "percentile",
    standardize_extra: bool = False,
):
    """Fits bin edges on the train split only, then encodes train/val/test with those same
    edges.

    - `classes` (default MLP_CLASSES): encode a different class taxonomy against the same
      fixed, sequence-level split (see mlp_variants.py).
    - `features` (default POINT_LEVEL_FEATURES): binned into `n_bins` columns each.
    - `extra_features` (default doppler_spread): appended unbinned. Swap doppler_spread for
      another point-level feature by moving it into `features` and passing `extra_features=[]`
      (see mlp_variants.py's "range_sc" variant).
    - `normalize` (default True): each feature's bins hold the fraction of the instance's
      points landing there (default) or raw counts (histogram_separability.
      build_histogram_features), raw counts embed point count back into the encoding
      implicitly, at the cost of putting sparse/busy instances on different scales.
    - `bin_range` (default "percentile"): how each feature's outer bin range is determined
      before splitting into `n_bins` equal-width bins, see histogram_separability.fit_bin_edges
      for "percentile" vs "gaussian" ([mean-2std, mean+2std]).
    - `feature_stats` (default None): replaces the histogram encoding entirely with explicit
      per-instance statistics (histogram_separability.build_stat_features), e.g. {"rcs":
      ["mean","median","std"], "radial": ["std"]}. When set, `n_bins`/`features`/`normalize`/
      `bin_range` are unused, there are no bin edges to fit since each instance's stats depend
      only on its own points.
    - `standardize_extra` (default False): z-score standardizes (train-fit mean/std, applied
      unchanged to val/test, same fit-on-train-only discipline as the bin edges) whichever
      columns aren't already bounded [0,1] fractions: in histogram mode that's just
      `extra_features` (e.g. `doppler_spread`, train range 0-59.2 vs. the bin columns' [0,1]);
      in feature_stats mode it's every column, none of them are fraction bins to begin with.
    - `splits` (default None): uses the project's standing fixed split (load_split()). Pass an
      explicit {"train"/"val"/"test": [sequence_name, ...]} dict to evaluate a different
      candidate split (sequence_split.select_best_split) without touching the standing one.

    Returns (X_train, y_train, X_val, y_val, X_test, y_test, feature_cols) as numpy arrays. X
    float32, y integer class indices in `classes` order."""
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    if splits is None:
        splits = load_split()

    train_df = df.loc[df["sequence_name"].isin(splits["train"])]
    val_df = df.loc[df["sequence_name"].isin(splits["val"])]
    test_df = df.loc[df["sequence_name"].isin(splits["test"])]

    if feature_stats is None:
        edges = fit_bin_edges(train_df, n_bins, features, range_method=bin_range)

        def to_xy(split_df):
            hist = build_histogram_features(
                split_df, n_bins, classes, features, edges=edges, extra_features=extra_features, normalize=normalize
            )
            feature_cols = [c for c in hist.columns if c not in ("sequence_name", "group")]
            X = hist[feature_cols].to_numpy(dtype="float32")
            y = hist["group"].map(class_to_idx).to_numpy(dtype="int64")
            return X, y, feature_cols
    else:
        def to_xy(split_df):
            stats = build_stat_features(split_df, classes, feature_stats, extra_features=extra_features)
            feature_cols = [c for c in stats.columns if c not in ("sequence_name", "group")]
            X = stats[feature_cols].to_numpy(dtype="float32")
            y = stats["group"].map(class_to_idx).to_numpy(dtype="int64")
            return X, y, feature_cols

    X_train, y_train, feature_cols = to_xy(train_df)
    X_val, y_val, _ = to_xy(val_df)
    X_test, y_test, _ = to_xy(test_df)

    if standardize_extra:
        n_unbounded = len(extra_features) if feature_stats is None else len(feature_cols)
        cols = slice(len(feature_cols) - n_unbounded, len(feature_cols))
        mean = X_train[:, cols].mean(axis=0)
        std = X_train[:, cols].std(axis=0)
        std = np.where(std > 0, std, 1.0)
        for X in (X_train, X_val, X_test):
            X[:, cols] = (X[:, cols] - mean) / std

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    classes: list[str] = MLP_CLASSES,
    epochs: int = EPOCHS,
    batch_size: int | None = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    random_state: int = RANDOM_STATE,
    hidden_dim: int = HIDDEN_DIM,
    n_hidden_layers: int = 2,
    dropout: float = 0.0,
    weight_decay: float = 0.0,
    batch_norm: bool = False,
):
    """Trains the MLP with Adam, class-count-weighted cross-entropy. `classes` (default
    MLP_CLASSES) is the class list y_train/y_val's integer labels index into.
    `n_hidden_layers`/`dropout`/`weight_decay`/`batch_norm` default to 2/0.0/0.0/False,
    reproducing the original 2-hidden-layer, no-regularization baseline exactly."""
    torch.manual_seed(random_state)

    weights_by_class = class_weights(pd.Series([classes[i] for i in y_train]))
    weight_tensor = torch.tensor(
        [weights_by_class[cls] for cls in classes], dtype=torch.float32, device=DEVICE
    )
    print(f"class weights: {dict(zip(classes, weight_tensor.tolist()))}")

    X_train_t = torch.tensor(X_train, device=DEVICE)
    y_train_t = torch.tensor(y_train, device=DEVICE)
    X_val_t = torch.tensor(X_val, device=DEVICE)
    y_val_t = torch.tensor(y_val, device=DEVICE)

    model = MLP(
        input_dim=X_train.shape[1], hidden_dim=hidden_dim, num_classes=len(classes),
        n_hidden_layers=n_hidden_layers, dropout=dropout, batch_norm=batch_norm,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    n = len(X_train_t)
    step = n if batch_size is None else batch_size

    history = []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        epoch_loss, epoch_correct = 0.0, 0
        for start in range(0, n, step):
            idx = perm[start : start + step]
            X_batch, y_batch = X_train_t[idx], y_train_t[idx]

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(idx)
            epoch_correct += (logits.argmax(dim=1) == y_batch).sum().item()

        train_loss = epoch_loss / n
        train_acc = epoch_correct / n

        model.eval()
        with torch.no_grad():
            val_acc = (model(X_val_t).argmax(dim=1) == y_val_t).float().mean().item()

        history.append({"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc, "val_acc": val_acc})
        print(f"epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

    return model, history


def evaluate_test(model, X_test: np.ndarray, y_test: np.ndarray) -> float:
    """The one and only time test should be touched: call this once, after training/tuning is
    fully finished, never inside the training loop or a hyperparameter search."""
    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, device=DEVICE)
        y_test_t = torch.tensor(y_test, device=DEVICE)
        test_acc = (model(X_test_t).argmax(dim=1) == y_test_t).float().mean().item()
    print(f"test accuracy: {test_acc:.4f}")
    return test_acc


def evaluate_test_metrics(
    df: pd.DataFrame,
    classes: list[str] = MLP_CLASSES,
    output_dir=MLP_DIR,
    features: list[str] = POINT_LEVEL_FEATURES,
    extra_features: list[str] = INSTANCE_LEVEL_FEATURES,
    splits: dict[str, list[str]] | None = None,
    normalize: bool = True,
    feature_stats: dict[str, list[str]] | None = None,
    bin_range: str = "percentile",
    standardize_extra: bool = False,
    hidden_dim: int = HIDDEN_DIM,
    n_hidden_layers: int = 2,
    dropout: float = 0.0,
    batch_norm: bool = False,
):
    """The one and only time test should be touched (same rule as evaluate_test, this is that
    check extended to full per-class metrics instead of just accuracy): per-class precision/
    recall/f1 + confusion matrix on test, computed from the cached trained model in
    `output_dir`, never retrains, only loads. Call once, after training/tuning is fully
    finished, never during model selection, never to compare against val's own number and then
    go back and tune further, that defeats the entire point of holding test out.

    Same mechanics as evaluate_val_metrics (see its docstring for parameter meaning), swapped to
    the test split, with its own cache files (TEST_METRICS_FILENAME/TEST_CONFUSION_MATRIX_FILENAME/
    TEST_METRICS_BAR_FILENAME) so this never overwrites the val-side artifacts.

    Returns (metrics_df, confusion_matrix_fig, bar_fig)."""
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_recall_fscore_support

    model_cache = output_dir / MODEL_FILENAME
    test_metrics_cache = output_dir / TEST_METRICS_FILENAME
    confusion_matrix_path = output_dir / TEST_CONFUSION_MATRIX_FILENAME
    metrics_bar_path = output_dir / TEST_METRICS_BAR_FILENAME

    if not model_cache.exists():
        raise FileNotFoundError(f"{model_cache} doesn't exist, run run_training() first")

    _, _, _, _, X_test, y_test, _ = prepare_split_features(
        df, classes=classes, features=features, extra_features=extra_features, splits=splits,
        normalize=normalize, feature_stats=feature_stats, bin_range=bin_range,
        standardize_extra=standardize_extra,
    )

    model = MLP(
        input_dim=X_test.shape[1], hidden_dim=hidden_dim, num_classes=len(classes),
        n_hidden_layers=n_hidden_layers, dropout=dropout, batch_norm=batch_norm,
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_cache, map_location=DEVICE))
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.tensor(X_test, device=DEVICE)).argmax(dim=1).cpu().numpy()

    if test_metrics_cache.exists() and test_metrics_cache.stat().st_mtime >= model_cache.stat().st_mtime:
        metrics_df = pd.read_json(test_metrics_cache, orient="index")
        print(f"{test_metrics_cache} already cached, reusing")
    else:
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, labels=range(len(classes)), zero_division=0
        )
        metrics_df = pd.DataFrame(
            {"precision": precision, "recall": recall, "f1": f1, "support": support}, index=classes
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_df.to_json(test_metrics_cache, orient="index", indent=2)
        print(f"Saved {test_metrics_cache}")
    print("per-class precision/recall/f1 (test):")
    print(metrics_df.round(3).to_string())

    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_test, y_pred, labels=range(len(classes)), normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, colorbar=False, values_format=".2f")
    ax.set_title("MLP: test confusion matrix (row-normalized)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(confusion_matrix_path, dpi=150)
    print(f"Saved {confusion_matrix_path}")

    bar_fig, bar_ax = plt.subplots(figsize=(9, 5))
    metrics_df[["precision", "recall", "f1"]].plot(kind="bar", ax=bar_ax, edgecolor="k")
    bar_ax.set_ylim(0, 1)
    bar_ax.set_ylabel("score")
    bar_ax.set_title("MLP: per-class precision/recall/f1 (test)")
    bar_ax.legend(loc="lower right")
    plt.setp(bar_ax.get_xticklabels(), rotation=45, ha="right")
    bar_fig.tight_layout()

    bar_fig.savefig(metrics_bar_path, dpi=150)
    print(f"Saved {metrics_bar_path}")

    return metrics_df, fig, bar_fig


def evaluate_val_metrics(
    df: pd.DataFrame,
    classes: list[str] = MLP_CLASSES,
    output_dir=MLP_DIR,
    features: list[str] = POINT_LEVEL_FEATURES,
    extra_features: list[str] = INSTANCE_LEVEL_FEATURES,
    splits: dict[str, list[str]] | None = None,
    normalize: bool = True,
    feature_stats: dict[str, list[str]] | None = None,
    bin_range: str = "percentile",
    standardize_extra: bool = False,
    hidden_dim: int = HIDDEN_DIM,
    n_hidden_layers: int = 2,
    dropout: float = 0.0,
    batch_norm: bool = False,
):
    """Per-class precision/recall/f1 + confusion matrix on val, computed from the cached
    trained model in `output_dir`, never retrains, only loads.

    - `classes`/`hidden_dim`/`n_hidden_layers`/`dropout`/`batch_norm` (default
      MLP_CLASSES/HIDDEN_DIM/2/0.0/False) must match what the cached model at `output_dir` was
      trained with, so the reconstructed architecture matches the saved weights (weight_decay
      is optimizer-only and isn't needed here, it doesn't affect architecture).
    - Rebuilding val's feature vectors (bin edges fit on train, applied to val) and running one
      forward pass is cheap regardless (not training), so it's redone each call; only the
      actual training step is skipped, via the model cache.
    - Uses val, not test, for the same reason the training curves did, test stays untouched
      until a deliberate one-time check.
    - Numeric metrics are cached to output_dir/mlp_val_metrics.json, reused only if that file
      is at least as new as model.pt, otherwise a retrained model in the same output_dir would
      silently serve stale metrics from whatever was trained there before.
    - Confusion matrix and bar plots are always regenerated fresh from the loaded model, cheap
      regardless.

    Returns (metrics_df, confusion_matrix_fig, bar_fig)."""
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_recall_fscore_support

    model_cache = output_dir / MODEL_FILENAME
    val_metrics_cache = output_dir / VAL_METRICS_FILENAME
    confusion_matrix_path = output_dir / CONFUSION_MATRIX_FILENAME
    metrics_bar_path = output_dir / METRICS_BAR_FILENAME

    if not model_cache.exists():
        raise FileNotFoundError(f"{model_cache} doesn't exist, run run_training() first")

    _, _, X_val, y_val, _, _, _ = prepare_split_features(
        df, classes=classes, features=features, extra_features=extra_features, splits=splits,
        normalize=normalize, feature_stats=feature_stats, bin_range=bin_range,
        standardize_extra=standardize_extra,
    )

    model = MLP(
        input_dim=X_val.shape[1], hidden_dim=hidden_dim, num_classes=len(classes),
        n_hidden_layers=n_hidden_layers, dropout=dropout, batch_norm=batch_norm,
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_cache, map_location=DEVICE))
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.tensor(X_val, device=DEVICE)).argmax(dim=1).cpu().numpy()

    if val_metrics_cache.exists() and val_metrics_cache.stat().st_mtime >= model_cache.stat().st_mtime:
        metrics_df = pd.read_json(val_metrics_cache, orient="index")
        print(f"{val_metrics_cache} already cached, reusing")
    else:
        precision, recall, f1, support = precision_recall_fscore_support(
            y_val, y_pred, labels=range(len(classes)), zero_division=0
        )
        metrics_df = pd.DataFrame(
            {"precision": precision, "recall": recall, "f1": f1, "support": support}, index=classes
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_df.to_json(val_metrics_cache, orient="index", indent=2)
        print(f"Saved {val_metrics_cache}")
    print("per-class precision/recall/f1 (val):")
    print(metrics_df.round(3).to_string())

    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_val, y_pred, labels=range(len(classes)), normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, colorbar=False, values_format=".2f")
    ax.set_title("MLP: val confusion matrix (row-normalized)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(confusion_matrix_path, dpi=150)
    print(f"Saved {confusion_matrix_path}")

    bar_fig, bar_ax = plt.subplots(figsize=(9, 5))
    metrics_df[["precision", "recall", "f1"]].plot(kind="bar", ax=bar_ax, edgecolor="k")
    bar_ax.set_ylim(0, 1)
    bar_ax.set_ylabel("score")
    bar_ax.set_title("MLP: per-class precision/recall/f1 (val)")
    bar_ax.legend(loc="lower right")
    plt.setp(bar_ax.get_xticklabels(), rotation=45, ha="right")
    bar_fig.tight_layout()

    bar_fig.savefig(metrics_bar_path, dpi=150)
    print(f"Saved {metrics_bar_path}")

    return metrics_df, fig, bar_fig


def evaluate_by_point_count(
    df: pd.DataFrame,
    classes: list[str] = MLP_CLASSES,
    output_dir=MLP_DIR,
    features: list[str] = POINT_LEVEL_FEATURES,
    extra_features: list[str] = INSTANCE_LEVEL_FEATURES,
    splits: dict[str, list[str]] | None = None,
    feature_stats: dict[str, list[str]] | None = None,
    bin_range: str = "percentile",
    standardize_extra: bool = False,
    hidden_dim: int = HIDDEN_DIM,
    n_hidden_layers: int = 2,
    dropout: float = 0.0,
    batch_norm: bool = False,
    bins: tuple[int, ...] = (1, 2, 3, 4, 5, 10, 999),
) -> pd.DataFrame:
    """Buckets val predictions from the cached model in `output_dir` by instance point count
    (n_points) and reports per-bucket support, accuracy, macro F1, and per-class F1 + support
    (support_<class> is that class's true instance count within the bucket, not a prediction
    count, it's what makes a per-class F1 like `f1_pedestrian=0.000` at the sparsest/densest
    buckets legible as "zero real instances there" rather than "the model failed"). Answers a
    different question than the n_points/raw_counts feature tests (section 8): those asked
    whether handing the network point count as an input helps; this asks whether point count
    itself, independent of any feature, explains why error is high, i.e. is sparsity a real
    driver of the ceiling, checked directly against the model's actual predictions rather than
    inferred from six null feature-design results. Never retrains, only loads (same cached
    model `evaluate_val_metrics` uses). `bins` (default 1/2/3/4/5/6-10/11+) sets the bucket
    edges, passed to `pandas.cut` with `bins=(0, *bins)`.

    n_points isn't necessarily part of `features`/`extra_features` (it isn't for baseline), so
    it's fetched via a second, separate call to prepare_split_features with n_points appended
    to extra_features, always with standardize_extra=False regardless of what the model itself
    was trained with, n_points here is the bucketing key, not a model input, so it must stay in
    raw units. The model's own input vector comes from a first, separate call using the real
    features/extra_features/standardize_extra, so the two never interfere."""
    from sklearn.metrics import f1_score, precision_recall_fscore_support

    model_cache = output_dir / MODEL_FILENAME
    if not model_cache.exists():
        raise FileNotFoundError(f"{model_cache} doesn't exist, run run_training() first")

    _, _, X_model, y_val, _, _, _ = prepare_split_features(
        df, classes=classes, features=features, extra_features=extra_features, splits=splits,
        feature_stats=feature_stats, bin_range=bin_range, standardize_extra=standardize_extra,
    )

    # n_points is fetched separately, always unstandardized, it's the bucketing key, not a model
    # input, so standardize_extra (if set) must not touch it regardless of what the model itself
    # was trained on
    raw_extra_features = extra_features if "n_points" in extra_features else [*extra_features, "n_points"]
    _, _, X_raw, _, _, _, raw_feature_cols = prepare_split_features(
        df, classes=classes, features=features, extra_features=raw_extra_features, splits=splits,
        feature_stats=feature_stats, bin_range=bin_range,
    )
    n_points_col = X_raw[:, raw_feature_cols.index("n_points")].astype(int)

    model = MLP(
        input_dim=X_model.shape[1], hidden_dim=hidden_dim, num_classes=len(classes),
        n_hidden_layers=n_hidden_layers, dropout=dropout, batch_norm=batch_norm,
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_cache, map_location=DEVICE))
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.tensor(X_model, device=DEVICE)).argmax(dim=1).cpu().numpy()

    labels = [str(bins[0])] + [
        str(bins[i]) if bins[i] == bins[i - 1] + 1 else f"{bins[i - 1] + 1}-{bins[i]}"
        for i in range(1, len(bins) - 1)
    ] + [f"{bins[-2] + 1}+"]
    bucket = pd.cut(n_points_col, bins=[0, *bins], labels=labels)

    rows = []
    for lbl in labels:
        mask = np.asarray(bucket == lbl)
        n = int(mask.sum())
        row = {"bucket": lbl, "n": n}
        if n:
            row["accuracy"] = (y_pred[mask] == y_val[mask]).mean()
            row["macro_f1"] = f1_score(y_val[mask], y_pred[mask], labels=range(len(classes)), average="macro", zero_division=0)
            _, _, per_class_f1, per_class_support = precision_recall_fscore_support(
                y_val[mask], y_pred[mask], labels=range(len(classes)), zero_division=0
            )
            row.update({f"f1_{cls}": v for cls, v in zip(classes, per_class_f1)})
            row.update({f"support_{cls}": int(v) for cls, v in zip(classes, per_class_support)})
        rows.append(row)

    summary = pd.DataFrame(rows).set_index("bucket")
    print(summary.round(3).to_string())
    return summary


def format_point_count_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    """Reformats evaluate_by_point_count's summary to show each metric's change off the
    sparsest bucket, not just its raw value: every value except the first bucket (n_points=1,
    the baseline row) is shown as "value (+/-delta)", delta always relative to that first
    bucket. `n` and `support_<class>` (sample sizes) are left as plain integers, counts rather
    than quality metrics that should carry a delta."""
    plain_cols = [c for c in summary.columns if c == "n" or c.startswith("support_")]
    value_cols = [c for c in summary.columns if c not in plain_cols]
    baseline = summary.iloc[0]
    out = summary[plain_cols].copy()
    for col in value_cols:
        cells = [f"{summary[col].iloc[0]:.3f}"]
        for v in summary[col].iloc[1:]:
            delta = v - baseline[col]
            sign = "+" if delta >= 0 else ""
            cells.append(f"{v:.3f} ({sign}{delta:.3f})")
        out[col] = cells
    return out


def plot_point_count_curve(summary: pd.DataFrame, classes: list[str] = MLP_CLASSES, output_dir=MLP_DIR):
    """Plots evaluate_by_point_count's summary as one curve per class (accuracy and macro F1 as
    dashed reference lines), point count bucket on the x-axis. Overlaying every class on one
    plot reads faster than the raw table, especially for spotting which classes rise/fall
    together vs. which stay flat."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(summary))
    ax.plot(x, summary["accuracy"], label="accuracy", color="0.5", linestyle="--", marker="o")
    ax.plot(x, summary["macro_f1"], label="macro F1", color="k", linestyle="--", marker="o")
    for cls in classes:
        ax.plot(x, summary[f"f1_{cls}"], label=cls, marker="o")
    ax.set_xticks(list(x))
    ax.set_xticklabels(summary.index)
    ax.set_ylim(0, 1)
    ax.set_xlabel("instance point count (bucket)")
    ax.set_ylabel("score")
    ax.set_title("MLP: val performance by instance point count")
    ax.legend(loc="center right", fontsize=8)
    fig.tight_layout()

    path = output_dir / POINT_COUNT_CURVE_FILENAME
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    return fig


def plot_training_curves(history: list[dict], output_dir=MLP_DIR):
    import matplotlib.pyplot as plt

    df = pd.DataFrame(history)
    fig, ax_acc = plt.subplots(figsize=(8, 5))
    ax_acc.plot(df["epoch"], df["train_acc"], label="train accuracy", color="tab:blue")
    ax_acc.plot(df["epoch"], df["val_acc"], label="val accuracy", color="tab:green")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_ylim(0, 1)

    ax_loss = ax_acc.twinx()
    ax_loss.plot(df["epoch"], df["train_loss"], label="train cost (cross-entropy)", color="tab:red", linestyle="--")
    ax_loss.set_ylabel("cost")

    lines1, labels1 = ax_acc.get_legend_handles_labels()
    lines2, labels2 = ax_loss.get_legend_handles_labels()
    ax_acc.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    ax_acc.set_title("MLP training curves (train vs val accuracy, train cost)")
    fig.tight_layout()

    curves_path = output_dir / CURVES_FILENAME
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(curves_path, dpi=150)
    print(f"Saved {curves_path}")
    return fig


def run_training(
    df: pd.DataFrame,
    classes: list[str] = MLP_CLASSES,
    epochs: int = EPOCHS,
    batch_size: int | None = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    random_state: int = RANDOM_STATE,
    output_dir=MLP_DIR,
    features: list[str] = POINT_LEVEL_FEATURES,
    extra_features: list[str] = INSTANCE_LEVEL_FEATURES,
    splits: dict[str, list[str]] | None = None,
    normalize: bool = True,
    feature_stats: dict[str, list[str]] | None = None,
    bin_range: str = "percentile",
    standardize_extra: bool = False,
    hidden_dim: int = HIDDEN_DIM,
    n_hidden_layers: int = 2,
    dropout: float = 0.0,
    weight_decay: float = 0.0,
    batch_norm: bool = False,
):
    """Builds train/val/test from the fixed split, trains (or loads from cache if this exact
    config was already run), plots train-vs-val curves.

    - `classes` (default MLP_CLASSES): train on a different class taxonomy against the same
      fixed split (see mlp_variants.py).
    - `features`/`extra_features`/`normalize`/`feature_stats`/`bin_range`/`standardize_extra`
      (see prepare_split_features): swap the feature set or encoding, e.g. binning range_sc in
      place of doppler_spread, raw per-bin counts instead of fractions, explicit per-instance
      statistics instead of histograms entirely, a mean/std-based bin range instead of
      percentile-based, or z-score standardizing the non-fraction columns.
    - `splits`: evaluate a different candidate split instead of the standing fixed one
      (sequence_split.select_best_split).
    - `hidden_dim`/`n_hidden_layers`: change model capacity (width/depth).
    - `dropout`/`weight_decay`/`batch_norm`: add regularization/stabilization (default
      0.0/0.0/False, off, reproducing the original baseline).

    Writes to `output_dir` (default MLP_DIR), pass a different directory (e.g. MLP_DIR /
    f"epochs_{epochs}") to keep a run's cache separate from the default baseline instead of
    overwriting it. Returns (model, history, X_test, y_test), X_test/y_test are returned but
    not evaluated here; call evaluate_test explicitly once."""
    cache_key = {
        "n_bins": N_BINS, "bin_range": bin_range, "feature_stats": feature_stats,
        "standardize_extra": standardize_extra, "hidden_dim": hidden_dim,
        "n_hidden_layers": n_hidden_layers, "dropout": dropout, "weight_decay": weight_decay,
        "batch_norm": batch_norm, "epochs": epochs, "batch_size": batch_size, "lr": lr,
        "random_state": random_state,
    }
    history_cache = output_dir / HISTORY_FILENAME
    model_cache = output_dir / MODEL_FILENAME

    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = prepare_split_features(
        df, classes=classes, features=features, extra_features=extra_features, splits=splits,
        normalize=normalize, feature_stats=feature_stats, bin_range=bin_range,
        standardize_extra=standardize_extra,
    )

    if history_cache.exists() and model_cache.exists():
        cached = json.loads(history_cache.read_text())
        if cached.get("key") == cache_key:
            print(f"{history_cache} already matches this config, loading cached model + history")
            model = MLP(
                input_dim=X_train.shape[1], hidden_dim=hidden_dim, num_classes=len(classes),
                n_hidden_layers=n_hidden_layers, dropout=dropout, batch_norm=batch_norm,
            ).to(DEVICE)
            model.load_state_dict(torch.load(model_cache, map_location=DEVICE))
            plot_training_curves(cached["history"], output_dir=output_dir)
            return model, cached["history"], X_test, y_test
        print(f"{history_cache} doesn't match this config, retraining")

    model, history = train_mlp(
        X_train, y_train, X_val, y_val, classes=classes, epochs=epochs, batch_size=batch_size, lr=lr,
        random_state=random_state, hidden_dim=hidden_dim, n_hidden_layers=n_hidden_layers,
        dropout=dropout, weight_decay=weight_decay, batch_norm=batch_norm,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    history_cache.write_text(json.dumps({"key": cache_key, "history": history}, indent=2))
    torch.save(model.state_dict(), model_cache)
    print(f"Saved {history_cache} and {model_cache}")

    plot_training_curves(history, output_dir=output_dir)
    return model, history, X_test, y_test


if __name__ == "__main__":
    from build_points_table import build_and_save_points_table
    from taxonomy_separability import add_relative_features

    df = build_and_save_points_table()
    df = add_relative_features(df)
    df = apply_mlp_class_groups(df)

    run_training(df)  # uses MLP_CLASSES and the EPOCHS/BATCH_SIZE/LEARNING_RATE constants above
