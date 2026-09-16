"""DeepReflecs (Ulrich, Glaser & Timm 2021, arXiv:2010.09273) reimplementation, trained/
evaluated on the same fixed sequence-grouped split and class taxonomy as scripts/
mlp_classifier.py, so its test metrics are directly comparable to the histogram-encoded MLP
baseline (see compare_with_baseline).

Unlike the baseline, which pools each instance's points into a fixed-length histogram vector
before the network ever sees them (scripts/histogram_separability.py), DeepReflecs consumes an
instance's raw, unordered, variable-length reflection list directly: a shared per-point linear
layer, a global context layer (masked global max pool concatenated back onto every point, Fig.
3 of the paper), a second shared per-point linear layer, then a final masked global max pool
and a dense classification head (Fig. 4 of the paper). Order/size invariance comes from the
shared per-point layers plus the masked max pools, not from any padding convention, so batches
are padded dynamically to that batch's own longest instance (see pad_batch) rather than to a
fixed global buffer length like the paper's (padding is otherwise identical in effect, the mask
makes padded entries invisible to both max pools)."""
import json

import numpy as np
import pandas as pd
import torch
from torch import nn

from dataloader import RESULTS_DIR
from feature_distributions import MLP_CLASSES
from mlp_classifier import apply_mlp_class_groups
from separability_probe import class_weights
from sequence_split import load_split
from taxonomy_separability import INSTANCE_COLS

# --- hyperparameters ---
CONV_DIM = 16  # paper Fig. 4: first shared per-point layer's width
POINT_DIM = 32  # paper Fig. 4: second shared per-point layer's width (== 2*CONV_DIM, GCL output)
LEARNING_RATE = 4e-5  # matches mlp_classifier.py's baseline LR
EPOCHS = 100
BATCH_SIZE = 128
EVAL_BATCH_SIZE = 1024  # val/test inference batch, see _predict_in_batches
RANDOM_STATE = 0
# ------------------------

# paper section IV.D: x/y position in the tracked object's own frame, RCS, range, ego-motion
# compensated radial velocity. x_rel/y_rel here are centroid-relative (add_relative_features),
# not rotated by a heading estimate like the paper (RadarScenes' track labels carry no
# per-instance heading), otherwise the same feature set.
REFLECTION_FEATURES = ["x_rel", "y_rel", "rcs", "vr_compensated", "range_sc"]

DEEPREFLECS_DIR = RESULTS_DIR / "deepreflecs"
HISTORY_FILENAME = "deepreflecs_training_history.json"
MODEL_FILENAME = "deepreflecs_model.pt"
CURVES_FILENAME = "deepreflecs_training_curves.png"
VAL_METRICS_FILENAME = "deepreflecs_val_metrics.json"
CONFUSION_MATRIX_FILENAME = "deepreflecs_confusion_matrix.png"
METRICS_BAR_FILENAME = "deepreflecs_precision_recall_f1.png"
TEST_METRICS_FILENAME = "deepreflecs_test_metrics.json"
TEST_CONFUSION_MATRIX_FILENAME = "deepreflecs_test_confusion_matrix.png"
TEST_METRICS_BAR_FILENAME = "deepreflecs_test_precision_recall_f1.png"
COMPARISON_BAR_FILENAME = "comparison_f1_bar.png"
COMPARISON_SUMMARY_FILENAME = "comparison_summary.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class GlobalContextLayer(nn.Module):
    """Paper Fig. 3: max-pools the local per-point features (over the point axis, ignoring
    padded points via `mask`) to one global feature vector, then concatenates that global
    vector, repeated once per point, back onto every point's own local features. Doubles the
    feature width; no trainable parameters, same as the paper."""

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, M, F), mask: (B, M) bool, True at real (non-padding) points
        masked = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        global_feature = masked.max(dim=1, keepdim=True).values  # (B, 1, F)
        global_feature = global_feature.expand(-1, x.shape[1], -1)  # (B, M, F)
        return torch.cat([x, global_feature], dim=-1)  # (B, M, 2F)


class DeepReflecs(nn.Module):
    """Reimplementation of the network in Fig. 4 of Ulrich, Glaser & Timm 2021. A 1-D
    convolution with kernel size 1 (the paper's term for a linear layer applied independently
    to every point, preserving order/size invariance) is just an `nn.Linear` on the feature
    axis here, since `nn.Linear` already broadcasts over any number of leading dimensions -
    no need for an actual `nn.Conv1d`.

    input: (M, n_features) associated reflections of one object, M arbitrary and unordered.
    (M, n_features) -> conv1+ReLU -> (M, CONV_DIM) -> GlobalContextLayer -> (M, 2*CONV_DIM)
    -> conv2+ReLU -> (M, POINT_DIM) -> masked global max pool -> (POINT_DIM) -> dense -> (C)
    """

    def __init__(self, n_features: int, num_classes: int, conv_dim: int = CONV_DIM, point_dim: int = POINT_DIM):
        super().__init__()
        self.point_conv1 = nn.Linear(n_features, conv_dim)
        self.context = GlobalContextLayer()
        self.point_conv2 = nn.Linear(conv_dim * 2, point_dim)
        self.classifier = nn.Linear(point_dim, num_classes)

    def forward(self, points: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.point_conv1(points))
        x = self.context(x, mask)
        x = torch.relu(self.point_conv2(x))
        x = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        x = x.max(dim=1).values
        return self.classifier(x)


def build_point_sets(
    df: pd.DataFrame, classes: list[str], features: list[str] = REFLECTION_FEATURES
) -> tuple[list[np.ndarray], np.ndarray]:
    """Groups df (must already have `group` and x_rel/y_rel/range_sc, i.e. taxonomy_
    separability.add_relative_features + apply_mlp_class_groups already applied) by instance
    into one variable-length (M_i, len(features)) float32 array per instance, plus one integer
    class index per instance (matched order). This is DeepReflecs' native input representation:
    the raw per-instance reflection list, unlike prepare_split_features's fixed-length
    histogram vector.

    Column selection (`df[features]`/`df["group"]`) is done exactly once, vectorized, up front,
    not once per instance inside the groupby loop: with this project's pandas build, repeating
    a label-based column lookup ~500k times (once per instance) re-resolves the DataFrame's
    (pyarrow-string-backed) columns Index on every single call, turning what should be a fast
    loop into a many-minutes-long one (found via py-spy while evaluate_val_metrics hung). The
    per-instance loop below only does plain numpy fancy indexing on an already-extracted array,
    no pandas column resolution left inside it."""
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    mask = df["group"].isin(classes)
    filtered = df.loc[mask]
    feat_matrix = filtered[features].to_numpy(dtype="float32")
    class_idx = filtered["group"].map(class_to_idx).to_numpy()

    point_sets, labels = [], []
    for positions in filtered.groupby(INSTANCE_COLS, sort=False).indices.values():
        point_sets.append(feat_matrix[positions])
        labels.append(class_idx[positions[0]])
    return point_sets, np.array(labels, dtype="int64")


def _predict_in_batches(
    model: "DeepReflecs", X: torch.Tensor, mask: torch.Tensor, batch_size: int = EVAL_BATCH_SIZE
) -> torch.Tensor:
    """Runs model(X, mask) in chunks along the instance dimension instead of one
    unbatched forward pass. DeepReflecs' per-point layers and per-instance masked max
    pool never mix information across instances, so chunking changes nothing
    numerically, only memory footprint. A full-batch forward pass is fine at single-
    scan point counts (m_max ~45) but runs out of GPU memory once window pooling
    (track-accumulation branch) pushes m_max into the hundreds."""
    model.eval()
    logits = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            logits.append(model(X[start : start + batch_size], mask[start : start + batch_size]))
    return torch.cat(logits, dim=0)


def pad_to_fixed(point_sets: list[np.ndarray], m_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Pads every instance's (M_i, N) point array to one shared (m_max, N) buffer (the paper's
    "padded to a list with fixed size, e.g. length 64" approach, Sec. IV.C), instead of padding
    per-batch inside the training loop: with m_max fixed once up front, a whole split becomes a
    single dense (n_instances, m_max, N) array that can be sliced like the baseline MLP's
    X_train_t, no per-step Python padding overhead. Padding value doesn't matter (0 here) since
    the returned mask makes padded entries invisible to both masked max pools in the model."""
    n_features = point_sets[0].shape[1]
    points = np.zeros((len(point_sets), m_max, n_features), dtype="float32")
    mask = np.zeros((len(point_sets), m_max), dtype=bool)
    for i, p in enumerate(point_sets):
        points[i, : len(p)] = p
        mask[i, : len(p)] = True
    return points, mask


def prepare_split_point_sets(
    df: pd.DataFrame,
    classes: list[str] = MLP_CLASSES,
    features: list[str] = REFLECTION_FEATURES,
    splits: dict[str, list[str]] | None = None,
    standardize: bool = True,
):
    """Splits df by the fixed (or given) sequence split, builds each split's point sets via
    build_point_sets, then pads all three splits to one shared m_max (the largest instance
    point count anywhere in df, computed before splitting so val/test can never exceed the
    buffer fit on train+val+test's own sizes - this is a fixed array shape, not a statistic
    that could leak label information, unlike the standardization below). `standardize`
    (default True, matching the paper's "network inputs are normalized to zero-mean and unit-
    variance"): per-feature mean/std fit on real (non-padding) train points only, applied
    unchanged to val/test - same fit-on-train-only discipline as prepare_split_features's bin
    edges.

    Returns (X_train, mask_train, y_train, X_val, mask_val, y_val, X_test, mask_test, y_test,
    features): X_* are (n_instances, m_max, len(features)) float32 arrays, mask_* are
    (n_instances, m_max) bool arrays, True at real (non-padding) points."""
    if splits is None:
        splits = load_split()

    train_df = df.loc[df["sequence_name"].isin(splits["train"])]
    val_df = df.loc[df["sequence_name"].isin(splits["val"])]
    test_df = df.loc[df["sequence_name"].isin(splits["test"])]

    train_sets, y_train = build_point_sets(train_df, classes, features)
    val_sets, y_val = build_point_sets(val_df, classes, features)
    test_sets, y_test = build_point_sets(test_df, classes, features)

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

    return X_train, mask_train, y_train, X_val, mask_val, y_val, X_test, mask_test, y_test, features


def train_deepreflecs(
    X_train: np.ndarray,
    mask_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    mask_val: np.ndarray,
    y_val: np.ndarray,
    classes: list[str] = MLP_CLASSES,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    random_state: int = RANDOM_STATE,
    conv_dim: int = CONV_DIM,
    point_dim: int = POINT_DIM,
):
    """Trains DeepReflecs with Adam, class-count-weighted cross-entropy (same weighting scheme
    as train_mlp, for a like-for-like comparison). X_train/mask_train are already padded to one
    shared m_max (prepare_split_point_sets), so a batch is just an index slice, same mechanics
    as train_mlp's tensor slicing, no per-step padding work."""
    torch.manual_seed(random_state)

    weights_by_class = class_weights(pd.Series([classes[i] for i in y_train]))
    weight_tensor = torch.tensor(
        [weights_by_class[cls] for cls in classes], dtype=torch.float32, device=DEVICE
    )
    print(f"class weights: {dict(zip(classes, weight_tensor.tolist()))}")

    n_features = X_train.shape[2]
    model = DeepReflecs(n_features, num_classes=len(classes), conv_dim=conv_dim, point_dim=point_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    X_train_t = torch.tensor(X_train, device=DEVICE)
    mask_train_t = torch.tensor(mask_train, device=DEVICE)
    y_train_t = torch.tensor(y_train, device=DEVICE)
    X_val_t = torch.tensor(X_val, device=DEVICE)
    mask_val_t = torch.tensor(mask_val, device=DEVICE)
    y_val_t = torch.tensor(y_val, device=DEVICE)

    n = len(X_train_t)
    history = []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        epoch_loss, epoch_correct = 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            batch_x, batch_mask, batch_y = X_train_t[idx], mask_train_t[idx], y_train_t[idx]

            optimizer.zero_grad()
            logits = model(batch_x, batch_mask)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(idx)
            epoch_correct += (logits.argmax(dim=1) == batch_y).sum().item()

        train_loss = epoch_loss / n
        train_acc = epoch_correct / n

        val_logits = _predict_in_batches(model, X_val_t, mask_val_t)
        val_acc = (val_logits.argmax(dim=1) == y_val_t).float().mean().item()

        history.append({"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc, "val_acc": val_acc})
        print(f"epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

    return model, history


def plot_training_curves(history: list[dict], output_dir=DEEPREFLECS_DIR):
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
    ax_acc.set_title("DeepReflecs training curves (train vs val accuracy, train cost)")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / CURVES_FILENAME
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    return fig


def run_training(
    df: pd.DataFrame,
    classes: list[str] = MLP_CLASSES,
    features: list[str] = REFLECTION_FEATURES,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    random_state: int = RANDOM_STATE,
    output_dir=DEEPREFLECS_DIR,
    splits: dict[str, list[str]] | None = None,
    standardize: bool = True,
    conv_dim: int = CONV_DIM,
    point_dim: int = POINT_DIM,
):
    """Builds train/val/test point sets from the fixed split, trains (or loads from cache if
    this exact config was already run), plots train-vs-val curves. Mirrors mlp_classifier.
    run_training's caching/return convention: returns (model, history, X_test, mask_test,
    y_test), the test tensors are returned but not evaluated here, call evaluate_test_metrics
    once.

    `splits` is resolved to its actual dict (load_split() if None) before building the cache
    key: two calls with different class taxonomies or different candidate splits but otherwise
    identical hyperparameters must not collide on the same cache_key, otherwise a call with a
    changed `classes`/`splits` would silently load and return a model trained on the previous
    (different) taxonomy/split instead of retraining."""
    if splits is None:
        splits = load_split()

    cache_key = {
        "classes": classes, "splits": splits, "features": features, "standardize": standardize,
        "conv_dim": conv_dim, "point_dim": point_dim, "epochs": epochs, "batch_size": batch_size,
        "lr": lr, "random_state": random_state,
    }
    history_cache = output_dir / HISTORY_FILENAME
    model_cache = output_dir / MODEL_FILENAME

    X_train, mask_train, y_train, X_val, mask_val, y_val, X_test, mask_test, y_test, _ = prepare_split_point_sets(
        df, classes=classes, features=features, splits=splits, standardize=standardize,
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


def _evaluate_metrics(
    model: DeepReflecs,
    X: np.ndarray,
    mask: np.ndarray,
    y_true: np.ndarray,
    classes: list[str],
    metrics_cache,
    model_cache,
    confusion_matrix_path,
    metrics_bar_path,
    split_name: str,
):
    """Shared body of evaluate_val_metrics/evaluate_test_metrics: per-class precision/recall/f1
    + row-normalized confusion matrix + a precision/recall/f1 bar chart, from an already-loaded
    model. Numeric metrics are cached, reused only if newer than model_cache (a retrained model
    in the same output_dir would otherwise silently serve stale metrics); plots are always
    regenerated fresh, cheap regardless."""
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_recall_fscore_support

    X_t = torch.tensor(X, device=DEVICE)
    mask_t = torch.tensor(mask, device=DEVICE)
    y_pred = _predict_in_batches(model, X_t, mask_t).argmax(dim=1).cpu().numpy()

    if metrics_cache.exists() and metrics_cache.stat().st_mtime >= model_cache.stat().st_mtime:
        metrics_df = pd.read_json(metrics_cache, orient="index")
        print(f"{metrics_cache} already cached, reusing")
    else:
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=range(len(classes)), zero_division=0
        )
        metrics_df = pd.DataFrame(
            {"precision": precision, "recall": recall, "f1": f1, "support": support}, index=classes
        )
        metrics_cache.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_json(metrics_cache, orient="index", indent=2)
        print(f"Saved {metrics_cache}")
    print(f"per-class precision/recall/f1 ({split_name}):")
    print(metrics_df.round(3).to_string())

    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)), normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, colorbar=False, values_format=".2f")
    ax.set_title(f"DeepReflecs: {split_name} confusion matrix (row-normalized)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    confusion_matrix_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(confusion_matrix_path, dpi=150)
    print(f"Saved {confusion_matrix_path}")

    bar_fig, bar_ax = plt.subplots(figsize=(9, 5))
    metrics_df[["precision", "recall", "f1"]].plot(kind="bar", ax=bar_ax, edgecolor="k")
    bar_ax.set_ylim(0, 1)
    bar_ax.set_ylabel("score")
    bar_ax.set_title(f"DeepReflecs: per-class precision/recall/f1 ({split_name})")
    bar_ax.legend(loc="lower right")
    plt.setp(bar_ax.get_xticklabels(), rotation=45, ha="right")
    bar_fig.tight_layout()
    bar_fig.savefig(metrics_bar_path, dpi=150)
    print(f"Saved {metrics_bar_path}")

    return metrics_df, fig, bar_fig


def evaluate_val_metrics(
    df: pd.DataFrame,
    classes: list[str] = MLP_CLASSES,
    features: list[str] = REFLECTION_FEATURES,
    output_dir=DEEPREFLECS_DIR,
    splits: dict[str, list[str]] | None = None,
    standardize: bool = True,
    conv_dim: int = CONV_DIM,
    point_dim: int = POINT_DIM,
):
    """Per-class precision/recall/f1 + confusion matrix on val, from the cached trained model in
    `output_dir`. Never retrains, only loads (same rule as mlp_classifier.evaluate_val_metrics:
    val is for freely comparing candidates, test is checked once at the end)."""
    model_cache = output_dir / MODEL_FILENAME
    if not model_cache.exists():
        raise FileNotFoundError(f"{model_cache} doesn't exist, run run_training() first")

    _, _, _, X_val, mask_val, y_val, _, _, _, _ = prepare_split_point_sets(
        df, classes=classes, features=features, splits=splits, standardize=standardize,
    )
    model = DeepReflecs(len(features), num_classes=len(classes), conv_dim=conv_dim, point_dim=point_dim).to(DEVICE)
    model.load_state_dict(torch.load(model_cache, map_location=DEVICE))

    return _evaluate_metrics(
        model, X_val, mask_val, y_val, classes,
        metrics_cache=output_dir / VAL_METRICS_FILENAME, model_cache=model_cache,
        confusion_matrix_path=output_dir / CONFUSION_MATRIX_FILENAME,
        metrics_bar_path=output_dir / METRICS_BAR_FILENAME, split_name="val",
    )


def evaluate_test_metrics(
    df: pd.DataFrame,
    classes: list[str] = MLP_CLASSES,
    features: list[str] = REFLECTION_FEATURES,
    output_dir=DEEPREFLECS_DIR,
    splits: dict[str, list[str]] | None = None,
    standardize: bool = True,
    conv_dim: int = CONV_DIM,
    point_dim: int = POINT_DIM,
):
    """The one and only time test should be touched: per-class precision/recall/f1 + confusion
    matrix on test, from the cached trained model in `output_dir`. Call once, after training/
    tuning is fully finished."""
    model_cache = output_dir / MODEL_FILENAME
    if not model_cache.exists():
        raise FileNotFoundError(f"{model_cache} doesn't exist, run run_training() first")

    _, _, _, _, _, _, X_test, mask_test, y_test, _ = prepare_split_point_sets(
        df, classes=classes, features=features, splits=splits, standardize=standardize,
    )
    model = DeepReflecs(len(features), num_classes=len(classes), conv_dim=conv_dim, point_dim=point_dim).to(DEVICE)
    model.load_state_dict(torch.load(model_cache, map_location=DEVICE))

    return _evaluate_metrics(
        model, X_test, mask_test, y_test, classes,
        metrics_cache=output_dir / TEST_METRICS_FILENAME, model_cache=model_cache,
        confusion_matrix_path=output_dir / TEST_CONFUSION_MATRIX_FILENAME,
        metrics_bar_path=output_dir / TEST_METRICS_BAR_FILENAME, split_name="test",
    )


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compare_with_baseline(
    classes: list[str] = MLP_CLASSES,
    baseline_test_metrics_path=None,
    deepreflecs_test_metrics_path=None,
    output_dir=DEEPREFLECS_DIR,
):
    """Reads both models' already-computed test metrics (never retrains/re-evaluates, that's
    evaluate_test_metrics' job for each model) and produces a side-by-side comparison: total
    accuracy (support-weighted from per-class recall*support) and macro F1, plus a grouped bar
    chart of per-class F1. Requires both scripts/mlp_classifier.py's evaluate_test_metrics and
    this module's evaluate_test_metrics to have already run at least once."""
    import matplotlib.pyplot as plt

    from mlp_classifier import MLP_DIR
    from mlp_classifier import MODEL_FILENAME as MLP_MODEL_FILENAME
    from mlp_classifier import TEST_METRICS_FILENAME as MLP_TEST_METRICS_FILENAME

    if baseline_test_metrics_path is None:
        baseline_test_metrics_path = MLP_DIR / MLP_TEST_METRICS_FILENAME
    if deepreflecs_test_metrics_path is None:
        deepreflecs_test_metrics_path = output_dir / TEST_METRICS_FILENAME
    baseline_model_path = MLP_DIR / MLP_MODEL_FILENAME
    deepreflecs_model_path = output_dir / MODEL_FILENAME

    for path, name in [(baseline_test_metrics_path, "baseline"), (deepreflecs_test_metrics_path, "DeepReflecs")]:
        if not path.exists():
            raise FileNotFoundError(f"{path} doesn't exist, run {name}'s evaluate_test_metrics() first")

    baseline_df = pd.read_json(baseline_test_metrics_path, orient="index").loc[classes]
    deepreflecs_df = pd.read_json(deepreflecs_test_metrics_path, orient="index").loc[classes]

    def summarize(metrics_df):
        support = metrics_df["support"]
        accuracy = (metrics_df["recall"] * support).sum() / support.sum()
        return {"accuracy": accuracy, "macro_f1": metrics_df["f1"].mean()}

    def n_params(model_path):
        # architecture-agnostic: sums tensor sizes straight out of the saved state_dict, so this
        # works for either model without importing/reconstructing its class.
        if not model_path.exists():
            return None
        state_dict = torch.load(model_path, map_location="cpu")
        return sum(t.numel() for t in state_dict.values())

    summary = {
        "baseline_mlp": {**summarize(baseline_df), "n_params": n_params(baseline_model_path)},
        "deepreflecs": {**summarize(deepreflecs_df), "n_params": n_params(deepreflecs_model_path)},
    }
    print("test set comparison:")
    print(pd.DataFrame(summary).round(4).to_string())

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / COMPARISON_SUMMARY_FILENAME
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved {summary_path}")

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(classes))
    width = 0.35
    ax.bar(x - width / 2, baseline_df["f1"], width, label="baseline MLP (histogram)", edgecolor="k")
    ax.bar(x + width / 2, deepreflecs_df["f1"], width, label="DeepReflecs (point set)", edgecolor="k")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 (test)")
    ax.set_title("Baseline MLP vs DeepReflecs: per-class test F1")
    ax.legend()
    fig.tight_layout()

    bar_path = output_dir / COMPARISON_BAR_FILENAME
    fig.savefig(bar_path, dpi=150)
    print(f"Saved {bar_path}")

    return summary, fig


if __name__ == "__main__":
    from build_points_table import build_and_save_points_table
    from taxonomy_separability import add_relative_features

    df = build_and_save_points_table()
    df = add_relative_features(df)
    df = apply_mlp_class_groups(df)

    model, history, X_test, mask_test, y_test = run_training(df)
    print(f"DeepReflecs parameter count: {count_params(model)}")
    evaluate_val_metrics(df)
    evaluate_test_metrics(df)
    compare_with_baseline()
