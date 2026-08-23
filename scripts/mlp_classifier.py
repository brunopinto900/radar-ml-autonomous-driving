"""Day 6: baseline MLP on the histogram-encoded feature vector (Design_Decisions.md decision 6),
trained/evaluated on the fixed sequence-grouped split (decision 5). Bin edges are fit on train
only and applied unchanged to val/test (see histogram_separability.fit_bin_edges) - unlike the
separability probes, which recomputed edges from whatever pool was passed in each time (a fine
shortcut for a quick probe, wrong for the actual model: edges are a fitted preprocessing
parameter, val/test must never influence them).

Trains on train, tracks train/val loss+accuracy per epoch (NOT test - test is checked once, at
the end, via evaluate_test, never watched during training/tuning). Caches the training history
and model weights, keyed by the exact architecture/hyperparameter config, so a repeat call with
the same config skips straight to the cached result.
"""
import json

import numpy as np
import pandas as pd
import torch
from torch import nn

from dataloader import MLP_CLASS_GROUPS, RESULTS_DIR
from feature_distributions import INSTANCE_LEVEL_FEATURES, MLP_CLASSES, POINT_LEVEL_FEATURES
from histogram_separability import build_histogram_features, fit_bin_edges
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MLP(nn.Module):
    """3 Linear layers: the first two ("inner") have HIDDEN_DIM output neurons each (with ReLU),
    the third maps to one logit per final class."""

    def __init__(self, input_dim: int, hidden_dim: int = HIDDEN_DIM, num_classes: int = len(MLP_CLASSES)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

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
):
    """Fits bin edges on the train split only, then encodes train/val/test with those same
    edges. `classes` (default MLP_CLASSES) lets a caller encode a different class taxonomy
    against the same fixed, sequence level split (see mlp_variants.py). `features` (default
    POINT_LEVEL_FEATURES) are binned into `n_bins` columns each; `extra_features` (default
    doppler_spread) are appended unbinned - a caller can swap doppler_spread for another
    point-level feature by moving it into `features` and passing extra_features=[] instead (see
    mlp_variants.py's "range_sc" variant). `splits` (default None) uses the project's standing
    fixed split (load_split()) - pass an explicit {"train"/"val"/"test": [sequence_name, ...]}
    dict instead to evaluate a different candidate split (see sequence_split.select_best_split),
    without touching the standing one. Returns
    (X_train, y_train, X_val, y_val, X_test, y_test, feature_cols) as numpy arrays. X float32,
    y integer class indices in `classes` order."""
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    if splits is None:
        splits = load_split()

    train_df = df.loc[df["sequence_name"].isin(splits["train"])]
    val_df = df.loc[df["sequence_name"].isin(splits["val"])]
    test_df = df.loc[df["sequence_name"].isin(splits["test"])]

    edges = fit_bin_edges(train_df, n_bins, features)

    def to_xy(split_df):
        hist = build_histogram_features(
            split_df, n_bins, classes, features, edges=edges, extra_features=extra_features
        )
        feature_cols = [c for c in hist.columns if c not in ("sequence_name", "group")]
        X = hist[feature_cols].to_numpy(dtype="float32")
        y = hist["group"].map(class_to_idx).to_numpy(dtype="int64")
        return X, y, feature_cols

    X_train, y_train, feature_cols = to_xy(train_df)
    X_val, y_val, _ = to_xy(val_df)
    X_test, y_test, _ = to_xy(test_df)
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
):
    """Trains the MLP with Adam, class-count-weighted cross-entropy. `classes` (default
    MLP_CLASSES) is the class list y_train/y_val's integer labels index into."""
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

    model = MLP(input_dim=X_train.shape[1], hidden_dim=hidden_dim, num_classes=len(classes)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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


def evaluate_val_metrics(
    df: pd.DataFrame,
    classes: list[str] = MLP_CLASSES,
    output_dir=MLP_DIR,
    features: list[str] = POINT_LEVEL_FEATURES,
    extra_features: list[str] = INSTANCE_LEVEL_FEATURES,
    splits: dict[str, list[str]] | None = None,
    hidden_dim: int = HIDDEN_DIM,
):
    """Per-class precision/recall/f1 + confusion matrix on val, computed from the cached trained
    model in `output_dir` - never retrains, only loads. `classes` (default MLP_CLASSES) must
    match what the cached model at `output_dir` was trained with. Rebuilding val's feature
    vectors (bin edges fit on train, applied to val) and running one forward pass is cheap either
    way (not training), so it's redone each call; only the actual training step is skipped, via
    the model cache. Uses val, not test, for the same reason the training curves did - test stays
    untouched until a deliberate one-time check. Numeric metrics are cached to
    output_dir/mlp_val_metrics.json (skip-if-cached printing, but the confusion matrix and bar
    plots are still regenerated - cheap). Returns (metrics_df, confusion_matrix_fig, bar_fig)."""
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_recall_fscore_support

    model_cache = output_dir / MODEL_FILENAME
    val_metrics_cache = output_dir / VAL_METRICS_FILENAME
    confusion_matrix_path = output_dir / CONFUSION_MATRIX_FILENAME
    metrics_bar_path = output_dir / METRICS_BAR_FILENAME

    if not model_cache.exists():
        raise FileNotFoundError(f"{model_cache} doesn't exist - run run_training() first")

    _, _, X_val, y_val, _, _, _ = prepare_split_features(
        df, classes=classes, features=features, extra_features=extra_features, splits=splits
    )

    model = MLP(input_dim=X_val.shape[1], hidden_dim=hidden_dim, num_classes=len(classes)).to(DEVICE)
    model.load_state_dict(torch.load(model_cache, map_location=DEVICE))
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.tensor(X_val, device=DEVICE)).argmax(dim=1).cpu().numpy()

    if val_metrics_cache.exists():
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
    ax.set_title("MLP - val confusion matrix (row-normalized)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(confusion_matrix_path, dpi=150)
    print(f"Saved {confusion_matrix_path}")

    bar_fig, bar_ax = plt.subplots(figsize=(9, 5))
    metrics_df[["precision", "recall", "f1"]].plot(kind="bar", ax=bar_ax, edgecolor="k")
    bar_ax.set_ylim(0, 1)
    bar_ax.set_ylabel("score")
    bar_ax.set_title("MLP - per-class precision/recall/f1 (val)")
    bar_ax.legend(loc="lower right")
    plt.setp(bar_ax.get_xticklabels(), rotation=45, ha="right")
    bar_fig.tight_layout()

    bar_fig.savefig(metrics_bar_path, dpi=150)
    print(f"Saved {metrics_bar_path}")

    return metrics_df, fig, bar_fig


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
    hidden_dim: int = HIDDEN_DIM,
):
    """Builds train/val/test from the fixed split, trains (or loads from cache if this exact
    config was already run), plots train-vs-val curves. `classes` (default MLP_CLASSES) lets a
    caller train on a different class taxonomy against the same fixed split (see
    mlp_variants.py); `features`/`extra_features` (see prepare_split_features) let a caller swap
    the feature set instead, e.g. binning range_sc in place of doppler_spread; `splits` lets a
    caller evaluate a different candidate split instead of the standing fixed one (see
    sequence_split.select_best_split); `hidden_dim` lets a caller change model capacity. Writes to
    `output_dir` (default MLP_DIR) - pass a different directory (e.g. MLP_DIR / f"epochs_{epochs}")
    to keep a run's cache separate from the default baseline instead of overwriting it. Returns
    (model, history, X_test, y_test) - X_test/y_test are returned but NOT evaluated here; call
    evaluate_test explicitly once."""
    cache_key = {
        "n_bins": N_BINS, "hidden_dim": hidden_dim, "epochs": epochs, "batch_size": batch_size,
        "lr": lr, "random_state": random_state,
    }
    history_cache = output_dir / HISTORY_FILENAME
    model_cache = output_dir / MODEL_FILENAME

    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = prepare_split_features(
        df, classes=classes, features=features, extra_features=extra_features, splits=splits
    )

    if history_cache.exists() and model_cache.exists():
        cached = json.loads(history_cache.read_text())
        if cached.get("key") == cache_key:
            print(f"{history_cache} already matches this config, loading cached model + history")
            model = MLP(input_dim=X_train.shape[1], hidden_dim=hidden_dim, num_classes=len(classes)).to(DEVICE)
            model.load_state_dict(torch.load(model_cache, map_location=DEVICE))
            plot_training_curves(cached["history"], output_dir=output_dir)
            return model, cached["history"], X_test, y_test
        print(f"{history_cache} doesn't match this config, retraining")

    model, history = train_mlp(
        X_train, y_train, X_val, y_val, classes=classes, epochs=epochs, batch_size=batch_size, lr=lr,
        random_state=random_state, hidden_dim=hidden_dim,
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
