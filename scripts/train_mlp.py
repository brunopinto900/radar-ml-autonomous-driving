"""Day 6 pipeline smoke test (see MLP_DESIGN.md section 0): validate the
training workflow on a small subset before the real run. One hidden layer, a
random subset of instances (not a positional slice), reusing the real
feature-extraction code from histogram_features.py rather than a simplified
stand-in. Edit N_TRAIN_INSTANCES to switch between "overfit a tiny slice"
(e.g. 200) and "reduced run" (e.g. ~90000, roughly 25% of the train split).
Accuracy numbers from this script mean nothing about the real approach - they
only mean the code runs.
"""
import numpy as np
import pandas as pd
import torch
from torch import nn

import matplotlib
matplotlib.use("Agg")  # headless: this script only ever saves plots, never shows them
import matplotlib.pyplot as plt  # noqa: E402

from dataloader import RESULTS_DIR, plot_predictions_grid  # noqa: E402
from histogram_features import FEATURES, N_BINS, add_relative_xy, fit_bin_edges, build_dataset  # noqa: E402

CLASSES = ["car", "large_vehicle", "pedestrian", "pedestrian_group", "two_wheeler"]
PLOTS_DIR = RESULTS_DIR / "mlp_smoke_test"
CACHE_DIR = RESULTS_DIR / "mlp_feature_cache"

# Smoke-test knobs (MLP_DESIGN.md section 0).
N_TRAIN_INSTANCES = 200
N_VAL_INSTANCES = 50
SEED = 42
HIDDEN_DIM = 16
LR = 1e-3  # NOTE: paper's 1e-5 is tuned for their 1000-epoch/3-layer run - too slow to see
           # a signal in a short smoke test, deliberately bumped up here.
EPOCHS = 200
# Bumped from 32: measured 5,536 batches/epoch at batch_size=64 on the full
# 354k-instance train set (~4.6s/epoch, mostly Python-loop/optimizer-step
# overhead, not compute, for a ~1,400-param model) - fewer, bigger batches
# should cut wall-clock time substantially at that scale with negligible
# effect on results. Only matters once N_TRAIN_INSTANCES is scaled up - at
# 200 instances there's only ever ~1 batch/epoch either way.
BATCH_SIZE = 256


class OneHiddenLayerMLP(nn.Module):
    """Smoke-test model only (MLP_DESIGN.md section 0) - the real run uses the
    paper's 3-layer architecture, built separately once this is validated."""

    def __init__(self, in_dim: int, hidden_dim: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_or_build_dataset(df: pd.DataFrame, bin_edges: dict, classes: list[str],
                           cache_name: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Loads cached feature vectors (+ instance keys) if present, otherwise
    builds and caches them. Building the full dataset takes ~3.5 min measured
    (169s train + 32s val) - caching turns that into a one-time cost instead of
    paying it on every run. Cache is keyed only by name (train/val) - delete
    results/mlp_feature_cache/ if FEATURES/N_BINS in histogram_features.py ever
    change, nothing here detects that automatically."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    x_path = CACHE_DIR / f"{cache_name}_X.npy"
    y_path = CACHE_DIR / f"{cache_name}_y.npy"
    keys_path = CACHE_DIR / f"{cache_name}_keys.parquet"
    if x_path.exists() and y_path.exists() and keys_path.exists():
        print(f"Loading cached {cache_name} features from {x_path}")
        return np.load(x_path), np.load(y_path), pd.read_parquet(keys_path)
    print(f"Building {cache_name} features (no cache found - takes a few minutes for the full split)")
    X, y, keys = build_dataset(df, bin_edges, classes)
    np.save(x_path, X)
    np.save(y_path, y)
    keys.to_parquet(keys_path)
    print(f"Cached {cache_name} features to {x_path}")
    return X, y, keys


def subsample(X: np.ndarray, y: np.ndarray, keys: pd.DataFrame, n: int,
              rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Random subsample of n instances from already-built feature vectors (not
    a positional slice - see MLP_DESIGN.md section 0). Keeps keys row-aligned
    with X/y so predictions can still be traced back to raw points afterward."""
    n = min(n, len(y))
    idx = rng.choice(len(y), size=n, replace=False)
    return X[idx], y[idx], keys.iloc[idx].reset_index(drop=True)


def class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    """w_i = N_samples / (N_classes * N_i), the paper's formula (MLP_DESIGN.md)."""
    counts = np.maximum(np.bincount(y, minlength=n_classes), 1)  # avoid /0 if a
    # class is absent from a small subset - expected at this scale, not a bug.
    weights = len(y) / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    train_df = add_relative_xy(pd.read_parquet(RESULTS_DIR / "train_points.parquet"))
    val_df = add_relative_xy(pd.read_parquet(RESULTS_DIR / "val_points.parquet"))

    # Bin edges fit on the FULL train split, not the subsample below - this
    # smoke test exercises the same edges the real run would use.
    bin_edges = fit_bin_edges(train_df)

    # Build (or load cached) feature vectors for the FULL splits first, then
    # subsample - so the one-time ~3.5 min pipeline cost is paid once total,
    # not once per smoke-test run, and switching N_TRAIN_INSTANCES later
    # (e.g. toward the full ~25% reduced run) never re-triggers it either.
    X_train_full, y_train_full, keys_train_full = load_or_build_dataset(train_df, bin_edges, CLASSES, "train")
    X_val_full, y_val_full, keys_val_full = load_or_build_dataset(val_df, bin_edges, CLASSES, "val")

    X_train, y_train, keys_train = subsample(X_train_full, y_train_full, keys_train_full, N_TRAIN_INSTANCES, rng)
    X_val, y_val, keys_val = subsample(X_val_full, y_val_full, keys_val_full, N_VAL_INSTANCES, rng)
    print(f"train: {len(y_train)} instances, val: {len(y_val)} instances")
    print(f"train class counts: {dict(zip(CLASSES, np.bincount(y_train, minlength=len(CLASSES))))}")

    X_train_t, y_train_t = torch.tensor(X_train), torch.tensor(y_train)
    X_val_t, y_val_t = torch.tensor(X_val), torch.tensor(y_val)

    model = OneHiddenLayerMLP(in_dim=len(FEATURES) * N_BINS, hidden_dim=HIDDEN_DIM,
                              n_classes=len(CLASSES))
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y_train, len(CLASSES)))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    n_train = len(y_train)
    history = {"loss": [], "train_acc": [], "val_acc": []}
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n_train)
        total_loss = 0.0
        for start in range(0, n_train, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            xb, yb = X_train_t[idx], y_train_t[idx]
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)

        # Tracked every epoch (cheap at this scale) for smooth curves, not just
        # the printed cadence below.
        model.eval()
        with torch.no_grad():
            train_acc = (model(X_train_t).argmax(1) == y_train_t).float().mean().item()
            val_acc = (model(X_val_t).argmax(1) == y_val_t).float().mean().item()
        history["loss"].append(total_loss / n_train)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"epoch {epoch + 1:4d}  loss {history['loss'][-1]:.4f}  "
                  f"train_acc {train_acc:.3f}  val_acc {val_acc:.3f}")

    # Final confusion matrix on the (tiny) val slice - exercises the eval code
    # path too, not just training (MLP_DESIGN.md section 0).
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_t).argmax(1).numpy()
    true_names = pd.Series(y_val).map(lambda i: CLASSES[i])
    pred_names = pd.Series(val_preds).map(lambda i: CLASSES[i])
    cm = pd.crosstab(true_names, pred_names, rownames=["true"], colnames=["pred"]).reindex(
        index=CLASSES, columns=CLASSES, fill_value=0)
    print("\nValidation confusion matrix:")
    print(cm)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # x/y bounding-box grid: true vs. predicted label per instance, green/red
    # by correctness (dataloader.py's plot_predictions_grid) - val_df is the
    # full val split, only used here to look up each sampled instance's own
    # raw points via keys_val.
    plot_predictions_grid(val_df, keys_val, y_val, val_preds, CLASSES,
                           PLOTS_DIR / "predictions_grid.png", n=9, rng=rng)

    epochs_axis = np.arange(1, EPOCHS + 1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs_axis, history["loss"], color="tab:red")
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss (class-weighted cross-entropy)")
    ax.set_title(f"Smoke test: cost vs. epoch (n_train={n_train})")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "cost.png", dpi=150)
    print(f"\nSaved {PLOTS_DIR / 'cost.png'}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs_axis, history["train_acc"], label="train", color="tab:blue")
    ax.plot(epochs_axis, history["val_acc"], label="val", color="tab:orange")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Smoke test: accuracy vs. epoch (n_train={n_train}, n_val={len(y_val)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "accuracy.png", dpi=150)
    print(f"Saved {PLOTS_DIR / 'accuracy.png'}")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm.to_numpy(), cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Smoke test: validation confusion matrix (n_val={len(y_val)})")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            count = cm.to_numpy()[i, j]
            color = "white" if count > cm.to_numpy().max() / 2 else "black"
            ax.text(j, i, str(count), ha="center", va="center", color=color)
    fig.colorbar(im, ax=ax, label="count")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "validation.png", dpi=150)
    print(f"Saved {PLOTS_DIR / 'validation.png'}")
