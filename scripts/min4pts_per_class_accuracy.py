"""Ad-hoc diagnostic: reproduces the min_train_points_ablation training run
exactly (same seed/data/hyperparameters as train_mlp_full.py with
MIN_TRAIN_POINTS=4), but tracks per-class accuracy at every epoch instead of
just the aggregate - to explain the train/val accuracy-peaks-around-epoch-8-9-
then-declines pattern noted in MLP_FINDINGS.md section 6 but never dug into.
Hypothesis under test (Bruno): a lower-representation class starts getting
misclassified as training continues past the peak, dragging aggregate accuracy
down, while loss keeps falling because the higher-representation classes keep
improving and dominate the (weighted) loss sum. Retrains because per-epoch
per-class accuracy isn't recoverable from the final checkpoint alone -
deterministic same seed, ~0.1 min, not the full feature-rebuild path (reuses
the cached train_min4pts/val features).
"""
import numpy as np
import torch
from torch import nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dataloader import RESULTS_DIR  # noqa: E402
from train_mlp import CLASSES, class_weights  # noqa: E402
from train_mlp_full import PaperMLP  # noqa: E402

CACHE_DIR = RESULTS_DIR / "mlp_feature_cache"
OUT_DIR = RESULTS_DIR / "mlp_full_run" / "min_train_points_ablation"

SEED = 42
HIDDEN_DIM = 16
LR = 1e-5
BATCH_SIZE = 256
EPOCHS = 20

CLASS_COLORS = {
    "car": "tab:blue",
    "large_vehicle": "tab:orange",
    "pedestrian": "tab:green",
    "pedestrian_group": "tab:red",
    "two_wheeler": "tab:purple",
}


def per_class_accuracy(logits: torch.Tensor, y: torch.Tensor) -> dict:
    preds = logits.argmax(1)
    out = {}
    for i, name in enumerate(CLASSES):
        mask = y == i
        out[name] = (preds[mask] == y[mask]).float().mean().item() if mask.any() else float("nan")
    return out


if __name__ == "__main__":
    torch.manual_seed(SEED)

    X_train = np.load(CACHE_DIR / "train_min4pts_X.npy")
    y_train = np.load(CACHE_DIR / "train_min4pts_y.npy")
    X_val = np.load(CACHE_DIR / "val_X.npy")
    y_val = np.load(CACHE_DIR / "val_y.npy")

    X_train_t, y_train_t = torch.tensor(X_train), torch.tensor(y_train)
    X_val_t, y_val_t = torch.tensor(X_val), torch.tensor(y_val)

    model = PaperMLP(in_dim=X_train.shape[1], hidden_dim=HIDDEN_DIM, n_classes=len(CLASSES))
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y_train, len(CLASSES)))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    n_train = len(y_train)
    train_acc_by_class = {name: [] for name in CLASSES}
    val_acc_by_class = {name: [] for name in CLASSES}
    train_loss_hist, val_loss_hist = [], []

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

        model.eval()
        with torch.no_grad():
            train_logits = model(X_train_t)
            val_logits = model(X_val_t)
            val_loss = loss_fn(val_logits, y_val_t).item()

        for name, acc in per_class_accuracy(train_logits, y_train_t).items():
            train_acc_by_class[name].append(acc)
        for name, acc in per_class_accuracy(val_logits, y_val_t).items():
            val_acc_by_class[name].append(acc)
        train_loss_hist.append(total_loss / n_train)
        val_loss_hist.append(val_loss)
        print(f"epoch {epoch + 1:4d}/{EPOCHS}  train_loss {train_loss_hist[-1]:.4f}  "
              f"val_loss {val_loss:.4f}  "
              + "  ".join(f"{n}={train_acc_by_class[n][-1]:.3f}" for n in CLASSES))

    epochs_axis = np.arange(1, EPOCHS + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for name in CLASSES:
        axes[0].plot(epochs_axis, train_acc_by_class[name], label=name, color=CLASS_COLORS[name])
        axes[1].plot(epochs_axis, val_acc_by_class[name], label=name, color=CLASS_COLORS[name])
    axes[0].set_title("train accuracy per class")
    axes[1].set_title("val accuracy per class (full val set)")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("accuracy")
    axes[0].legend(fontsize=8)
    fig.suptitle("min_train_points_ablation: per-class accuracy vs. epoch "
                 "(aggregate peaks ~epoch 8-9 then declines - which class(es) drive it?)")
    fig.tight_layout()

    path = OUT_DIR / "per_class_accuracy.png"
    fig.savefig(path, dpi=150)
    print(f"\nSaved {path}")
