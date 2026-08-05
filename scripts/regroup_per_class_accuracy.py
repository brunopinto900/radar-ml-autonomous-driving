"""Where did the regrouped model's accuracy gain actually come from, epoch by epoch?
MLP_FINDINGS.md section 10's converged (1000-epoch) result showed car/pedestrian_group
improving, not the targeted large_vehicle - this tracks per-class accuracy at every epoch
(not just the final checkpoint) to see which classes' accuracy migrated from/to as training
progressed, same method as scripts/min4pts_per_class_accuracy.py used for section 6's
peaks-then-declines investigation.

Deterministic re-run of scripts/train_mlp_regroup.py (same seed/data/hyperparameters,
1000 epochs) - per-epoch per-class accuracy isn't recoverable from the final checkpoint
alone, so this retrains rather than reusing the saved model.
"""
import numpy as np
import pandas as pd
import torch
from torch import nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dataloader import RESULTS_DIR  # noqa: E402
from train_mlp import CLASSES, class_weights, load_or_build_dataset  # noqa: E402
from train_mlp_full import BATCH_SIZE, HIDDEN_DIM, LR, PaperMLP, SEED  # noqa: E402
from train_mlp_regroup import regrouped_labels  # noqa: E402

EPOCHS = 1000
OUT_DIR = RESULTS_DIR / "mlp_full_run" / "car_large_vehicle_regroup_1000epoch"

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

    train_df = pd.read_parquet(RESULTS_DIR / "train_points.parquet")
    val_df = pd.read_parquet(RESULTS_DIR / "val_points.parquet")
    X_train, _, keys_train = load_or_build_dataset(train_df, {}, CLASSES, "train")
    X_val, _, keys_val = load_or_build_dataset(val_df, {}, CLASSES, "val")
    y_train = regrouped_labels(train_df, keys_train, CLASSES)
    y_val = regrouped_labels(val_df, keys_val, CLASSES)

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
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"epoch {epoch + 1:4d}/{EPOCHS}  train_loss {train_loss_hist[-1]:.4f}  "
                  f"val_loss {val_loss:.4f}  "
                  + "  ".join(f"{n}={val_acc_by_class[n][-1]:.3f}" for n in CLASSES))

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
    fig.suptitle("Regrouped car/large_vehicle: per-class accuracy vs. epoch - "
                 "which class(es) did the gain migrate from/to?")
    fig.tight_layout()

    path = OUT_DIR / "per_class_accuracy.png"
    fig.savefig(path, dpi=150)
    print(f"\nSaved {path}")

    np.save(OUT_DIR / "per_class_accuracy_history.npy",
            {"train": train_acc_by_class, "val": val_acc_by_class}, allow_pickle=True)
    print(f"Saved {OUT_DIR / 'per_class_accuracy_history.npy'}")
