"""Combines two open threads from MLP_FINDINGS.md section 10 into one run: the revised
car/large_vehicle grouping (car=old car+truck+large_vehicle(raw), large_vehicle=bus-only,
scripts/train_mlp_regroup.py) AND the magnitude/shape confound hypothesis - histogram blocks
normalized to each bin's share of the instance's own points (density) instead of raw counts,
with n_points kept as an explicit 81st feature so magnitude isn't discarded, just decoupled
from shape (histogram_features.py's normalize_density flag).

Density normalization changes X's values (not just dimensionality), so this can't reuse the
raw-count baseline caches - builds fresh "_dens_pc"-suffixed caches. 50 epochs per explicit
request, not a plateau check - if this looks promising, re-run to convergence before drawing
conclusions the way section 10's regroup result had to be.
"""
import numpy as np
import pandas as pd
import torch
from torch import nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dataloader import RESULTS_DIR, plot_predictions_grid  # noqa: E402
from histogram_features import add_relative_xy, fit_bin_edges  # noqa: E402
from train_mlp import CLASSES, class_weights, load_or_build_dataset  # noqa: E402
from train_mlp_full import BATCH_SIZE, HIDDEN_DIM, LR, PaperMLP, SEED  # noqa: E402
from train_mlp_regroup import regrouped_labels  # noqa: E402

EPOCHS = 1000  # the 50-epoch run (density_norm_regroup_50epoch/) hadn't plateaued - cost was
               # still visibly decreasing. Same plateau check as section 2/section 10's regroup.
OUT_DIR = RESULTS_DIR / "mlp_full_run" / "density_norm_regroup_1000epoch"

if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    train_df_raw = pd.read_parquet(RESULTS_DIR / "train_points.parquet")
    val_df_raw = pd.read_parquet(RESULTS_DIR / "val_points.parquet")
    train_df = add_relative_xy(train_df_raw)
    val_df = add_relative_xy(val_df_raw)
    bin_edges = fit_bin_edges(train_df)  # same edges as the canonical baseline - normalization
                                          # happens after binning, doesn't change bin edges.

    X_train, _, keys_train = load_or_build_dataset(
        train_df, bin_edges, CLASSES, "train_dens_pc",
        include_n_points=True, normalize_density=True)
    X_val, _, keys_val = load_or_build_dataset(
        val_df, bin_edges, CLASSES, "val_dens_pc",
        include_n_points=True, normalize_density=True)

    y_train = regrouped_labels(train_df_raw, keys_train, CLASSES)
    y_val = regrouped_labels(val_df_raw, keys_val, CLASSES)

    print(f"train: {len(y_train)} instances, val: {len(y_val)} instances, in_dim={X_train.shape[1]}")
    print(f"train class counts: {dict(zip(CLASSES, np.bincount(y_train, minlength=len(CLASSES))))}")

    X_train_t, y_train_t = torch.tensor(X_train), torch.tensor(y_train)
    X_val_t, y_val_t = torch.tensor(X_val), torch.tensor(y_val)

    model = PaperMLP(in_dim=X_train.shape[1], hidden_dim=HIDDEN_DIM, n_classes=len(CLASSES))
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y_train, len(CLASSES)))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    n_train = len(y_train)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
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
            train_acc = (train_logits.argmax(1) == y_train_t).float().mean().item()
            val_acc = (val_logits.argmax(1) == y_val_t).float().mean().item()
            val_loss = loss_fn(val_logits, y_val_t).item()
        history["train_loss"].append(total_loss / n_train)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        print(f"epoch {epoch + 1:4d}/{EPOCHS}  train_loss {history['train_loss'][-1]:.4f}  "
              f"val_loss {val_loss:.4f}  train_acc {train_acc:.3f}  val_acc {val_acc:.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "history.npy", history, allow_pickle=True)
    torch.save(model.state_dict(), OUT_DIR / "model.pt")
    print(f"\nSaved {OUT_DIR / 'history.npy'} and {OUT_DIR / 'model.pt'}")

    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_t).argmax(1).numpy()
    true_names = pd.Series(y_val).map(lambda i: CLASSES[i])
    pred_names = pd.Series(val_preds).map(lambda i: CLASSES[i])
    cm = pd.crosstab(true_names, pred_names, rownames=["true"], colnames=["pred"]).reindex(
        index=CLASSES, columns=CLASSES, fill_value=0)
    print("\nValidation confusion matrix:")
    print(cm)

    recall = np.diag(cm.to_numpy()) / cm.to_numpy().sum(axis=1)
    print("\nPer-class recall:")
    for c, r in zip(CLASSES, recall):
        print(f"  {c}: {r:.3f}")
    print(f"raw accuracy: {(val_preds == y_val).mean():.3f}")
    print(f"balanced accuracy (mean recall): {recall.mean():.3f}")

    plot_predictions_grid(val_df, keys_val, y_val, val_preds, CLASSES,
                           OUT_DIR / "predictions_grid.png", n=9, rng=rng)

    epochs_axis = np.arange(1, EPOCHS + 1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs_axis, history["train_loss"], label="train", color="tab:red")
    ax.plot(epochs_axis, history["val_loss"], label="val", color="tab:orange")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (class-weighted cross-entropy)")
    ax.set_title(f"Density-normalized + regrouped: cost vs. epoch (n_train={n_train}, n_val={len(y_val)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cost.png", dpi=150)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs_axis, history["train_acc"], label="train", color="tab:blue")
    ax.plot(epochs_axis, history["val_acc"], label="val", color="tab:orange")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Density-normalized + regrouped: accuracy vs. epoch (n_train={n_train}, n_val={len(y_val)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "accuracy.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm.to_numpy(), cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Density-normalized + regrouped: validation confusion matrix (n_val={len(y_val)})")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            count = cm.to_numpy()[i, j]
            color = "white" if count > cm.to_numpy().max() / 2 else "black"
            ax.text(j, i, str(count), ha="center", va="center", color=color)
    fig.colorbar(im, ax=ax, label="count")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "validation.png", dpi=150)
    print(f"\nSaved plots to {OUT_DIR}")
