"""Retrains the baseline model with a revised car/large_vehicle grouping (MLP_FINDINGS.md
section 10): truck and raw large_vehicle merge into car - both are mutually close to car in
the separability probe (sparse AUC 0.811-0.823, vs. 0.844-0.864 for every pair involving bus).
bus (+ the negligible train label, 57 instances) becomes the new, much smaller large_vehicle
class alone, instead of DESIGN_DECISIONS.md decision 1's original scheme (truck+bus+train+
large_vehicle all merged together).

Everything else identical to the canonical baseline (results/mlp_full_run/baseline_20epoch_h16/)
- same HIDDEN_DIM/LR/BATCH_SIZE/EPOCHS/architecture, same 80-dim features. X doesn't depend on
class labels (only on which points belong to which instance, unaffected by regrouping), so this
reuses the already-cached baseline feature vectors and only recomputes y from each instance's
raw label_name via REGROUP below, instead of rebuilding features from scratch.
"""
import numpy as np
import pandas as pd
import torch
from torch import nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/ core modules

from dataloader import RESULTS_DIR, plot_predictions_grid  # noqa: E402
from histogram_features import GROUP_KEY  # noqa: E402
from train_mlp import CLASSES, class_weights, load_or_build_dataset  # noqa: E402
from train_mlp_full import BATCH_SIZE, HIDDEN_DIM, LR, PaperMLP, SEED  # noqa: E402

EPOCHS = 1000  # the 20-epoch run (car_large_vehicle_regroup/) hadn't plateaued - cost was
               # still visibly decreasing with no train/val gap (no overfitting), same plateau
               # check MLP_FINDINGS.md section 2 ran for the original grouping's baseline.
OUT_DIR = RESULTS_DIR / "mlp_full_run" / "car_large_vehicle_regroup_1000epoch"

REGROUP = {
    "car": "car",
    "truck": "car",
    "large_vehicle": "car",
    "bus": "large_vehicle",
    "train": "large_vehicle",
    "bicycle": "two_wheeler",
    "motorized_two_wheeler": "two_wheeler",
    "pedestrian": "pedestrian",
    "pedestrian_group": "pedestrian_group",
}


def regrouped_labels(df: pd.DataFrame, keys: pd.DataFrame, classes: list[str]) -> np.ndarray:
    label_by_key = df.groupby(GROUP_KEY)["label_name"].first().rename("label_name")
    merged = keys.merge(label_by_key, on=GROUP_KEY, how="left")
    if merged["label_name"].isna().any():
        raise ValueError("some cached instances had no matching label_name - GROUP_KEY join failed")
    new_class = merged["label_name"].map(REGROUP)
    if new_class.isna().any():
        missing = sorted(merged.loc[new_class.isna(), "label_name"].unique())
        raise ValueError(f"REGROUP is missing raw labels present in the data: {missing}")
    class_to_idx = {name: i for i, name in enumerate(classes)}
    return new_class.map(class_to_idx).to_numpy()


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    train_df = pd.read_parquet(RESULTS_DIR / "train_points.parquet")
    val_df = pd.read_parquet(RESULTS_DIR / "val_points.parquet")

    # Reuse the canonical baseline's cached 80-dim features (unaffected by regrouping).
    X_train, _, keys_train = load_or_build_dataset(train_df, {}, CLASSES, "train")
    X_val, _, keys_val = load_or_build_dataset(val_df, {}, CLASSES, "val")

    y_train = regrouped_labels(train_df, keys_train, CLASSES)
    y_val = regrouped_labels(val_df, keys_val, CLASSES)

    print(f"train: {len(y_train)} instances, val: {len(y_val)} instances")
    print(f"train class counts: {dict(zip(CLASSES, np.bincount(y_train, minlength=len(CLASSES))))}")
    print(f"val class counts:   {dict(zip(CLASSES, np.bincount(y_val, minlength=len(CLASSES))))}")

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
    ax.set_title(f"Regrouped car/large_vehicle: cost vs. epoch (n_train={n_train}, n_val={len(y_val)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cost.png", dpi=150)
    print(f"Saved {OUT_DIR / 'cost.png'}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs_axis, history["train_acc"], label="train", color="tab:blue")
    ax.plot(epochs_axis, history["val_acc"], label="val", color="tab:orange")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Regrouped car/large_vehicle: accuracy vs. epoch (n_train={n_train}, n_val={len(y_val)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "accuracy.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm.to_numpy(), cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Regrouped car/large_vehicle: validation confusion matrix (n_val={len(y_val)})")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            count = cm.to_numpy()[i, j]
            color = "white" if count > cm.to_numpy().max() / 2 else "black"
            ax.text(j, i, str(count), ha="center", va="center", color=color)
    fig.colorbar(im, ax=ax, label="count")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "validation.png", dpi=150)
    print(f"\nSaved plots to {OUT_DIR}")
