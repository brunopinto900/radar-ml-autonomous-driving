"""Day 6 real run: paper-faithful 3-layer architecture, full train/val splits,
final batch size (see MLP_DESIGN.md's "Architecture" section and
DESIGN_DECISIONS.md decisions 3-4). Timing confirmed first with a 2-epoch pass
(results/mlp_full_run/timing_run_history.npy, ~1.4s/epoch). The first full
1000-epoch run (results/mlp_full_run/*_1000epoch.*) showed val accuracy/loss
plateauing by ~epoch 20 with no overfitting (MLP_FINDINGS.md) - EPOCHS below
was dropped to 20 accordingly. Outputs are tagged with the epoch count so
different-length runs don't overwrite each other. Reuses histogram_features.py
and train_mlp.py's caching/class-weight helpers rather than reimplementing them.
"""
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dataloader import RESULTS_DIR, plot_predictions_grid  # noqa: E402
from histogram_features import FEATURES, N_BINS, add_relative_xy, fit_bin_edges  # noqa: E402
from train_mlp import CLASSES, class_weights, load_or_build_dataset  # noqa: E402

OUT_DIR = RESULTS_DIR / "mlp_full_run"

SEED = 42
HIDDEN_DIM = 16
LR = 1e-5  # paper's value (MLP_DESIGN.md) - tuned for their batch_size=64, not
           # our 256. The 2-epoch timing pass showed real, continuing loss/acc
           # movement at this lr through the deeper 3-layer net, so keeping it
           # for the real run rather than guessing a correction with no evidence.
BATCH_SIZE = 256  # DESIGN_DECISIONS.md decision 4
EPOCHS = 20  # not the paper's 1000 - see MLP_FINDINGS.md, val plateaued by ~epoch 20
TAG = f"{EPOCHS}epoch"


class PaperMLP(nn.Module):
    """Paper-faithful architecture (MLP_DESIGN.md): 80 -> 16 -> 16 -> 5."""

    def __init__(self, in_dim: int, hidden_dim: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    train_df = add_relative_xy(pd.read_parquet(RESULTS_DIR / "train_points.parquet"))
    val_df = add_relative_xy(pd.read_parquet(RESULTS_DIR / "val_points.parquet"))
    bin_edges = fit_bin_edges(train_df)

    X_train, y_train, keys_train = load_or_build_dataset(train_df, bin_edges, CLASSES, "train")
    X_val, y_val, keys_val = load_or_build_dataset(val_df, bin_edges, CLASSES, "val")
    print(f"train: {len(y_train)} instances, val: {len(y_val)} instances (full splits)")

    X_train_t, y_train_t = torch.tensor(X_train), torch.tensor(y_train)
    X_val_t, y_val_t = torch.tensor(X_val), torch.tensor(y_val)

    model = PaperMLP(in_dim=len(FEATURES) * N_BINS, hidden_dim=HIDDEN_DIM, n_classes=len(CLASSES))
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y_train, len(CLASSES)))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Linear layers in model.net, input-side to output-side - tracked separately
    # to check for vanishing/exploding gradients (does grad norm shrink/blow up
    # from the output layer back to the input layer as depth increases?).
    grad_layers = {"layer1_in": model.net[0], "layer2_mid": model.net[2], "layer3_out": model.net[4]}

    n_train = len(y_train)
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "lr": [],
        "epoch_time": [],
        "grad_norm_layer1_in": [],
        "grad_norm_layer2_mid": [],
        "grad_norm_layer3_out": [],
    }
    for epoch in range(EPOCHS):
        t0 = time.time()
        model.train()
        perm = torch.randperm(n_train)
        total_loss = 0.0
        batch_grad_norms = {name: [] for name in grad_layers}
        for start in range(0, n_train, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            xb, yb = X_train_t[idx], y_train_t[idx]
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            for name, layer in grad_layers.items():
                batch_grad_norms[name].append(layer.weight.grad.norm().item())
            optimizer.step()
            total_loss += loss.item() * len(idx)

        model.eval()
        with torch.no_grad():
            train_logits = model(X_train_t)
            val_logits = model(X_val_t)
            train_acc = (train_logits.argmax(1) == y_train_t).float().mean().item()
            val_acc = (val_logits.argmax(1) == y_val_t).float().mean().item()
            val_loss = loss_fn(val_logits, y_val_t).item()
        epoch_time = time.time() - t0

        history["train_loss"].append(total_loss / n_train)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["epoch_time"].append(epoch_time)
        for name in grad_layers:
            history[f"grad_norm_{name}"].append(float(np.mean(batch_grad_norms[name])))
        print(f"epoch {epoch + 1:4d}/{EPOCHS}  train_loss {history['train_loss'][-1]:.4f}  "
              f"val_loss {val_loss:.4f}  train_acc {train_acc:.3f}  val_acc {val_acc:.3f}  "
              f"grad_norms [in={history['grad_norm_layer1_in'][-1]:.4f} "
              f"mid={history['grad_norm_layer2_mid'][-1]:.4f} "
              f"out={history['grad_norm_layer3_out'][-1]:.4f}]  time {epoch_time:.1f}s")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    history_path = OUT_DIR / f"history_{TAG}.npy"
    np.save(history_path, history, allow_pickle=True)
    print(f"\nSaved {history_path}")

    model_path = OUT_DIR / f"model_{TAG}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Saved {model_path}")

    total_min = sum(history["epoch_time"]) / 60
    print(f"Total training time: {total_min:.1f} min "
          f"(mean {np.mean(history['epoch_time']):.2f}s/epoch)")

    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_t).argmax(1).numpy()
    true_names = pd.Series(y_val).map(lambda i: CLASSES[i])
    pred_names = pd.Series(val_preds).map(lambda i: CLASSES[i])
    cm = pd.crosstab(true_names, pred_names, rownames=["true"], colnames=["pred"]).reindex(
        index=CLASSES, columns=CLASSES, fill_value=0)
    print("\nValidation confusion matrix:")
    print(cm)

    plot_predictions_grid(val_df, keys_val, y_val, val_preds, CLASSES,
                           OUT_DIR / f"predictions_grid_{TAG}.png", n=9, rng=rng)

    epochs_axis = np.arange(1, EPOCHS + 1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs_axis, history["train_loss"], label="train", color="tab:red")
    ax.plot(epochs_axis, history["val_loss"], label="val", color="tab:orange")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (class-weighted cross-entropy)")
    ax.set_title(f"Full run ({TAG}): cost vs. epoch (n_train={n_train}, n_val={len(y_val)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"cost_{TAG}.png", dpi=150)
    print(f"Saved {OUT_DIR / f'cost_{TAG}.png'}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs_axis, history["train_acc"], label="train", color="tab:blue")
    ax.plot(epochs_axis, history["val_acc"], label="val", color="tab:orange")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Full run ({TAG}): accuracy vs. epoch (n_train={n_train}, n_val={len(y_val)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"accuracy_{TAG}.png", dpi=150)
    print(f"Saved {OUT_DIR / f'accuracy_{TAG}.png'}")

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, color in zip(grad_layers, ["tab:blue", "tab:green", "tab:red"]):
        ax.plot(epochs_axis, history[f"grad_norm_{name}"], label=name, color=color)
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean per-batch gradient L2 norm (weight matrix)")
    ax.set_yscale("log")
    ax.set_title(f"Full run ({TAG}): gradient norms vs. epoch, input-side to output-side")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"grad_norms_{TAG}.png", dpi=150)
    print(f"Saved {OUT_DIR / f'grad_norms_{TAG}.png'}")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm.to_numpy(), cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Full run ({TAG}): validation confusion matrix (n_val={len(y_val)})")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            count = cm.to_numpy()[i, j]
            color = "white" if count > cm.to_numpy().max() / 2 else "black"
            ax.text(j, i, str(count), ha="center", va="center", color=color)
    fig.colorbar(im, ax=ax, label="count")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"validation_{TAG}.png", dpi=150)
    print(f"Saved {OUT_DIR / f'validation_{TAG}.png'}")
