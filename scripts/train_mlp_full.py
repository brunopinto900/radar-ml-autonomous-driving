"""Day 6 real run: paper-faithful 3-layer architecture, full train/val splits,
final batch size (see MLP_DESIGN.md's "Architecture" section and
DESIGN_DECISIONS.md decisions 3-4). Timing confirmed first with a 2-epoch pass
(results/mlp_full_run/baseline_20epoch_h16/timing_2epoch_check.npy, ~1.4s/epoch).
The first full 1000-epoch run (results/mlp_full_run/epoch_ablation_1000epoch/)
showed val accuracy/loss plateauing by ~epoch 20 with no overfitting
(MLP_FINDINGS.md) - EPOCHS below was dropped to 20 accordingly. Each run writes
to its own results/mlp_full_run/<EXPERIMENT>/ subfolder (set EXPERIMENT to match
whatever HIDDEN_DIM/EPOCHS/MIN_TRAIN_POINTS you're running) so different runs
don't overwrite each other. Reuses histogram_features.py and train_mlp.py's
caching/class-weight helpers rather than reimplementing them.
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
from histogram_features import (FEATURES, GROUP_KEY, N_BINS, add_relative_vr,  # noqa: E402
                                 add_relative_xy, fit_bin_edges)
from train_mlp import CLASSES, class_weights, load_or_build_dataset  # noqa: E402

OUT_ROOT = RESULTS_DIR / "mlp_full_run"

SEED = 42
HIDDEN_DIM = 16  # canonical baseline (MLP_DESIGN.md). Bump to run a capacity
                 # ablation (MLP_FINDINGS.md) - pair with EXPERIMENT below so
                 # ablation runs don't overwrite this checkpoint.
LR = 1e-5  # paper's value (MLP_DESIGN.md) - tuned for their batch_size=64, not
           # our 256. The 2-epoch timing pass showed real, continuing loss/acc
           # movement at this lr through the deeper 3-layer net, so keeping it
           # for the real run rather than guessing a correction with no evidence.
BATCH_SIZE = 256  # DESIGN_DECISIONS.md decision 4
EPOCHS = 20  # not the paper's 1000 - see MLP_FINDINGS.md, val plateaued by ~epoch 20. Bump to
            # 80 to reproduce the gradient-step-matched min4pts rerun (section 6): "20 epochs"
            # means ~27,700 total gradient steps for the full 354,277-instance train set
            # (1,385 batches/epoch) but only ~7,100 for the 91,269-instance min4pts set (357
            # batches/epoch) - steps/epoch scales with dataset size at fixed BATCH_SIZE, so 80
            # epochs there matches the baseline's total step count instead of its epoch count.
MIN_TRAIN_POINTS = 1  # >1 drops sparse instances from TRAINING only (val stays full/
                      # unfiltered by default, see MIN_VAL_POINTS below) - tests whether
                      # near-empty training instances are actively harmful (noise) vs.
                      # weak-but-real signal, MLP_FINDINGS.md. bin_edges are still fit on the
                      # FULL unfiltered train split below, so this isolates one variable.
MIN_VAL_POINTS = 1  # >1 drops sparse instances from VALIDATION too - answers a different
                    # question than MIN_TRAIN_POINTS: not "is sparse training data harmful"
                    # but "how good is this checkpoint at the instances it could plausibly
                    # be expected to get right" - i.e. is the min-train-points checkpoint's
                    # poor overall val accuracy just it being scored on cases (sparse val
                    # instances) it was never trained to handle in the first place.
INCLUDE_N_POINTS = False  # appends n_points as an explicit 81st input dim (MLP_FINDINGS.md
                          # section 7) - ~fully redundant with the existing 80 dims in
                          # principle (r=0.985-0.997 with each block's own sum), so this tests
                          # whether removing the optimization burden of re-deriving it helps
                          # this small/slowly-trained network, against the real risk of
                          # amplifying the large_vehicle shortcut already observed (section 6).
                          # Uses separate "_pc"-suffixed caches, doesn't touch the 80-dim ones.
INCLUDE_VR_REL = False  # appends (or, with REPLACE_VR_WITH_VR_REL below, swaps in for
                       # vr_compensated) a 16-bin vr_rel block (Doppler spread / micro-Doppler,
                       # median-centered, explicit +-3.0 m/s range - histogram_features.py's
                       # VR_REL_RANGE) - real signal validated in FEATURE_MAP.md but deferred
                       # from the paper-faithful v1 baseline (MLP_DESIGN.md), unlike
                       # INCLUDE_N_POINTS this is NOT redundant with any existing feature.
REPLACE_VR_WITH_VR_REL = False  # only matters if INCLUDE_VR_REL=True. False (default): vr_rel
                               # is a 6th block ADDED alongside vr_compensated (81/96-dim).
                               # True: vr_rel SWAPS IN for vr_compensated in place, staying at
                               # 80-dim - tests vr_rel as a replacement, not an addition
                               # (MLP_FINDINGS.md section 9). Uses separate "_vr"/"_vrswap"
                               # -suffixed caches, doesn't touch the plain 80-dim ones.

EXPERIMENT = "baseline_20epoch_h16"  # output subfolder under results/mlp_full_run/ - set
                                      # this together with HIDDEN_DIM/EPOCHS/
                                      # MIN_TRAIN_POINTS/MIN_VAL_POINTS/INCLUDE_N_POINTS/
                                      # INCLUDE_VR_REL/REPLACE_VR_WITH_VR_REL above when
                                      # running an ablation, e.g. "vr_rel_feature",
                                      # "vr_rel_replace_feature", "point_count_feature",
                                      # "capacity_ablation_h64", "min_train_points_ablation",
                                      # "epoch_ablation_1000epoch",
                                      # "min_train_points_epoch_matched"
OUT_DIR = OUT_ROOT / EXPERIMENT


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
    if INCLUDE_VR_REL:
        train_df = add_relative_vr(train_df)
        val_df = add_relative_vr(val_df)
    bin_edges = fit_bin_edges(train_df, INCLUDE_VR_REL)  # fit on the FULL unfiltered train
                                          # split always - keeps this the same variable as the
                                          # canonical run (vr_rel's own edges are an explicit
                                          # range, not fit from data - see VR_REL_RANGE).

    train_cache_name = "train"
    if MIN_TRAIN_POINTS > 1:
        n_points = train_df.groupby(GROUP_KEY).size()
        keep_keys = n_points[n_points >= MIN_TRAIN_POINTS].index
        n_before = len(n_points)
        train_df = train_df.set_index(GROUP_KEY).loc[keep_keys].reset_index()
        train_cache_name = f"train_min{MIN_TRAIN_POINTS}pts"
        print(f"Filtered train instances: {n_before} -> {len(keep_keys)} "
              f"(dropped {n_before - len(keep_keys)} with <{MIN_TRAIN_POINTS} points)")

    val_cache_name = "val"
    if MIN_VAL_POINTS > 1:
        n_points_val = val_df.groupby(GROUP_KEY).size()
        keep_keys_val = n_points_val[n_points_val >= MIN_VAL_POINTS].index
        n_before_val = len(n_points_val)
        val_df = val_df.set_index(GROUP_KEY).loc[keep_keys_val].reset_index()
        val_cache_name = f"val_min{MIN_VAL_POINTS}pts"
        print(f"Filtered val instances: {n_before_val} -> {len(keep_keys_val)} "
              f"(dropped {n_before_val - len(keep_keys_val)} with <{MIN_VAL_POINTS} points)")

    if INCLUDE_N_POINTS:
        train_cache_name += "_pc"
        val_cache_name += "_pc"
    if INCLUDE_VR_REL:
        train_cache_name += "_vrswap" if REPLACE_VR_WITH_VR_REL else "_vr"
        val_cache_name += "_vrswap" if REPLACE_VR_WITH_VR_REL else "_vr"

    X_train, y_train, keys_train = load_or_build_dataset(
        train_df, bin_edges, CLASSES, train_cache_name, INCLUDE_N_POINTS, INCLUDE_VR_REL,
        REPLACE_VR_WITH_VR_REL)
    X_val, y_val, keys_val = load_or_build_dataset(
        val_df, bin_edges, CLASSES, val_cache_name, INCLUDE_N_POINTS, INCLUDE_VR_REL,
        REPLACE_VR_WITH_VR_REL)
    print(f"train: {len(y_train)} instances, val: {len(y_val)} instances")

    X_train_t, y_train_t = torch.tensor(X_train), torch.tensor(y_train)
    X_val_t, y_val_t = torch.tensor(X_val), torch.tensor(y_val)

    in_dim = len(FEATURES) * N_BINS
    if INCLUDE_VR_REL and not REPLACE_VR_WITH_VR_REL:
        in_dim += N_BINS
    in_dim += 1 if INCLUDE_N_POINTS else 0
    model = PaperMLP(in_dim=in_dim, hidden_dim=HIDDEN_DIM, n_classes=len(CLASSES))
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
    history_path = OUT_DIR / "history.npy"
    np.save(history_path, history, allow_pickle=True)
    print(f"\nSaved {history_path}")

    model_path = OUT_DIR / "model.pt"
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
                           OUT_DIR / "predictions_grid.png", n=9, rng=rng)

    epochs_axis = np.arange(1, EPOCHS + 1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs_axis, history["train_loss"], label="train", color="tab:red")
    ax.plot(epochs_axis, history["val_loss"], label="val", color="tab:orange")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (class-weighted cross-entropy)")
    ax.set_title(f"Full run ({EXPERIMENT}): cost vs. epoch (n_train={n_train}, n_val={len(y_val)})")
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
    ax.set_title(f"Full run ({EXPERIMENT}): accuracy vs. epoch (n_train={n_train}, n_val={len(y_val)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "accuracy.png", dpi=150)
    print(f"Saved {OUT_DIR / 'accuracy.png'}")

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, color in zip(grad_layers, ["tab:blue", "tab:green", "tab:red"]):
        ax.plot(epochs_axis, history[f"grad_norm_{name}"], label=name, color=color)
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean per-batch gradient L2 norm (weight matrix)")
    ax.set_yscale("log")
    ax.set_title(f"Full run ({EXPERIMENT}): gradient norms vs. epoch, input-side to output-side")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "grad_norms.png", dpi=150)
    print(f"Saved {OUT_DIR / 'grad_norms.png'}")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm.to_numpy(), cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Full run ({EXPERIMENT}): validation confusion matrix (n_val={len(y_val)})")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            count = cm.to_numpy()[i, j]
            color = "white" if count > cm.to_numpy().max() / 2 else "black"
            ax.text(j, i, str(count), ha="center", va="center", color=color)
    fig.colorbar(im, ax=ax, label="count")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "validation.png", dpi=150)
    print(f"Saved {OUT_DIR / 'validation.png'}")
