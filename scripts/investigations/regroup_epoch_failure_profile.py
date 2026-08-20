"""Per-epoch tracking of the true large_vehicle instances misclassified as car (density-
normalized + regrouped setup, scripts/train_mlp_density_regroup.py) - not just the aggregate
accuracy curve, but the actual feature profile (n_points, extent, mean_rcs, mean_vr_compensated,
vr_rel_std, mean_range_sc) of whichever instances are failing at each epoch. Answers whether
the population of failures is stable over training or shifts character as the boundary moves
(MLP_FINDINGS.md section 10/11 discussion: why does large_vehicle recall peak early then
decline with more training).

Deterministic re-run (same seed/data/hyperparameters, 50 epochs) - per-epoch failure identity
isn't recoverable from the final checkpoint alone.
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

from dataloader import RESULTS_DIR  # noqa: E402
from histogram_features import GROUP_KEY, add_relative_vr, add_relative_xy, fit_bin_edges  # noqa: E402
from train_mlp import CLASSES, class_weights, load_or_build_dataset  # noqa: E402
from train_mlp_full import BATCH_SIZE, HIDDEN_DIM, LR, PaperMLP, SEED  # noqa: E402
from train_mlp_regroup import regrouped_labels  # noqa: E402

EPOCHS = 50
OUT_DIR = RESULTS_DIR / "mlp_full_run" / "density_norm_regroup_50epoch"
STAT_COLS = ["n_points", "extent", "mean_rcs", "mean_vr_compensated", "vr_rel_std", "mean_range_sc"]

if __name__ == "__main__":
    torch.manual_seed(SEED)

    train_df_raw = pd.read_parquet(RESULTS_DIR / "train_points.parquet")
    val_df_raw = pd.read_parquet(RESULTS_DIR / "val_points.parquet")
    train_df = add_relative_xy(train_df_raw)
    val_df = add_relative_vr(add_relative_xy(val_df_raw))
    bin_edges = fit_bin_edges(train_df)

    X_train, _, keys_train = load_or_build_dataset(
        train_df, bin_edges, CLASSES, "train_dens_pc", include_n_points=True, normalize_density=True)
    X_val, _, keys_val = load_or_build_dataset(
        val_df, bin_edges, CLASSES, "val_dens_pc", include_n_points=True, normalize_density=True)
    y_train = regrouped_labels(train_df_raw, keys_train, CLASSES)
    y_val = regrouped_labels(val_df_raw, keys_val, CLASSES)

    # Per-instance feature stats, computed once - looked up by epoch instead of recomputed.
    rows = []
    for key, pts in val_df.groupby(GROUP_KEY):
        n = len(pts)
        extent = (np.hypot(pts["x_rel"].max() - pts["x_rel"].min(),
                            pts["y_rel"].max() - pts["y_rel"].min()) if n > 1 else 0.0)
        vr_rel_std = pts["vr_rel"].std() if n > 1 else 0.0
        rows.append((*key, n, extent, pts["rcs"].mean(), pts["vr_compensated"].mean(),
                     vr_rel_std, pts["range_sc"].mean()))
    inst_df = pd.DataFrame(rows, columns=GROUP_KEY + STAT_COLS)
    keys_val = keys_val.reset_index(drop=True).merge(inst_df, on=GROUP_KEY, how="left")

    LV_IDX = CLASSES.index("large_vehicle")
    CAR_IDX = CLASSES.index("car")
    lv_mask = (y_val == LV_IDX)

    X_train_t, y_train_t = torch.tensor(X_train), torch.tensor(y_train)
    X_val_t = torch.tensor(X_val)

    model = PaperMLP(in_dim=X_train.shape[1], hidden_dim=HIDDEN_DIM, n_classes=len(CLASSES))
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y_train, len(CLASSES)))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    n_train = len(y_train)
    epoch_rows = []
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n_train)
        for start in range(0, n_train, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            xb, yb = X_train_t[idx], y_train_t[idx]
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t).argmax(1).numpy()

        fail_mask = lv_mask & (val_preds == CAR_IDX)
        n_fail = int(fail_mask.sum())
        n_lv = int(lv_mask.sum())
        row = {"epoch": epoch + 1, "n_lv": n_lv, "n_fail": n_fail,
               "fail_rate": n_fail / n_lv if n_lv else float("nan")}
        for col in STAT_COLS:
            vals = keys_val.loc[fail_mask, col]
            row[f"{col}_mean"] = vals.mean() if len(vals) else float("nan")
            row[f"{col}_median"] = vals.median() if len(vals) else float("nan")
        epoch_rows.append(row)
        print(f"epoch {epoch + 1:3d}/{EPOCHS}  n_fail={n_fail:4d}/{n_lv}  "
              f"n_points_mean={row['n_points_mean']:.2f}  extent_mean={row['extent_mean']:.2f}")

    epoch_df = pd.DataFrame(epoch_rows)
    epoch_df.to_csv(OUT_DIR / "epoch_failure_profile.csv", index=False)
    print(f"\nSaved {OUT_DIR / 'epoch_failure_profile.csv'}")

    fig, axes = plt.subplots(3, 2, figsize=(13, 12), sharex=True)
    for ax, col in zip(axes.flat, STAT_COLS):
        ax.plot(epoch_df["epoch"], epoch_df[f"{col}_mean"], label="mean", color="tab:blue")
        ax.plot(epoch_df["epoch"], epoch_df[f"{col}_median"], label="median", color="tab:orange")
        ax.set_title(col)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Feature profile of true large_vehicle instances misclassified as car, per epoch")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "epoch_failure_profile.png", dpi=150)
    print(f"Saved {OUT_DIR / 'epoch_failure_profile.png'}")
