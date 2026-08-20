"""Class-count ("class imbalance") bar charts, one per results/mlp_full_run/
subfolder - shows how the 5-class distribution differs between train/val
splits and between the sparse/dense val subsets, since composition shift (not
just data volume) came up repeatedly as a candidate explanation for the
min_train_points_ablation results (MLP_FINDINGS.md section 6). Reads only
cached labels/points, no retraining or feature rebuilding.
"""
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/ core modules

from dataloader import RESULTS_DIR  # noqa: E402
from histogram_features import GROUP_KEY  # noqa: E402
from train_mlp import CLASSES  # noqa: E402

RUN_DIR = RESULTS_DIR / "mlp_full_run"
CACHE_DIR = RESULTS_DIR / "mlp_feature_cache"

SPLIT_COLORS = {"train": "tab:blue", "val": "tab:orange"}


def class_counts(y: np.ndarray) -> pd.Series:
    return pd.Series(np.bincount(y, minlength=len(CLASSES)), index=CLASSES)


def plot_class_counts(counts_by_split: dict, title: str, path) -> None:
    n_splits = len(counts_by_split)
    x = np.arange(len(CLASSES))
    width = 0.8 / n_splits

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, (split, counts) in enumerate(counts_by_split.items()):
        offset = (i - (n_splits - 1) / 2) * width
        color = SPLIT_COLORS.get(split, f"C{i}")
        bars = ax.bar(x + offset, counts.to_numpy(), width, label=split, color=color)
        for rect, count in zip(bars, counts.to_numpy()):
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(), f"{count:,}",
                    ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x, CLASSES, rotation=20, ha="right")
    ax.set_ylabel("instance count")
    ax.set_title(title)
    if n_splits > 1:
        ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")


if __name__ == "__main__":
    train_y = np.load(CACHE_DIR / "train_y.npy")
    train_min4pts_y = np.load(CACHE_DIR / "train_min4pts_y.npy")
    val_y = np.load(CACHE_DIR / "val_y.npy")
    val_min4pts_y = np.load(CACHE_DIR / "val_min4pts_y.npy")

    train_counts = class_counts(train_y)
    train_min4pts_counts = class_counts(train_min4pts_y)
    val_counts = class_counts(val_y)
    val_min4pts_counts = class_counts(val_min4pts_y)

    # full-data train/val split - baseline, epoch ablation, and capacity ablation all share it
    for folder in ["baseline_20epoch_h16", "epoch_ablation_1000epoch", "capacity_ablation_h64"]:
        plot_class_counts(
            {"train": train_counts, "val": val_counts},
            f"class counts ({folder}): train (n={len(train_y)}) vs. val (n={len(val_y)})",
            RUN_DIR / folder / "class_imbalance.png",
        )

    # sparse-training-filter ablations
    plot_class_counts(
        {"train": train_min4pts_counts, "val": val_counts},
        f"class counts (min_train_points_ablation): train (n={len(train_min4pts_y)}, "
        f"<4pt dropped) vs. val (n={len(val_y)}, full)",
        RUN_DIR / "min_train_points_ablation" / "class_imbalance.png",
    )
    plot_class_counts(
        {"train": train_min4pts_counts, "val": val_min4pts_counts},
        f"class counts (min_train_val_points_ablation): train (n={len(train_min4pts_y)}) vs. "
        f"val (n={len(val_min4pts_y)}) - both <4pt dropped",
        RUN_DIR / "min_train_val_points_ablation" / "class_imbalance.png",
    )

    # large_vehicle/car diagnostics run on the full baseline val set
    plot_class_counts(
        {"val": val_counts},
        f"class counts (large_vehicle_car_confusion): full val set (n={len(val_y)})",
        RUN_DIR / "large_vehicle_car_confusion" / "class_imbalance.png",
    )

    # sparse/dense val subsets used by the two *_subset_diagnostics folders
    val_keys = pd.read_parquet(CACHE_DIR / "val_keys.parquet")
    n_points = pd.read_parquet(RESULTS_DIR / "val_points.parquet").groupby(GROUP_KEY).size()
    n_points.name = "n_points"
    val_meta = val_keys.copy()
    val_meta["true_name"] = [CLASSES[i] for i in val_y]
    val_meta = val_meta.merge(n_points, on=GROUP_KEY, how="left")

    sparse_counts = val_meta.loc[val_meta["n_points"] <= 3, "true_name"].value_counts().reindex(
        CLASSES, fill_value=0)
    dense_counts = val_meta.loc[val_meta["n_points"] >= 4, "true_name"].value_counts().reindex(
        CLASSES, fill_value=0)

    plot_class_counts(
        {"val, ≤3 pts": sparse_counts},
        f"class counts (sparse_subset_diagnostics): val instances with ≤3 points "
        f"(n={sparse_counts.sum()})",
        RUN_DIR / "sparse_subset_diagnostics" / "class_imbalance.png",
    )
    plot_class_counts(
        {"val, ≥4 pts": dense_counts},
        f"class counts (dense_subset_diagnostics): val instances with ≥4 points "
        f"(n={dense_counts.sum()})",
        RUN_DIR / "dense_subset_diagnostics" / "class_imbalance.png",
    )
