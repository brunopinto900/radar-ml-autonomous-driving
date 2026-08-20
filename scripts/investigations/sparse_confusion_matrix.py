"""Ad-hoc diagnostic: standard confusion matrix (rows = true class, columns =
predicted class), restricted to val instances with <=POINT_THRESHOLD points.
Replaces the earlier stacked-bar version (sparse_instances_by_prediction.py,
removed) - that plot grouped by prediction, which made "given a true class,
what does it get predicted as" (the question actually being asked) hard to
read off the chart directly. A confusion matrix answers that with one row.
Reuses a saved checkpoint and cached val features (val is always the full/
unfiltered split, even for checkpoints trained on a filtered train set - see
MLP_FINDINGS.md), no retraining.
"""
import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/ core modules

from dataloader import RESULTS_DIR  # noqa: E402
from histogram_features import FEATURES, GROUP_KEY, N_BINS  # noqa: E402
from train_mlp import CLASSES  # noqa: E402
from train_mlp_full import PaperMLP  # noqa: E402

RUN_DIR = RESULTS_DIR / "mlp_full_run"
CACHE_DIR = RESULTS_DIR / "mlp_feature_cache"

MODEL_EXPERIMENT = "baseline_20epoch_h16"  # which trained checkpoint to diagnose - folder
                                            # under results/mlp_full_run/, e.g.
                                            # "min_train_points_ablation" for the min4pts rerun
HIDDEN_DIM = 16  # must match that experiment's architecture
MODEL_PATH = RUN_DIR / MODEL_EXPERIMENT / "model.pt"
LABEL = "baseline"  # short name used in this diagnostic's own output filenames below - keep in
                    # sync with MODEL_EXPERIMENT (e.g. "min4pts" for min_train_points_ablation)

OUT_DIR = RUN_DIR / "sparse_subset_diagnostics"
POINT_THRESHOLD = 3  # instances with n_points <= this are included


if __name__ == "__main__":
    X_val = np.load(CACHE_DIR / "val_X.npy")
    y_val = np.load(CACHE_DIR / "val_y.npy")
    keys_val = pd.read_parquet(CACHE_DIR / "val_keys.parquet")

    model = PaperMLP(in_dim=len(FEATURES) * N_BINS, hidden_dim=HIDDEN_DIM, n_classes=len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    with torch.no_grad():
        val_preds = model(torch.tensor(X_val)).argmax(1).numpy()

    keys = keys_val.copy()
    keys["true_name"] = [CLASSES[i] for i in y_val]
    keys["pred_name"] = [CLASSES[i] for i in val_preds]

    n_points = pd.read_parquet(RESULTS_DIR / "val_points.parquet").groupby(GROUP_KEY).size()
    n_points.name = "n_points"
    keys = keys.merge(n_points, on=GROUP_KEY, how="left")

    sparse = keys[keys["n_points"] <= POINT_THRESHOLD]

    counts = pd.crosstab(sparse["true_name"], sparse["pred_name"]).reindex(
        index=CLASSES, columns=CLASSES, fill_value=0
    )
    row_pct = counts.div(counts.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(row_pct.to_numpy(), cmap="Blues", vmin=0, vmax=100)

    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            count = counts.iat[i, j]
            pct = row_pct.iat[i, j]
            color = "white" if pct > 55 else "black"
            ax.text(j, i, f"{count}\n({pct:.1f}%)", ha="center", va="center", fontsize=8, color=color)

    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=30, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title(f"confusion matrix, val instances with ≤{POINT_THRESHOLD} points\n"
                 f"(n={len(sparse)} of {len(keys)} total val instances, {MODEL_EXPERIMENT})")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("% of row (true class) total")
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"confusion_{LABEL}.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")

    print(f"\nConfusion matrix, val instances with <= {POINT_THRESHOLD} points "
          f"(n={len(sparse)}), true (rows) x predicted (columns), {MODEL_EXPERIMENT}:\n")
    print(counts.to_string())

    summary_path = OUT_DIR / f"confusion_{LABEL}.csv"
    counts.to_csv(summary_path)
    print(f"\nSaved {summary_path}")
