"""Ad-hoc diagnostic, run before deciding to add n_points as an explicit
feature (MLP_FINDINGS.md "Next steps"): (1) feature-feature Pearson
correlation across the 80 existing histogram dims + a candidate n_points
81st dim, to quantify how redundant n_points already is with the existing
features (each feature block's 16 bins sum to exactly n_points by
construction, so a very high correlation is expected, not a bug); (2)
point-biserial correlation between each feature and each one-hot class
label, to see which features matter for which class on their own
(univariate only - doesn't say what the actual network will do with them
jointly). Reuses cached train features, no retraining.
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
from histogram_features import FEATURES, GROUP_KEY, N_BINS  # noqa: E402
from train_mlp import CLASSES  # noqa: E402

CACHE_DIR = RESULTS_DIR / "mlp_feature_cache"
OUT_DIR = RESULTS_DIR / "mlp_full_run" / "feature_correlation"

BLOCK_NAMES = FEATURES + ["n_points"]
BLOCK_BOUNDARIES = [i * N_BINS for i in range(len(FEATURES) + 1)] + [len(FEATURES) * N_BINS + 1]
FEATURE_NAMES = [f"{feat}_bin{i}" for feat in FEATURES for i in range(N_BINS)] + ["n_points"]


def block_ticks() -> tuple[list[float], list[str]]:
    centers = [(BLOCK_BOUNDARIES[i] + BLOCK_BOUNDARIES[i + 1]) / 2 - 0.5 for i in range(len(BLOCK_NAMES))]
    return centers, BLOCK_NAMES


if __name__ == "__main__":
    X = np.load(CACHE_DIR / "train_X.npy")
    y = np.load(CACHE_DIR / "train_y.npy")
    keys = pd.read_parquet(CACHE_DIR / "train_keys.parquet")

    n_points = pd.read_parquet(RESULTS_DIR / "train_points.parquet").groupby(GROUP_KEY).size()
    n_points.name = "n_points"
    n_points_aligned = keys.merge(n_points, on=GROUP_KEY, how="left")["n_points"].to_numpy()

    X_aug = np.hstack([X, n_points_aligned.reshape(-1, 1)]).astype(np.float64)

    # --- 1. feature-feature correlation ---
    with np.errstate(invalid="ignore"):
        corr = np.corrcoef(X_aug, rowvar=False)
    n_nan = np.isnan(corr).sum()
    if n_nan:
        print(f"Note: {n_nan} NaN entries in the correlation matrix (zero-variance bins - "
              f"e.g. a histogram bin with the same count, often 0, for every instance) - set to 0.")
        corr = np.nan_to_num(corr)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    for b in BLOCK_BOUNDARIES[1:-1]:
        ax.axhline(b - 0.5, color="black", linewidth=0.6)
        ax.axvline(b - 0.5, color="black", linewidth=0.6)
    ticks, labels = block_ticks()
    ax.set_xticks(ticks, labels, rotation=45, ha="right")
    ax.set_yticks(ticks, labels)
    ax.set_title("Feature-feature correlation (80 histogram dims + candidate n_points, train set)")
    fig.colorbar(im, ax=ax, label="Pearson correlation")
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "feature_feature_correlation.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")

    # focused readout: n_points vs. each feature block's own point-count (sum of its 16 bins)
    print("\nn_points correlation with each existing feature block's row-sum (= that block's own "
          "point count):")
    np_idx = len(FEATURE_NAMES) - 1
    for i, feat in enumerate(FEATURES):
        block = X[:, i * N_BINS:(i + 1) * N_BINS]
        block_sum = block.sum(axis=1)
        r = np.corrcoef(block_sum, n_points_aligned)[0, 1]
        print(f"  {feat:16s} block-sum vs. n_points: r = {r:.4f}")

    # --- 2. point-biserial correlation: each feature vs. each one-hot class ---
    class_corr = np.zeros((len(CLASSES), X_aug.shape[1]))
    for ci, _ in enumerate(CLASSES):
        y_bin = (y == ci).astype(np.float64)
        for fi in range(X_aug.shape[1]):
            col = X_aug[:, fi]
            if col.std() == 0:
                class_corr[ci, fi] = 0.0
            else:
                class_corr[ci, fi] = np.corrcoef(col, y_bin)[0, 1]

    fig, ax = plt.subplots(figsize=(11, 3.2))
    im = ax.imshow(class_corr, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    for b in BLOCK_BOUNDARIES[1:-1]:
        ax.axvline(b - 0.5, color="black", linewidth=0.6)
    ticks, labels = block_ticks()
    ax.set_xticks(ticks, labels, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_title("Point-biserial correlation on RAW bin counts (still count-confounded - see the\n"
                 "share-based version below for the shape-only signal)")
    fig.colorbar(im, ax=ax, label="correlation", shrink=0.8)
    fig.tight_layout()
    path = OUT_DIR / "feature_class_point_biserial_raw.png"
    fig.savefig(path, dpi=150)
    print(f"\nSaved {path}")

    # --- 3. magnitude (n_points) vs. shape (each bin's share of the instance's own n_points),
    # kept separate so a block's correlation with class isn't just re-measuring point count -
    # see the vr_compensated bug this was built to fix (chat discussion, not a file this repo
    # tracks a summary of).
    print("\nn_points (magnitude) point-biserial correlation per class - one number, since every "
          "block's sum is ~equally redundant with it (feature_feature_correlation.png):")
    for ci, name in enumerate(CLASSES):
        print(f"  {name:18s} r = {class_corr[ci, np_idx]:+.4f}")

    with np.errstate(invalid="ignore", divide="ignore"):
        X_share = np.where(n_points_aligned[:, None] > 0, X / n_points_aligned[:, None], 0)

    shape_corr = np.zeros((len(CLASSES), X_share.shape[1]))
    for ci in range(len(CLASSES)):
        y_bin = (y == ci).astype(np.float64)
        for fi in range(X_share.shape[1]):
            col = X_share[:, fi]
            shape_corr[ci, fi] = 0.0 if col.std() == 0 else np.corrcoef(col, y_bin)[0, 1]

    fig, ax = plt.subplots(figsize=(11, 3.2))
    im = ax.imshow(shape_corr, cmap="RdBu_r", vmin=-0.3, vmax=0.3, aspect="auto")
    for b in BLOCK_BOUNDARIES[1:-1]:
        ax.axvline(b - 0.5, color="black", linewidth=0.6)
    ax.set_xticks(ticks[:-1], labels[:-1], rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_title("Point-biserial correlation on SHARE of n_points (magnitude/count removed - the\n"
                 "genuine within-instance shape signal, independent of how many points there are)")
    fig.colorbar(im, ax=ax, label="correlation", shrink=0.8)
    fig.tight_layout()
    path = OUT_DIR / "feature_class_point_biserial_shape.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")

    # per-block summary: mean |shape correlation| across that block's 16 bins, per class - a
    # single number per block/class that only reflects shape, not magnitude
    summary = pd.DataFrame(index=CLASSES, columns=FEATURES, dtype=float)
    for i, feat in enumerate(FEATURES):
        block_corr = shape_corr[:, i * N_BINS:(i + 1) * N_BINS]
        summary[feat] = np.abs(block_corr).mean(axis=1)
    summary["n_points (magnitude)"] = class_corr[:, np_idx]
    summary_path = OUT_DIR / "class_correlation_shape_vs_magnitude.csv"
    summary.round(4).to_csv(summary_path)
    print(f"\nPer-class mean |shape correlation| per block (magnitude removed), plus n_points' own "
          f"magnitude correlation for comparison - saved to {summary_path}:")
    print(summary.round(3).to_string())
