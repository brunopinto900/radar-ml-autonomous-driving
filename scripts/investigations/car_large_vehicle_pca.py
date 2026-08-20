"""Ad-hoc diagnostic: PCA projection of the raw 80-dim feature vectors for
three mutually-exclusive groups - sparse car (<4pt, dropped by the min4pts
filter), dense car (>=4pt, the min4pts-surviving tail), and large_vehicle
(min4pts population) - to visually check what car_large_vehicle_feature_
overlap.py measured quantitatively (98.5% AUC separability even for the dense
tail). A visual companion to that script, not a replacement - PCA can hide
separation that exists in the full 80-dim space, see the caveat in
MLP_FINDINGS.md. Reuses cached features, no retraining.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

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
SEED = 42
N_SAMPLE = 2000  # per group, for a readable scatter

CAR_IDX = CLASSES.index("car")
LV_IDX = CLASSES.index("large_vehicle")

GROUP_COLORS = {
    "car, sparse (<4pt, dropped)": "tab:blue",
    "car, dense (>=4pt, kept)": "tab:cyan",
    "large_vehicle (min4pts)": "tab:orange",
}


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)

    X_train_full = np.load(CACHE_DIR / "train_X.npy")
    y_train_full = np.load(CACHE_DIR / "train_y.npy")
    train_keys = pd.read_parquet(CACHE_DIR / "train_keys.parquet")
    X_train_min4 = np.load(CACHE_DIR / "train_min4pts_X.npy")
    y_train_min4 = np.load(CACHE_DIR / "train_min4pts_y.npy")

    n_points = pd.read_parquet(RESULTS_DIR / "train_points.parquet").groupby(GROUP_KEY).size()
    n_points.name = "n_points"
    train_meta = train_keys.merge(n_points, on=GROUP_KEY, how="left")

    car_mask = y_train_full == CAR_IDX
    sparse_mask = car_mask & (train_meta["n_points"].to_numpy() < 4)
    car_sparse_vec = X_train_full[sparse_mask]

    car_dense_vec = X_train_min4[y_train_min4 == CAR_IDX]
    lv_vec = X_train_min4[y_train_min4 == LV_IDX]

    print(f"car, sparse (<4pt): n={len(car_sparse_vec)}")
    print(f"car, dense (>=4pt): n={len(car_dense_vec)}")
    print(f"large_vehicle (min4pts): n={len(lv_vec)}")

    def sample(vecs: np.ndarray) -> np.ndarray:
        if len(vecs) <= N_SAMPLE:
            return vecs
        idx = rng.choice(len(vecs), N_SAMPLE, replace=False)
        return vecs[idx]

    groups = {
        "car, sparse (<4pt, dropped)": sample(car_sparse_vec),
        "car, dense (>=4pt, kept)": sample(car_dense_vec),
        "large_vehicle (min4pts)": sample(lv_vec),
    }

    combined = np.concatenate(list(groups.values()))
    pca = PCA(n_components=2, random_state=SEED)
    pca.fit(combined)
    explained = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(8, 6.5))
    for name, vecs in groups.items():
        proj = pca.transform(vecs)
        ax.scatter(proj[:, 0], proj[:, 1], s=8, alpha=0.35, color=GROUP_COLORS[name],
                   label=f"{name} (n={len(vecs)})", edgecolors="none")

    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% var)")
    ax.set_title("PCA of raw 80-dim feature vectors: car (sparse vs. dense) vs. large_vehicle\n"
                 "(2D projection - see caveat: can hide separation visible in the full space)")
    ax.legend(fontsize=8, markerscale=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = RUN_DIR / "large_vehicle_car_confusion" / "car_large_vehicle_pca.png"
    fig.savefig(path, dpi=150)
    print(f"\nSaved {path}")
    print(f"PC1+PC2 explained variance: {explained.sum() * 100:.1f}%")
