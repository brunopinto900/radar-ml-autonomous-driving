"""Ad-hoc diagnostic: the car/large_vehicle confusion asymmetry, restricted to
instances predicted as either car or large_vehicle (i.e. the same 2325+... /
4290 and 18848 denominators used in the chat discussion, not each class's full
row total - see MLP_FINDINGS.md). Reuses the saved 20-epoch checkpoint and
cached val features, no retraining.
"""
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/ core modules

from dataloader import RESULTS_DIR  # noqa: E402
from histogram_features import FEATURES, N_BINS  # noqa: E402
from train_mlp import CLASSES  # noqa: E402
from train_mlp_full import HIDDEN_DIM, PaperMLP  # noqa: E402

RUN_DIR = RESULTS_DIR / "mlp_full_run"
CACHE_DIR = RESULTS_DIR / "mlp_feature_cache"
MODEL_PATH = RUN_DIR / "baseline_20epoch_h16" / "model.pt"
OUT_DIR = RUN_DIR / "large_vehicle_car_confusion"


if __name__ == "__main__":
    X_val = np.load(CACHE_DIR / "val_X.npy")
    y_val = np.load(CACHE_DIR / "val_y.npy")

    model = PaperMLP(in_dim=len(FEATURES) * N_BINS, hidden_dim=HIDDEN_DIM, n_classes=len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    with torch.no_grad():
        val_preds = model(torch.tensor(X_val)).argmax(1).numpy()

    car_idx, lv_idx = CLASSES.index("car"), CLASSES.index("large_vehicle")
    pred_is_pair = (val_preds == car_idx) | (val_preds == lv_idx)

    rows = []
    for true_idx, true_name in [(lv_idx, "large_vehicle"), (car_idx, "car")]:
        mask = (y_val == true_idx) & pred_is_pair
        total = mask.sum()
        correct = ((y_val == true_idx) & (val_preds == true_idx)).sum()
        other = ((y_val == true_idx) & (val_preds != true_idx) & pred_is_pair).sum()
        rows.append((true_name, total, correct / total * 100, other / total * 100))

    fig, ax = plt.subplots(figsize=(8, 3))
    y_pos = np.arange(len(rows))
    correct_pct = [r[2] for r in rows]
    other_pct = [r[3] for r in rows]
    labels = [f"true {r[0]}\n(n={r[1]})" for r in rows]

    ax.barh(y_pos, correct_pct, color="tab:green", label="classified correctly")
    ax.barh(y_pos, other_pct, left=correct_pct, color="tab:red", label="classified as the other class")
    for i, (c, o) in enumerate(zip(correct_pct, other_pct)):
        ax.text(c / 2, i, f"{c:.1f}%", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        ax.text(c + o / 2, i, f"{o:.1f}%", ha="center", va="center", color="white", fontsize=10, fontweight="bold")

    ax.set_yticks(y_pos, labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of instances predicted as either car or large_vehicle")
    ax.set_title("car vs. large_vehicle confusion, conditioned on true class (val, 20-epoch model)")
    ax.legend(loc="lower right", fontsize=8)
    ax.invert_yaxis()
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "car_large_vehicle_confusion.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
