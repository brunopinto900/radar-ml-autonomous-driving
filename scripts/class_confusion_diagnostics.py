"""Ad-hoc diagnostics for a specific class's confusion: for every true
TRUE_CLASS val instance, violin-plot point count / mean range / mean
extent_rel / bbox diagonal per predicted-class group, with mean+median+n
labeled directly on each plot, plus the same numbers printed/saved as a plain
table. Violin, not scatter: an overlapping-scatter version of these plots
visually hid a real ~3-4x difference in central tendency between "correctly
predicted large_vehicle" and "misclassified as car" that the actual numbers
caught - a violin shows each group's density shape directly instead of
requiring the eye to judge density from raw point overlap, and the table
means/medians are printed alongside so nothing needs to be read off the plot
by eye either. Edit TRUE_CLASS to switch which class this runs for. Reuses
the saved 20-epoch checkpoint and cached val features, no retraining.
"""
import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dataloader import RESULTS_DIR  # noqa: E402
from histogram_features import FEATURES, GROUP_KEY, N_BINS, add_relative_xy  # noqa: E402
from train_mlp import CLASSES  # noqa: E402
from train_mlp_full import HIDDEN_DIM, PaperMLP  # noqa: E402

OUT_DIR = RESULTS_DIR / "mlp_full_run"
CACHE_DIR = RESULTS_DIR / "mlp_feature_cache"
MODEL_PATH = OUT_DIR / "model_20epoch.pt"

TRUE_CLASS = "car"

METRICS = {
    "n_points": "point count (per instance)",
    "mean_range": "mean range_sc [m] (per instance)",
    "mean_extent": "mean extent_rel [m] (per instance, distance from own centroid)",
    "bbox_diagonal": "bounding-box diagonal [m] (per instance)",
}


def plot_metric(keys: pd.DataFrame, classes: list[str], true_class: str, metric: str, xlabel: str, path) -> None:
    positions, data, colors = [], [], []
    for i, name in enumerate(classes):
        rows = keys.loc[keys["pred_name"] == name, metric].dropna()
        if len(rows) < 2:
            continue
        positions.append(i)
        data.append(rows.to_numpy())
        colors.append("tab:green" if name == true_class else "tab:red")

    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    parts = ax.violinplot(data, positions=positions, orientation="horizontal", widths=0.8,
                           showmeans=True, showmedians=True, showextrema=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.6)
    parts["cmeans"].set_color("black")
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linestyle("--")

    for i, values in zip(positions, data):
        mean_v, median_v = values.mean(), np.median(values)
        ax.text(values.max() + 0.02 * (keys[metric].max() - keys[metric].min()), i,
                f"mean={mean_v:.2f}  med={median_v:.2f}  n={len(values)}",
                ha="left", va="center", fontsize=7, color="black")

    ax.set_yticks(range(len(classes)), classes)
    ax.set_ylim(-0.7, len(classes) - 0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("model's prediction")
    ax.set_title(f"true {true_class} instances (n={len(keys)}): {metric} vs. prediction (val, 20-epoch model)")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")


if __name__ == "__main__":
    X_val = np.load(CACHE_DIR / "val_X.npy")
    y_val = np.load(CACHE_DIR / "val_y.npy")
    keys_val = pd.read_parquet(CACHE_DIR / "val_keys.parquet")

    model = PaperMLP(in_dim=len(FEATURES) * N_BINS, hidden_dim=HIDDEN_DIM, n_classes=len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    with torch.no_grad():
        val_preds = model(torch.tensor(X_val)).argmax(1).numpy()

    val_df = add_relative_xy(pd.read_parquet(RESULTS_DIR / "val_points.parquet"))
    val_df["extent_rel"] = np.hypot(val_df["x_rel"], val_df["y_rel"])

    per_instance = val_df.groupby(GROUP_KEY).agg(
        n_points=("x_cc", "size"),
        mean_range=("range_sc", "mean"),
        mean_extent=("extent_rel", "mean"),
        x_span=("x_cc", lambda s: s.max() - s.min()),
        y_span=("y_cc", lambda s: s.max() - s.min()),
    ).reset_index()
    per_instance["bbox_diagonal"] = np.hypot(per_instance["x_span"], per_instance["y_span"])

    true_idx = CLASSES.index(TRUE_CLASS)
    true_mask = y_val == true_idx

    keys = keys_val[true_mask].reset_index(drop=True)
    keys["pred_name"] = [CLASSES[i] for i in val_preds[true_mask]]
    keys["correct"] = keys["pred_name"] == TRUE_CLASS
    keys = keys.merge(per_instance, on=GROUP_KEY, how="left")

    for metric, xlabel in METRICS.items():
        plot_metric(keys, CLASSES, TRUE_CLASS, metric, xlabel, OUT_DIR / f"{TRUE_CLASS}_{metric}_vs_prediction.png")

    agg = {"n": ("pred_name", "size")}
    for metric in METRICS:
        agg[f"{metric}_mean"] = (metric, "mean")
        agg[f"{metric}_median"] = (metric, "median")
    summary = keys.groupby("pred_name").agg(**agg).round(2).reindex(CLASSES).dropna(how="all")

    print(f"\nTrue {TRUE_CLASS} instances (n={true_mask.sum()}), grouped by predicted class:\n")
    print(summary.to_string())

    summary_path = OUT_DIR / f"{TRUE_CLASS}_confusion_summary.csv"
    summary.to_csv(summary_path)
    print(f"\nSaved {summary_path}")
