import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/ core modules

from dataloader import RESULTS_DIR  # noqa: E402
from histogram_features import GROUP_KEY, add_relative_vr, add_relative_xy  # noqa: E402

OUT_DIR = RESULTS_DIR / "mlp_full_run" / "large_vehicle_car_confusion"
GROUPS = ["car", "truck", "bus", "large_vehicle"]
COLORS = {"car": "tab:blue", "truck": "tab:orange", "bus": "tab:green",
          "large_vehicle": "tab:red"}
SPARSE_MAX_POINTS = 3

POINT_FEATURES = {
    "vr_rel": (-3.0, 3.0),
    "x_rel": (-10.0, 10.0),
    "y_rel": (-10.0, 10.0),
    "rcs": (-20.0, 30.0),
}

df = add_relative_vr(add_relative_xy(pd.read_parquet(RESULTS_DIR / "train_points.parquet")))
df = df[df["label_name"].isin(GROUPS)]

n_points = df.groupby(GROUP_KEY).size().rename("n_points")
df = df.join(n_points, on=GROUP_KEY)

extent_rows = []
for key, pts in df.groupby(GROUP_KEY):
    n = len(pts)
    extent = (np.hypot(pts["x_rel"].max() - pts["x_rel"].min(),
                        pts["y_rel"].max() - pts["y_rel"].min()) if n > 1 else 0.0)
    extent_rows.append((pts["label_name"].iloc[0], n, extent))
extent_df = pd.DataFrame(extent_rows, columns=["label_name", "n_points", "extent"])

row_labels = list(POINT_FEATURES) + ["extent"]
stats_rows = []
fig, axes = plt.subplots(5, 2, figsize=(14, 21))

for row, feat in enumerate(row_labels):
    for col, sparse in enumerate([False, True]):
        ax = axes[row, col]
        regime = f"sparse (n_points<={SPARSE_MAX_POINTS})" if sparse else "overall"
        if feat == "extent":
            data = extent_df[extent_df["n_points"] <= SPARSE_MAX_POINTS] if sparse else extent_df
            lo, hi = 0.0, 20.0
        else:
            data = df[df["n_points"] <= SPARSE_MAX_POINTS] if sparse else df
            lo, hi = POINT_FEATURES[feat]
        bins = np.linspace(lo, hi, 41)
        for g in GROUPS:
            vals = data.loc[data["label_name"] == g, feat]
            mean, median, std, n = vals.mean(), vals.median(), vals.std(), len(vals)
            stats_rows.append((feat, regime, g, n, mean, median, std))
            ax.hist(vals, bins=bins, density=True, histtype="step", linewidth=1.8,
                     color=COLORS[g], label=f"{g}  μ={mean:.2f}  med={median:.2f}")
            ax.axvline(mean, color=COLORS[g], linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_title(f"{feat} - {regime}")
        ax.set_xlabel(feat)
        ax.set_ylabel("density")
        ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(OUT_DIR / "subtype_histograms.png", dpi=150)
print(f"Saved {OUT_DIR / 'subtype_histograms.png'}")

stats_df = pd.DataFrame(stats_rows, columns=["feature", "regime", "label_name", "n", "mean",
                                              "median", "std"])
stats_df.to_csv(OUT_DIR / "subtype_histograms_stats.csv", index=False)
print(f"Saved {OUT_DIR / 'subtype_histograms_stats.csv'}")
with pd.option_context("display.width", 120, "display.max_rows", None):
    print(stats_df.to_string(index=False))
