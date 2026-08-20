"""Is the large_vehicle/car confusion driven equally by truck/bus, or mostly one of them?

large_vehicle is a merge of 4 raw RadarScenes labels (DESIGN_DECISIONS.md decision 1):
large_vehicle (raw), truck, bus, train. train_points.parquet keeps the pre-merge label in
`label_name`, so we can break large_vehicle back down by original subtype and compare each
one to car directly - both overall and restricted to the sparse (n_points<=3) regime that
section 4 of MLP_FINDINGS.md already showed is what actually drives large_vehicle->car
misclassifications (mean point count 2.50 for misclassified large_vehicle instances).
"""
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/ core modules

from dataloader import RESULTS_DIR
from histogram_features import GROUP_KEY, add_relative_vr, add_relative_xy

SPARSE_MAX_POINTS = 3  # matches section 4's confused-instance point count (~2.5 mean)

train_df = add_relative_vr(add_relative_xy(pd.read_parquet(RESULTS_DIR / "train_points.parquet")))

rows = []
for key, pts in train_df.groupby(GROUP_KEY):
    n = len(pts)
    # extent: diagonal of the instance's own x_rel/y_rel bounding box (object-centered,
    # same x_rel/y_rel used as MLP features) - a direct, unitless-in-meters size proxy.
    # Undefined (0) for a 1-point instance, same convention as vr_rel_sparsity_check.py.
    if n > 1:
        extent = np.hypot(pts["x_rel"].max() - pts["x_rel"].min(),
                           pts["y_rel"].max() - pts["y_rel"].min())
        vr_rel_std = pts["vr_rel"].std()
    else:
        extent = 0.0
        vr_rel_std = 0.0
    rows.append((pts["class_name"].iloc[0], pts["label_name"].iloc[0], n,
                 pts["rcs"].mean(), extent, vr_rel_std))

stats = pd.DataFrame(rows, columns=["class_name", "label_name", "n_points", "rcs_mean",
                                     "extent", "vr_rel_std"])

print("Instance counts by raw label_name (within car + large_vehicle):")
print(stats[stats["class_name"].isin(["car", "large_vehicle"])]["label_name"].value_counts())

print("\n--- Overall per-instance means, car vs. truck vs. bus (all n_points) ---")
print(stats[stats["label_name"].isin(["car", "truck", "bus"])]
      .groupby("label_name")[["n_points", "rcs_mean", "extent", "vr_rel_std"]].mean())

sparse = stats[stats["n_points"] <= SPARSE_MAX_POINTS]
print(f"\n--- Sparse-only (n_points <= {SPARSE_MAX_POINTS}) per-instance means ---")
print(sparse[sparse["label_name"].isin(["car", "truck", "bus"])]
      .groupby("label_name")[["n_points", "rcs_mean", "extent", "vr_rel_std"]].mean())

print("\nInstance counts in the sparse regime:")
print(sparse[sparse["label_name"].isin(["car", "truck", "bus"])]["label_name"].value_counts())
