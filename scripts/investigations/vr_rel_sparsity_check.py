"""Diagnostic: is vr_rel's weak signal explained by instance point-sparsity?

vr_rel is median-centered per instance -> for a 1-point instance it is
identically 0 (median of one point is itself), and for a 2-3 point instance
it's barely informative either. Most instances in this dataset are point-sparse
(established earlier: filtering to n_points>=4 shrinks classes drastically).
This checks directly whether Doppler spread separates classes cleanly once you
condition on having enough points to actually measure a spread, i.e. whether
the weak aggregate result is a sparsity-dilution artifact rather than the
micro-Doppler physics being genuinely weak.
"""
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/ core modules

from dataloader import RESULTS_DIR
from histogram_features import GROUP_KEY, add_relative_vr

CLASSES = ["car", "large_vehicle", "pedestrian", "pedestrian_group", "two_wheeler"]

train_df = add_relative_vr(pd.read_parquet(RESULTS_DIR / "train_points.parquet"))

rows = []
for key, points in train_df.groupby(GROUP_KEY):
    n = len(points)
    spread = points["vr_rel"].std() if n > 1 else 0.0
    rng = points["vr_rel"].max() - points["vr_rel"].min() if n > 1 else 0.0
    rows.append((points["class_name"].iloc[0], n, spread, rng))

stats = pd.DataFrame(rows, columns=["class_name", "n_points", "vr_rel_std", "vr_rel_range"])

print("Overall n_points distribution:")
print(stats["n_points"].describe())
print(f"\nFraction of instances with n_points <= 3: {(stats['n_points'] <= 3).mean():.3f}")
print(f"Fraction of instances with n_points == 1: {(stats['n_points'] == 1).mean():.3f}")

print("\n--- Per-class mean Doppler spread (std of vr_rel), ALL instances ---")
print(stats.groupby("class_name")["vr_rel_std"].mean().reindex(CLASSES))

for min_pts in [1, 4, 8, 16]:
    sub = stats[stats["n_points"] >= min_pts]
    print(f"\n--- n_points >= {min_pts} (n={len(sub)} instances, "
          f"{len(sub) / len(stats):.1%} of data) ---")
    print(sub.groupby("class_name")["vr_rel_std"].mean().reindex(CLASSES))

print("\n--- Correlation between n_points and vr_rel_std (within class) ---")
for c in CLASSES:
    sub = stats[stats["class_name"] == c]
    print(f"{c:20s} corr(n_points, vr_rel_std) = {sub['n_points'].corr(sub['vr_rel_std']):.3f}  "
          f"(n={len(sub)})")
