"""Why does the regrouped model (MLP_FINDINGS.md section 10 - car=old car+truck+
large_vehicle(raw), large_vehicle=bus-only) still predict 44% of true large_vehicle
instances as car? Same methodology as section 4's original car_large_vehicle deep dive
(scripts/class_confusion_diagnostics.py): split true large_vehicle val instances into
correct vs. misclassified-as-car, compare n_points, spatial extent, and Doppler spread
(vr_rel std) between the two groups.

Loads the already-trained, converged checkpoint
(results/mlp_full_run/car_large_vehicle_regroup_1000epoch/model.pt) rather than retraining.
"""
import numpy as np
import pandas as pd
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/ core modules

from dataloader import RESULTS_DIR  # noqa: E402
from histogram_features import GROUP_KEY, add_relative_vr, add_relative_xy  # noqa: E402
from train_mlp import CLASSES, load_or_build_dataset  # noqa: E402
from train_mlp_full import HIDDEN_DIM, PaperMLP  # noqa: E402
from train_mlp_regroup import regrouped_labels  # noqa: E402

CKPT_DIR = RESULTS_DIR / "mlp_full_run" / "car_large_vehicle_regroup_1000epoch"

if __name__ == "__main__":
    val_df = pd.read_parquet(RESULTS_DIR / "val_points.parquet")
    X_val, _, keys_val = load_or_build_dataset(val_df, {}, CLASSES, "val")
    y_val = regrouped_labels(val_df, keys_val, CLASSES)

    model = PaperMLP(in_dim=X_val.shape[1], hidden_dim=HIDDEN_DIM, n_classes=len(CLASSES))
    model.load_state_dict(torch.load(CKPT_DIR / "model.pt"))
    model.eval()
    with torch.no_grad():
        val_preds = model(torch.tensor(X_val)).argmax(1).numpy()

    LV_IDX = CLASSES.index("large_vehicle")
    CAR_IDX = CLASSES.index("car")

    lv_mask = y_val == LV_IDX
    correct_mask = lv_mask & (val_preds == LV_IDX)
    wrong_as_car_mask = lv_mask & (val_preds == CAR_IDX)
    print(f"true large_vehicle: {lv_mask.sum()}  correct: {correct_mask.sum()}  "
          f"wrong-as-car: {wrong_as_car_mask.sum()}")

    val_df = add_relative_vr(add_relative_xy(val_df))
    n_points = val_df.groupby(GROUP_KEY).size().rename("n_points")

    rows = []
    for key, pts in val_df.groupby(GROUP_KEY):
        n = len(pts)
        extent = (np.hypot(pts["x_rel"].max() - pts["x_rel"].min(),
                            pts["y_rel"].max() - pts["y_rel"].min()) if n > 1 else 0.0)
        vr_rel_std = pts["vr_rel"].std() if n > 1 else 0.0
        rows.append((*key, n, extent, vr_rel_std))
    inst_df = pd.DataFrame(rows, columns=GROUP_KEY + ["n_points", "extent", "vr_rel_std"])

    keys_val = keys_val.reset_index(drop=True)
    keys_val["group"] = np.where(correct_mask, "correct",
                                  np.where(wrong_as_car_mask, "wrong_as_car", "other"))
    merged = keys_val.merge(inst_df, on=GROUP_KEY, how="left")

    summary = merged[merged["group"] != "other"].groupby("group")[
        ["n_points", "extent", "vr_rel_std"]].agg(["mean", "median", "count"])
    print("\n--- true large_vehicle, correct vs. misclassified-as-car ---")
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(summary)
