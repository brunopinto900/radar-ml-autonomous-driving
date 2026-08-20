"""Checks whether pedestrian instances the baseline model (results/mlp_full_run/
baseline_20epoch_h16) misclassifies as two_wheeler are disproportionately the
fast-moving ones - i.e. whether misclassification concentrates at the high-velocity
tail of the pedestrian population, which would support a gait/speed coverage-gap
explanation (RadarScenes pedestrian instances mostly walking-pace) rather than an
even-across-the-board error pattern.

No retraining - loads the existing baseline checkpoint and the cached baseline val
features (results/mlp_feature_cache), reruns inference once.
"""
import numpy as np
import pandas as pd
import torch

from dataloader import RESULTS_DIR
from histogram_features import GROUP_KEY
from train_mlp import CLASSES, load_or_build_dataset
from train_mlp_full import PaperMLP

CKPT_DIR = RESULTS_DIR / "mlp_full_run" / "baseline_20epoch_h16"

if __name__ == "__main__":
    val_df = pd.read_parquet(RESULTS_DIR / "val_points.parquet")
    X_val, y_val, keys_val = load_or_build_dataset(val_df, {}, CLASSES, "val")

    model = PaperMLP(in_dim=X_val.shape[1], hidden_dim=16, n_classes=len(CLASSES))
    model.load_state_dict(torch.load(CKPT_DIR / "model.pt"))
    model.eval()
    with torch.no_grad():
        val_preds = model(torch.tensor(X_val)).argmax(1).numpy()

    # Per-instance mean ground-relative (ego-motion-compensated) radial velocity.
    vel_rows = []
    for key, pts in val_df.groupby(GROUP_KEY):
        vel_rows.append((*key, pts["vr_compensated"].mean()))
    vel_df = pd.DataFrame(vel_rows, columns=GROUP_KEY + ["mean_vr_compensated"])

    keys_val = keys_val.reset_index(drop=True)
    keys_val["mean_vr_compensated"] = np.abs(vel_df.set_index(GROUP_KEY)["mean_vr_compensated"]
                                              .reindex(pd.MultiIndex.from_frame(keys_val[GROUP_KEY]))
                                              .to_numpy())
    keys_val["pred"] = val_preds
    keys_val["true"] = y_val

    PED_IDX = CLASSES.index("pedestrian")
    TW_IDX = CLASSES.index("two_wheeler")

    ped_mask = keys_val["true"] == PED_IDX
    correct_mask = ped_mask & (keys_val["pred"] == PED_IDX)
    misclf_mask = ped_mask & (keys_val["pred"] == TW_IDX)

    correct_vel = keys_val.loc[correct_mask, "mean_vr_compensated"]
    misclf_vel = keys_val.loc[misclf_mask, "mean_vr_compensated"]

    print(f"True pedestrian instances in val: {ped_mask.sum()}")
    print(f"  correctly classified as pedestrian: {correct_mask.sum()}")
    print(f"  misclassified as two_wheeler:        {misclf_mask.sum()}")
    print()
    print("abs(mean_vr_compensated) per instance [m/s]:")
    print(f"  correct        -> mean={correct_vel.mean():.3f}  median={correct_vel.median():.3f}  "
          f"std={correct_vel.std():.3f}")
    print(f"  misclf->2wheel -> mean={misclf_vel.mean():.3f}  median={misclf_vel.median():.3f}  "
          f"std={misclf_vel.std():.3f}")
