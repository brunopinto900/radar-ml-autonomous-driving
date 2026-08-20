"""Same check as pedestrian_two_wheeler_velocity_check.py, but on RCS instead of
velocity: does the baseline model's pedestrian-misclassified-as-two_wheeler group
have an RCS profile closer to "person" or closer to "person + bike" (more metal/
surface area, higher RCS)? A cross-check against the fast-pedestrian hypothesis
using a signal velocity has nothing to do with.

No retraining - loads the existing baseline checkpoint and the cached baseline val
features (results/mlp_feature_cache), reruns inference once.
"""
import numpy as np
import pandas as pd
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/ core modules

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

    # Per-instance mean RCS.
    rcs_rows = []
    for key, pts in val_df.groupby(GROUP_KEY):
        rcs_rows.append((*key, pts["rcs"].mean()))
    rcs_df = pd.DataFrame(rcs_rows, columns=GROUP_KEY + ["mean_rcs"])

    keys_val = keys_val.reset_index(drop=True)
    keys_val["mean_rcs"] = (rcs_df.set_index(GROUP_KEY)["mean_rcs"]
                             .reindex(pd.MultiIndex.from_frame(keys_val[GROUP_KEY]))
                             .to_numpy())
    keys_val["pred"] = val_preds
    keys_val["true"] = y_val

    PED_IDX = CLASSES.index("pedestrian")
    TW_IDX = CLASSES.index("two_wheeler")

    ped_mask = keys_val["true"] == PED_IDX
    correct_mask = ped_mask & (keys_val["pred"] == PED_IDX)
    misclf_mask = ped_mask & (keys_val["pred"] == TW_IDX)
    tw_correct_mask = (keys_val["true"] == TW_IDX) & (keys_val["pred"] == TW_IDX)

    correct_rcs = keys_val.loc[correct_mask, "mean_rcs"]
    misclf_rcs = keys_val.loc[misclf_mask, "mean_rcs"]
    tw_rcs = keys_val.loc[tw_correct_mask, "mean_rcs"]

    print(f"True pedestrian instances in val: {ped_mask.sum()}")
    print(f"  correctly classified as pedestrian: {correct_mask.sum()}")
    print(f"  misclassified as two_wheeler:        {misclf_mask.sum()}")
    print(f"True two_wheeler, correctly classified: {tw_correct_mask.sum()}")
    print()
    print("mean_rcs per instance [dBsm]:")
    print(f"  pedestrian, correct        -> mean={correct_rcs.mean():.3f}  median={correct_rcs.median():.3f}  "
          f"std={correct_rcs.std():.3f}")
    print(f"  pedestrian, misclf->2wheel -> mean={misclf_rcs.mean():.3f}  median={misclf_rcs.median():.3f}  "
          f"std={misclf_rcs.std():.3f}")
    print(f"  two_wheeler, correct       -> mean={tw_rcs.mean():.3f}  median={tw_rcs.median():.3f}  "
          f"std={tw_rcs.std():.3f}")
