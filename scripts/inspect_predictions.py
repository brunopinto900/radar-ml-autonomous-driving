"""Day 6 qualitative check: one full scene + closest camera frame (same layout
as the original dataloader.inspect_scene plot), with every tracked object's
box colored/labeled by the model's prediction vs. ground truth, instead of a
plain black box. Reuses the saved 20-epoch checkpoint and cached val features
rather than retraining or rebuilding anything.
"""
import numpy as np
import pandas as pd
import torch

from dataloader import RESULTS_DIR, axis_limits, load_scene, plot_scene, scene_image_path
from histogram_features import FEATURES, N_BINS
from train_mlp import CLASSES
from train_mlp_full import HIDDEN_DIM, PaperMLP

OUT_DIR = RESULTS_DIR / "predictions_camera"
CACHE_DIR = RESULTS_DIR / "mlp_feature_cache"
MODEL_PATH = RESULTS_DIR / "mlp_full_run" / "model_20epoch.pt"
SEED = 42


if __name__ == "__main__":
    X_val = np.load(CACHE_DIR / "val_X.npy")
    y_val = np.load(CACHE_DIR / "val_y.npy")
    keys_val = pd.read_parquet(CACHE_DIR / "val_keys.parquet")

    model = PaperMLP(in_dim=len(FEATURES) * N_BINS, hidden_dim=HIDDEN_DIM, n_classes=len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    with torch.no_grad():
        val_preds = model(torch.tensor(X_val)).argmax(1).numpy()

    keys_val = keys_val.copy()
    keys_val["true_name"] = [CLASSES[i] for i in y_val]
    keys_val["pred_name"] = [CLASSES[i] for i in val_preds]

    rng = np.random.default_rng(SEED)
    scenes = keys_val[["sequence_name", "timestamp"]].drop_duplicates().reset_index(drop=True)
    sequence_name, timestamp = scenes.iloc[rng.integers(len(scenes))]
    timestamp = int(timestamp)

    scene_keys = keys_val[(keys_val["sequence_name"] == sequence_name) & (keys_val["timestamp"] == timestamp)]
    predictions = {row["track_id"]: (row["true_name"], row["pred_name"]) for _, row in scene_keys.iterrows()}

    detections = load_scene(sequence_name, timestamp)
    xlim, ylim = axis_limits(detections)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{sequence_name}_{timestamp}.png"
    plot_scene(
        detections,
        title=f"{sequence_name} @ {timestamp} - predictions overlaid (green=correct, red=wrong)",
        path=path,
        image_path=scene_image_path(sequence_name, timestamp),
        xlim=xlim,
        ylim=ylim,
        predictions=predictions,
    )
    print(f"Saved {path}")
