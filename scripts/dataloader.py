"""Minimal RadarScenes loader: load one scene (single radar measurement) and plot its point cloud."""
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data/RadarScenes/RadarScenes/data"
RESULTS_DIR = PROJECT_ROOT / "results"

# label_id -> (name, color) per the RadarScenes readme
LABELS = {
    0: ("car", "tab:red"),
    1: ("large_vehicle", "tab:orange"),
    2: ("truck", "gold"),
    3: ("bus", "tab:brown"),
    4: ("train", "tab:pink"),
    5: ("bicycle", "tab:purple"),
    6: ("motorized_two_wheeler", "tab:cyan"),
    7: ("pedestrian", "tab:green"),
    8: ("pedestrian_group", "lime"),
    9: ("animal", "tab:olive"),
    10: ("other_dynamic", "magenta"),
    11: ("static", "tab:gray"),
}


def load_scene(sequence_name: str, timestamp: int | None = None) -> np.ndarray:
    """Load radar detections for one scene (one radar measurement) of a sequence.

    If timestamp is None, the first scene of the sequence is used.
    Returns a structured numpy array (rows = detections, see radar_data.h5 dtype).
    """
    seq_dir = DATA_ROOT / sequence_name
    with open(seq_dir / "scenes.json") as f:
        scenes = json.load(f)["scenes"]

    if timestamp is None:
        timestamp = next(iter(scenes))
    scene = scenes[str(timestamp)]
    start, end = scene["radar_indices"]

    with h5py.File(seq_dir / "radar_data.h5", "r") as f:
        detections = f["radar_data"][start:end]

    return detections


def scene_image_path(sequence_name: str, timestamp: int) -> Path:
    """Path to the camera image closest in time to a given scene."""
    seq_dir = DATA_ROOT / sequence_name
    with open(seq_dir / "scenes.json") as f:
        scenes = json.load(f)["scenes"]
    image_name = scenes[str(timestamp)]["image_name"]
    return seq_dir / "camera" / image_name


def list_track_ids(detections: np.ndarray) -> np.ndarray:
    """Unique, non-empty track_ids (dynamic objects) present in a scene."""
    ids = np.unique(detections["track_id"])
    return ids[ids != b""]


def object_detections(detections: np.ndarray, track_id: bytes) -> np.ndarray:
    """Detections belonging to one tracked object within a scene."""
    return detections[detections["track_id"] == track_id]


# attributes relevant for training: RCS, Doppler, range, azimuth, position, sensor
OBJECT_ATTRS = ["sensor_id", "range_sc", "azimuth_sc", "rcs", "vr", "vr_compensated", "x_cc", "y_cc"]


def object_instance(detections: np.ndarray, track_id: bytes) -> dict:
    """One tracked object's points + attributes + label, as a plain dict.

    This is the Day-3 building block: {points, attributes, label} per labeled
    object instance. Plotting functions are just views on top of this.
    """
    points = object_detections(detections, track_id)
    label_id = int(points["label_id"][0])
    return {
        "track_id": track_id,
        "label_id": label_id,
        "label_name": LABELS[label_id][0],
        "points": points,
    }


def largest_object(detections: np.ndarray) -> bytes | None:
    """track_id with the most points in a scene, or None if there are no tracked objects."""
    track_ids = list_track_ids(detections)
    if len(track_ids) == 0:
        return None
    return track_ids[np.argmax([len(object_detections(detections, t)) for t in track_ids])]


def axis_limits(detections: np.ndarray, pad: float = 2.0) -> tuple[tuple[float, float], tuple[float, float]]:
    """(xlim, ylim) covering a scene's full extent, for sharing across plots."""
    xlim = (detections["y_cc"].min() - pad, detections["y_cc"].max() + pad)
    ylim = (detections["x_cc"].min() - pad, detections["x_cc"].max() + pad)
    return xlim, ylim


def plot_scene(
    detections: np.ndarray,
    title: str,
    path: str | Path,
    image_path: Path | None = None,
    show_object_boxes: bool = True,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    if image_path is not None:
        fig, (ax_img, ax) = plt.subplots(1, 2, figsize=(14, 7))
        ax_img.imshow(plt.imread(image_path))
        ax_img.axis("off")
        ax_img.set_title("closest camera frame")
    else:
        fig, ax = plt.subplots(figsize=(7, 7))

    for label_id, (name, color) in LABELS.items():
        mask = detections["label_id"] == label_id
        if not mask.any():
            continue
        ax.scatter(
            detections["y_cc"][mask],
            detections["x_cc"][mask],
            c=color,
            label=name,
            s=25,
            edgecolors="k",
            linewidths=0.3,
        )

    if show_object_boxes:
        pad = 0.5
        for i, track_id in enumerate(list_track_ids(detections)):
            obj = object_detections(detections, track_id)
            y_min, y_max = obj["y_cc"].min() - pad, obj["y_cc"].max() + pad
            x_min, x_max = obj["x_cc"].min() - pad, obj["x_cc"].max() + pad
            ax.add_patch(plt.Rectangle(
                (y_min, x_min),
                y_max - y_min,
                x_max - x_min,
                fill=False,
                edgecolor="black",
                linewidth=1.2,
                label="tracked object" if i == 0 else None,
            ))

    ax.scatter(0, 0, c="black", marker="^", s=100, label="sensor")
    ax.set_xlabel("y_cc [m] (left)")
    ax.set_ylabel("x_cc [m] (forward)")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.invert_xaxis()  # y_cc positive = left, flip so left is on the left of the plot
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def plot_object_attributes(
    obj: np.ndarray,
    title: str,
    path: str | Path,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Visualize one object's own points colored by RCS and by Doppler (vr_compensated)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, field, cmap, cbar_label in [
        (axes[0], "rcs", "viridis", "RCS [dBsm]"),
        (axes[1], "vr_compensated", "coolwarm", "radial velocity, ego-compensated [m/s]"),
    ]:
        sc = ax.scatter(obj["y_cc"], obj["x_cc"], c=obj[field], cmap=cmap, s=25, edgecolors="k", linewidths=0.3)
        fig.colorbar(sc, ax=ax, label=cbar_label)
        ax.set_xlabel("y_cc [m] (left)")
        ax.set_ylabel("x_cc [m] (forward)")
        ax.set_title(field)
        ax.set_aspect("equal", adjustable="box")
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.invert_xaxis()
        ax.grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def inspect_scene(sequence_name: str, timestamp: int, track_id: bytes | None = None) -> dict | None:
    """Load one scene, plot it (+ camera frame), and plot one object's RCS/Doppler.

    track_id=None auto-picks the object with the most points. Returns the
    object_instance dict, or None if the scene has no tracked (dynamic) objects.
    Writes scene_plot.png and (if an object was found) object_attributes.png.
    """
    detections = load_scene(sequence_name, timestamp)
    print(f"Loaded {len(detections)} detections from {sequence_name} @ {timestamp}")

    RESULTS_DIR.mkdir(exist_ok=True)
    scene_plot_path = RESULTS_DIR / "scene_plot.png"
    object_attrs_path = RESULTS_DIR / "object_attributes.png"

    xlim, ylim = axis_limits(detections)
    plot_scene(
        detections,
        title=f"{sequence_name} - one radar scan",
        path=scene_plot_path,
        image_path=scene_image_path(sequence_name, timestamp),
        xlim=xlim,
        ylim=ylim,
    )
    print(f"Saved plot to {scene_plot_path}")

    if track_id is None:
        track_id = largest_object(detections)
    if track_id is None:
        print("No tracked (dynamic) objects in this scene.")
        return None

    instance = object_instance(detections, track_id)
    obj, label_name = instance["points"], instance["label_name"]
    print(f"Object {track_id.decode()}: {len(obj)} points, label={label_name}")
    obj_df = pd.DataFrame(obj[OBJECT_ATTRS].tolist(), columns=OBJECT_ATTRS)
    print(obj_df.round(3).to_string())

    plot_object_attributes(
        obj,
        title=f"{label_name} ({track_id.decode()[:8]}) - {len(obj)} points",
        path=object_attrs_path,
        xlim=xlim,
        ylim=ylim,
    )
    print(f"Saved plot to {object_attrs_path}")

    return instance


if __name__ == "__main__":
    inspect_scene("sequence_1", 156947099838)