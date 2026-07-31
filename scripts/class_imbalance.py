"""class imbalance check on the points table built by build_points_table.py.

Counts instances (one physical object's appearance in one scene), not raw points -
an object with many points (e.g. a large_vehicle) shouldn't outweigh one with few
(e.g. a car often has just 1-3 detections) in the class balance.
"""
import matplotlib.pyplot as plt
import pandas as pd

from dataloader import LABELS, RESULTS_DIR, sensor_label

NAME_TO_COLOR = {name: color for name, color in LABELS.values()}


if __name__ == "__main__":
    table_path = RESULTS_DIR / "points_table.parquet"
    df = pd.read_parquet(table_path)
    print(f"Loaded {len(df)} points from {table_path}")

    instances = df.groupby(["sequence_name", "timestamp", "track_id"])["label_name"].first()
    print(f"{len(instances)} object instances")

    counts = instances.value_counts()
    print(counts)

    sensor_ids = df["sensor_id"].unique()
    sensor_str = sensor_label(int(sensor_ids[0])) if len(sensor_ids) == 1 else f"{len(sensor_ids)} sensors"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts.index, counts.values, color=[NAME_TO_COLOR[name] for name in counts.index], edgecolor="k")
    ax.set_yscale("log")
    ax.set_ylabel("instance count (log scale)")
    ax.set_title(f"Object instance counts ({df['sequence_name'].nunique()} sequences, {sensor_str})")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()

    RESULTS_DIR.mkdir(exist_ok=True)
    plot_path = RESULTS_DIR / "class_counts.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Saved plot to {plot_path}")
