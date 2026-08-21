"""class imbalance check on the points table (a parquet file) built by build_points_table.py.
Counts number instances per class and outputs an image
"""
import matplotlib.pyplot as plt
import pandas as pd

from dataloader import LABELS, RESULTS_DIR, sensor_label

NAME_TO_COLOR = {name: color for name, color in LABELS.values()}


def plot_class_imbalance(table_path=None):
    """Load the points table, count object instances per class, and plot/save a
    log-scale bar chart. Returns (summary, fig), where summary is a count/pct DataFrame."""
    if table_path is None:
        table_path = RESULTS_DIR / "data" / "points_table.parquet"
    df = pd.read_parquet(table_path)
    print(f"Loaded {len(df)} points from {table_path}")

    instances = df.groupby(["sequence_name", "timestamp", "track_id"])["label_name"].first()
    print(f"{len(instances)} object instances")

    counts = instances.value_counts()
    pct = (counts / counts.sum() * 100).round(1)
    summary = pd.DataFrame({"count": counts, "pct": pct})
    print(summary.to_string())

    sensor_ids = df["sensor_id"].unique()
    sensor_str = sensor_label(int(sensor_ids[0])) if len(sensor_ids) == 1 else f"{len(sensor_ids)} sensors"

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values, color=[NAME_TO_COLOR[name] for name in counts.index], edgecolor="k")
    ax.set_yscale("log")
    ax.set_ylabel("instance count (log scale)")
    ax.set_title(f"Object instance counts ({df['sequence_name'].nunique()} sequences, {sensor_str})")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for bar, p in zip(bars, pct):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{p:.1f}%", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()

    plot_path = RESULTS_DIR / "class_imbalance" / "class_counts.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    print(f"Saved plot to {plot_path}")

    return summary, fig


if __name__ == "__main__":
    plot_class_imbalance()