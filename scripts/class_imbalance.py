"""Class imbalance check on the points table (parquet, built by build_points_table.py):
counts instances per class and saves a bar chart."""
import matplotlib.pyplot as plt
import pandas as pd

from dataloader import FINAL_CLASS_COLORS, LABELS, RESULTS_DIR, sensor_label

NAME_TO_COLOR = {name: color for name, color in LABELS.values()}


def plot_class_imbalance(table_path=None, class_groups: dict[str, str] | None = None, tag: str = "class_counts"):
    """Load the points table, count object instances per class, and plot/save a
    log-scale bar chart. Returns (summary, fig), where summary is a count/pct DataFrame.

    `class_groups` (default None): maps raw `label_name` to a merged class first (e.g.
    dataloader.MLP_CLASS_GROUPS, truck/train/bus folded into large_vehicle), dropping instances
    mapped to None, instead of counting the 12 raw labels as-is. Colors come from
    dataloader.FINAL_CLASS_COLORS in that case, since merged class names (e.g. two_wheeler)
    aren't in the raw LABELS palette. `tag` sets the saved filename
    (results/class_imbalance/<tag>.png), pass a different one alongside class_groups to avoid
    overwriting the raw-taxonomy plot."""
    if table_path is None:
        table_path = RESULTS_DIR / "data" / "points_table.parquet"
    df = pd.read_parquet(table_path)
    print(f"Loaded {len(df)} points from {table_path}")

    label_col = "label_name"
    if class_groups is not None:
        df = df.copy()
        df["group"] = df["label_name"].map(class_groups)
        df = df.loc[df["group"].notna()]
        label_col = "group"

    instances = df.groupby(["sequence_name", "timestamp", "track_id"])[label_col].first()
    print(f"{len(instances)} object instances")

    counts = instances.value_counts()
    pct = (counts / counts.sum() * 100).round(1)
    summary = pd.DataFrame({"count": counts, "pct": pct})
    print(summary.to_string())

    sensor_ids = df["sensor_id"].unique()
    sensor_str = sensor_label(int(sensor_ids[0])) if len(sensor_ids) == 1 else f"{len(sensor_ids)} sensors"
    name_to_color = FINAL_CLASS_COLORS if class_groups is not None else NAME_TO_COLOR

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values, color=[name_to_color[name] for name in counts.index], edgecolor="k")
    ax.set_yscale("log")
    ax.set_ylabel("instance count (log scale)")
    ax.set_title(f"Object instance counts ({df['sequence_name'].nunique()} sequences, {sensor_str})")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for bar, p in zip(bars, pct):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{p:.1f}%", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()

    plot_path = RESULTS_DIR / "class_imbalance" / f"{tag}.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    print(f"Saved plot to {plot_path}")

    return summary, fig


if __name__ == "__main__":
    plot_class_imbalance()