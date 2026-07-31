"""class imbalance check on the training points table built by build_points_table.py.

Counts instances (one physical object's appearance in one scene), not raw points -
an object with many points (e.g. a large_vehicle) shouldn't outweigh one with few
(e.g. a car often has just 1-3 detections) in the class balance.
"""
import matplotlib
import pandas as pd

matplotlib.use("Agg")  # headless: this script only ever saves plots, never shows them
import matplotlib.pyplot as plt  # noqa: E402

from dataloader import GROUP_COLORS, RESULTS_DIR, sensor_label  # noqa: E402

SPLIT_COLORS = {"train": "tab:blue", "val": "tab:orange", "test": "tab:green"}


def load_instances(split: str) -> pd.DataFrame:
    """One row per object instance in a split, with its class_name."""
    df = pd.read_parquet(RESULTS_DIR / f"{split}_points.parquet")
    instances = df.groupby(["sequence_name", "timestamp", "track_id"])["class_name"].first().reset_index()
    instances["split"] = split
    return instances


def plot_split_balance(all_instances: pd.DataFrame, expected: pd.Series, path) -> None:
    """For each class, what % of its instances landed in each split, vs the expected
    (sequence-count) ratio - dashed lines. Flags classes the random split skewed."""
    pivot = pd.crosstab(all_instances["class_name"], all_instances["split"], normalize="index")[
        ["train", "val", "test"]
    ] * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(pivot))
    width = 0.25
    for i, split in enumerate(["train", "val", "test"]):
        offsets = [xi + (i - 1) * width for xi in x]
        ax.bar(offsets, pivot[split], width=width, color=SPLIT_COLORS[split], edgecolor="k", label=split)
        ax.axhline(expected[split], color=SPLIT_COLORS[split], linestyle="--", linewidth=1, alpha=0.7)

    ax.set_xticks(list(x))
    ax.set_xticklabels(pivot.index, rotation=45, ha="right")
    ax.set_ylabel("% of that class's instances in this split")
    ax.set_title("Per-class split balance vs expected (dashed = sequence-count ratio)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    table_path = RESULTS_DIR / "train_points.parquet"
    df = pd.read_parquet(table_path)
    print(f"Loaded {len(df)} points from {table_path}")

    instances = df.groupby(["sequence_name", "timestamp", "track_id"])["class_name"].first()
    print(f"{len(instances)} object instances")

    counts = instances.value_counts()
    print(counts)

    sensor_ids = df["sensor_id"].unique()
    sensor_str = sensor_label(int(sensor_ids[0])) if len(sensor_ids) == 1 else f"{len(sensor_ids)} sensors"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts.index, counts.values, color=[GROUP_COLORS[name] for name in counts.index], edgecolor="k")
    ax.set_yscale("log")
    ax.set_ylabel("instance count (log scale)")
    ax.set_title(f"Object instance counts ({df['sequence_name'].nunique()} sequences, {sensor_str})")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()

    RESULTS_DIR.mkdir(exist_ok=True)
    plot_path = RESULTS_DIR / "class_counts.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Saved plot to {plot_path}")

    # Split balance: does each class's distribution across train/val/test match the
    # sequence-count ratio, or did the (unstratified, see DESIGN_DECISIONS.md) random
    # split skew any class into one split disproportionately?
    splits_meta = pd.read_csv(RESULTS_DIR / "sequence_splits.csv")
    expected = splits_meta["split"].value_counts(normalize=True) * 100

    all_instances = pd.concat([load_instances(s) for s in ["train", "val", "test"]], ignore_index=True)
    pivot = pd.crosstab(all_instances["class_name"], all_instances["split"], normalize="index")[
        ["train", "val", "test"]
    ] * 100
    print()
    print(f"Per-class % of instances by split (expected: train {expected['train']:.1f}%, "
          f"val {expected['val']:.1f}%, test {expected['test']:.1f}%)")
    print(pivot.round(1))

    balance_path = RESULTS_DIR / "split_balance.png"
    plot_split_balance(all_instances, expected, balance_path)
    print(f"Saved plot to {balance_path}")
