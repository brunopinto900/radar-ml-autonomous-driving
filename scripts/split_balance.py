"""Verify the train/val/test split (built by make_split.py) didn't skew any class.

Checks whether each class's distribution across splits matches the expected
sequence-count ratio, or whether the (unstratified, see DESIGN_DECISIONS.md)
random split disproportionately dumped a class into one split.
"""
import matplotlib
import pandas as pd

matplotlib.use("Agg")  # headless: this script only ever saves plots, never shows them
import matplotlib.pyplot as plt  # noqa: E402

from dataloader import RESULTS_DIR  # noqa: E402

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
    splits_meta = pd.read_csv(RESULTS_DIR / "sequence_splits.csv")
    expected = splits_meta["split"].value_counts(normalize=True) * 100

    all_instances = pd.concat([load_instances(s) for s in ["train", "val", "test"]], ignore_index=True)
    pivot = pd.crosstab(all_instances["class_name"], all_instances["split"], normalize="index")[
        ["train", "val", "test"]
    ] * 100
    print(f"Per-class % of instances by split (expected: train {expected['train']:.1f}%, "
          f"val {expected['val']:.1f}%, test {expected['test']:.1f}%)")
    print(pivot.round(1))

    RESULTS_DIR.mkdir(exist_ok=True)
    balance_path = RESULTS_DIR / "split_balance.png"
    plot_split_balance(all_instances, expected, balance_path)
    print(f"Saved plot to {balance_path}")
