"""Build the sequence-level train/val/test split. See DESIGN_DECISIONS.md.

Unit of assignment is the whole sequence (never individual scenes/instances),
to avoid leaking the same tracked object into two splits and to keep sequences
usable as contiguous time series for future temporal work.
"""
import json

import pandas as pd
from sklearn.model_selection import train_test_split

from dataloader import DATA_ROOT, RESULTS_DIR

RANDOM_SEED = 42
INTERNAL_VAL_FRACTION = 0.15  # of the official "train" sequences


def official_categories() -> dict[str, str]:
    with open(DATA_ROOT / "sequences.json") as f:
        sequences = json.load(f)["sequences"]
    return {name: info["category"] for name, info in sequences.items()}


def build_split() -> pd.DataFrame:
    categories = official_categories()
    official_train = sorted(name for name, cat in categories.items() if cat == "train")
    official_val = sorted(name for name, cat in categories.items() if cat == "validation")

    train_names, val_names = train_test_split(
        official_train, test_size=INTERNAL_VAL_FRACTION, random_state=RANDOM_SEED
    )

    rows = (
        [(name, "train") for name in train_names]
        + [(name, "val") for name in val_names]
        + [(name, "test") for name in official_val]
    )
    return pd.DataFrame(rows, columns=["sequence_name", "split"]).sort_values("sequence_name").reset_index(drop=True)


if __name__ == "__main__":
    df = build_split()
    print(df["split"].value_counts())

    RESULTS_DIR.mkdir(exist_ok=True)
    split_path = RESULTS_DIR / "sequence_splits.csv"
    df.to_csv(split_path, index=False)
    print(f"Saved sequence split to {split_path}")
