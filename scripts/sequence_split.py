"""Fixed train/val/test split of *sequences* (not instances, not scans) - a sequence's points
never span more than one split, so background/weather/vehicle correlation within a sequence
can't leak across splits (same leakage risk StratifiedGroupKFold's CV folds were already
guarding against). Unlike the CV folds used inside separability_probe.run_probe, this split is
computed once, cached to disk, and meant to stay fixed: val is for comparing candidates freely
(bin count, and later model/architecture choices), test is checked once at the end, per
Design_Decisions.md decision 3's revisit note.
"""
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from dataloader import RESULTS_DIR
from feature_distributions import FINAL_CLASSES

INSTANCE_COLS = ["sequence_name", "timestamp", "track_id"]
SPLIT_CACHE = RESULTS_DIR / "sequence_split.json"


def load_split() -> dict[str, list[str]]:
    """Load the cached split without needing a df - just {"train": [...], "val": [...], "test": [...]}.
    Raises if split_sequences() hasn't been run yet."""
    if not SPLIT_CACHE.exists():
        raise FileNotFoundError(f"{SPLIT_CACHE} doesn't exist yet - run split_sequences() first")
    return json.loads(SPLIT_CACHE.read_text())["splits"]


def split_sequences(
    df: pd.DataFrame,
    classes: list[str] = FINAL_CLASSES,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    random_state: int = 0,
) -> dict[str, list[str]]:
    """Splits sequences into train/val/test via two chained StratifiedGroupKFold calls (grouped
    by sequence_name, stratified by class at the instance level) - first carve out test, then
    carve val out of what's left. Fractions are approximate: fold counts are discrete (round(1/frac)
    folds), so actual splits land close to but not exactly at val_frac/test_frac. Cached to
    results/sequence_split.json and reused on repeat calls with the same classes/fractions/seed -
    this split is meant to stay fixed once created, not get recomputed per run."""
    cache_key = {"classes": sorted(classes), "val_frac": val_frac, "test_frac": test_frac, "random_state": random_state}
    if SPLIT_CACHE.exists():
        cached = json.loads(SPLIT_CACHE.read_text())
        if cached.get("key") == cache_key:
            print(f"{SPLIT_CACHE} already matches this config, reusing fixed split")
            splits = cached["splits"]
            _print_summary(df, splits, classes)
            return splits
        print(f"{SPLIT_CACHE} exists but doesn't match this config, rebuilding")

    instances = df.loc[df["group"].isin(classes)].drop_duplicates(INSTANCE_COLS)
    y = instances["group"].to_numpy(dtype=str)
    groups = instances["sequence_name"].to_numpy(dtype=str)
    X_dummy = np.zeros(len(y))

    n_splits_test = round(1 / test_frac)
    test_splitter = StratifiedGroupKFold(n_splits=n_splits_test, shuffle=True, random_state=random_state)
    trainval_idx, test_idx = next(test_splitter.split(X_dummy, y, groups))
    test_sequences = set(groups[test_idx])

    trainval_mask = ~np.isin(groups, list(test_sequences))
    y_tv, groups_tv = y[trainval_mask], groups[trainval_mask]
    val_frac_of_remainder = val_frac / (1 - test_frac)
    n_splits_val = round(1 / val_frac_of_remainder)
    val_splitter = StratifiedGroupKFold(n_splits=n_splits_val, shuffle=True, random_state=random_state)
    train_idx, val_idx = next(val_splitter.split(np.zeros(len(y_tv)), y_tv, groups_tv))
    val_sequences = set(groups_tv[val_idx])
    train_sequences = set(groups_tv[train_idx])

    assert not (train_sequences & val_sequences | train_sequences & test_sequences | val_sequences & test_sequences), (
        "a sequence ended up in more than one split - this should be impossible, something's wrong"
    )
    for name, seqs in [("train", train_sequences), ("val", val_sequences), ("test", test_sequences)]:
        split_y = set(y[np.isin(groups, list(seqs))])
        assert split_y == set(classes), f"{name} split is missing classes: {set(classes) - split_y}"

    splits = {
        "train": sorted(train_sequences),
        "val": sorted(val_sequences),
        "test": sorted(test_sequences),
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    SPLIT_CACHE.write_text(json.dumps({"key": cache_key, "splits": splits}, indent=2))
    print(f"Saved {SPLIT_CACHE}")
    _print_summary(df, splits, classes)
    return splits


def _print_summary(df: pd.DataFrame, splits: dict[str, list[str]], classes: list[str]) -> None:
    instances = df.loc[df["group"].isin(classes)].drop_duplicates(INSTANCE_COLS)
    total = len(instances)

    rows = []
    for name, seqs in splits.items():
        sub = instances.loc[instances["sequence_name"].isin(seqs)]
        row = {"split": name, "n_sequences": len(seqs), "n_instances": len(sub), "pct_of_total": 100 * len(sub) / total}
        counts = sub["group"].value_counts()
        row.update({f"pct_{cls}": 100 * counts.get(cls, 0) / len(sub) for cls in classes})
        rows.append(row)

    print(pd.DataFrame(rows).set_index("split").round(1).to_string())


if __name__ == "__main__":
    from build_points_table import build_and_save_points_table
    from feature_distributions import apply_class_groups
    from taxonomy_separability import add_relative_features

    df = build_and_save_points_table()
    df = add_relative_features(df)
    df = apply_class_groups(df)

    split_sequences(df)
