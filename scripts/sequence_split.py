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
SPLIT_CACHE = RESULTS_DIR / "data" / "sequence_split.json"


def load_split() -> dict[str, list[str]]:
    """Load the cached split without needing a df - just {"train": [...], "val": [...], "test": [...]}.
    Raises if split_sequences() hasn't been run yet."""
    if not SPLIT_CACHE.exists():
        raise FileNotFoundError(f"{SPLIT_CACHE} doesn't exist yet - run split_sequences() first")
    return json.loads(SPLIT_CACHE.read_text())["splits"]


def _build_split(
    df: pd.DataFrame, classes: list[str], val_frac: float, test_frac: float, random_state: int
) -> dict[str, list[str]]:
    """Core splitting logic, no caching/printing - two chained StratifiedGroupKFold calls (grouped
    by sequence_name, stratified by class at the instance level): first carve out test, then carve
    val out of what's left. Fractions are approximate: fold counts are discrete (round(1/frac)
    folds), so actual splits land close to but not exactly at val_frac/test_frac."""
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

    return {
        "train": sorted(train_sequences),
        "val": sorted(val_sequences),
        "test": sorted(test_sequences),
    }


def split_sequences(
    df: pd.DataFrame,
    classes: list[str] = FINAL_CLASSES,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    random_state: int = 0,
) -> dict[str, list[str]]:
    """Builds (via _build_split) or loads the fixed train/val/test split, cached to
    results/sequence_split.json and reused on repeat calls with the same classes/fractions/seed -
    this split is meant to stay fixed once created, not get recomputed per run. See
    select_best_split to choose random_state deliberately instead of defaulting to 0."""
    cache_key = {"classes": sorted(classes), "val_frac": val_frac, "test_frac": test_frac, "random_state": random_state}
    if SPLIT_CACHE.exists():
        cached = json.loads(SPLIT_CACHE.read_text())
        if cached.get("key") == cache_key:
            print(f"{SPLIT_CACHE} already matches this config, reusing fixed split")
            splits = cached["splits"]
            _print_summary(df, splits, classes)
            return splits
        print(f"{SPLIT_CACHE} exists but doesn't match this config, rebuilding")

    splits = _build_split(df, classes, val_frac, test_frac, random_state)
    SPLIT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_CACHE.write_text(json.dumps({"key": cache_key, "splits": splits}, indent=2))
    print(f"Saved {SPLIT_CACHE}")
    _print_summary(df, splits, classes)
    return splits


def select_best_split(
    df: pd.DataFrame,
    classes: list[str],
    features: list[str],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    n_seeds: int = 10,
    base_random_state: int = 0,
) -> pd.DataFrame:
    """Searches for the val carve (out of train+val, at the fixed val_frac) whose per-class
    feature distributions match train's most closely - NOT which val gives the best model score,
    that would be circular. test is carved once with base_random_state and held fixed throughout,
    this only re-explores val, since test is meant to be touched once, not re-picked.

    random_state alone doesn't give real diversity here: StratifiedGroupKFold's greedy assignment
    turned out to be largely deterministic for this data's group-size distribution, so every seed
    landed on the same first fold when only next() was called once (see chat/commit history).
    Enumerating every fold of a single split() call is what actually yields distinct val
    candidates, since each fold is a genuinely different chunk of the same partition; multiple
    seeds are layered on top for extra variety.

    Scores each candidate's train vs val distributional match per class per feature via a
    two-sample KS statistic (lower = more alike). Does NOT touch SPLIT_CACHE or pick a winner
    itself, purely diagnostic - build the winning candidate's exact train/val/test lists from the
    returned row and write them to SPLIT_CACHE deliberately to commit. Returns one row per
    candidate, sorted by worst-case (max) KS ascending with mean_ks as tiebreaker, worst-case
    first because one badly-mismatched class/feature matters more than a good average elsewhere."""
    from feature_distributions import feature_values
    from scipy.stats import ks_2samp

    instances = df.loc[df["group"].isin(classes)].drop_duplicates(INSTANCE_COLS)
    y = instances["group"].to_numpy(dtype=str)
    groups = instances["sequence_name"].to_numpy(dtype=str)
    X_dummy = np.zeros(len(y))

    n_splits_test = round(1 / test_frac)
    test_splitter = StratifiedGroupKFold(n_splits=n_splits_test, shuffle=True, random_state=base_random_state)
    _, test_idx = next(test_splitter.split(X_dummy, y, groups))
    test_sequences = sorted(set(groups[test_idx]))

    trainval_mask = ~np.isin(groups, test_sequences)
    y_tv, groups_tv = y[trainval_mask], groups[trainval_mask]
    val_frac_of_remainder = val_frac / (1 - test_frac)
    n_splits_val = round(1 / val_frac_of_remainder)

    rows = []
    for seed in range(base_random_state, base_random_state + n_seeds):
        val_splitter = StratifiedGroupKFold(n_splits=n_splits_val, shuffle=True, random_state=seed)
        for fold, (train_idx, val_idx) in enumerate(val_splitter.split(np.zeros(len(y_tv)), y_tv, groups_tv)):
            val_sequences = sorted(set(groups_tv[val_idx]))
            train_sequences = sorted(set(groups_tv[train_idx]))

            train_df = df.loc[df["sequence_name"].isin(train_sequences)]
            val_df = df.loc[df["sequence_name"].isin(val_sequences)]
            train_instances = instances.loc[instances["sequence_name"].isin(train_sequences)]
            val_instances = instances.loc[instances["sequence_name"].isin(val_sequences)]

            ks_scores = {}
            for cls in classes:
                for feature in features:
                    train_vals = feature_values(train_df.loc[train_df["group"] == cls], train_instances.loc[train_instances["group"] == cls], feature)
                    val_vals = feature_values(val_df.loc[val_df["group"] == cls], val_instances.loc[val_instances["group"] == cls], feature)
                    ks_scores[f"{cls}_{feature}"] = ks_2samp(train_vals, val_vals).statistic

            rows.append({
                "random_state": seed, "fold": fold, "n_val_sequences": len(val_sequences),
                "mean_ks": np.mean(list(ks_scores.values())), "max_ks": max(ks_scores.values()),
                "train_sequences": train_sequences, "val_sequences": val_sequences, "test_sequences": test_sequences,
                **ks_scores,
            })

    summary = pd.DataFrame(rows).sort_values(["max_ks", "mean_ks"]).reset_index(drop=True)
    print_cols = ["random_state", "fold", "n_val_sequences", "mean_ks", "max_ks"]
    print(summary[print_cols].round(4).to_string(index=False))
    return summary


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
