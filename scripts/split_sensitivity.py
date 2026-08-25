"""Establishes the noise floor: how much the MLP's macro F1 depends on split choice alone,
vs. which split has the best-matching train/val feature distributions.

- Generates the distinct valid splits at fixed 70/15/15 proportions, scored by train/val
  distributional match (sequence_split.select_best_split).
- Trains mlp_classifier's baseline config on each (only the split changes) and compares macro
  F1. Result: 0.651-0.734 across 6 candidates, uncorrelated with distributional match
  (MLP_Decisions_and_Findings.md split selection finding).
- `features`/`extra_features`/`normalize` (default: baseline's) measure the noise floor for a
  different feature set, since a differently-encoded feature set isn't guaranteed to share
  baseline's noise floor. `output_subdir` keeps results separate per feature set.

Cached to results/mlp/<output_subdir>/fold_<n>/, same caching as any mlp_classifier.py run."""
import pandas as pd

from feature_distributions import HISTOGRAM_FEATURES, MLP_CLASSES
from mlp_classifier import MLP_DIR, evaluate_val_metrics, run_training
from sequence_split import select_best_split

SPLIT_SEARCH_DIR = MLP_DIR / "split_search"


def run_split_sensitivity(
    df: pd.DataFrame,
    n_seeds: int = 10,
    base_random_state: int = 0,
    features: list[str] | None = None,
    extra_features: list[str] | None = None,
    normalize: bool | None = None,
    output_subdir: str = "split_search",
) -> pd.DataFrame:
    """Generates the distinct valid val splits (see select_best_split) and trains mlp_classifier's
    config on each, baseline's by default, or a different feature set via `features`/
    `extra_features`/`normalize`. Returns one row per fold: distributional match score (max_ks)
    and macro F1."""
    candidates = select_best_split(
        df, classes=MLP_CLASSES, features=HISTOGRAM_FEATURES, n_seeds=n_seeds, base_random_state=base_random_state
    ).drop_duplicates("fold").sort_values("fold")

    feature_kwargs = {}
    if features is not None:
        feature_kwargs["features"] = features
    if extra_features is not None:
        feature_kwargs["extra_features"] = extra_features
    if normalize is not None:
        feature_kwargs["normalize"] = normalize

    split_search_dir = MLP_DIR / output_subdir
    rows = []
    for _, row in candidates.iterrows():
        fold = row["fold"]
        splits = {"train": row["train_sequences"], "val": row["val_sequences"], "test": row["test_sequences"]}
        output_dir = split_search_dir / f"fold_{fold}"
        print(f"=== fold {fold} (max_ks={row['max_ks']:.4f}) ===")
        run_training(df, output_dir=output_dir, splits=splits, **feature_kwargs)
        metrics_df, _, _ = evaluate_val_metrics(df, output_dir=output_dir, splits=splits, **feature_kwargs)
        macro_f1 = metrics_df["f1"].mean()
        print(f"fold {fold} macro F1: {macro_f1:.4f}")
        rows.append({"fold": fold, "max_ks": row["max_ks"], "macro_f1": macro_f1})

    summary = pd.DataFrame(rows)
    print()
    print(f"macro F1 range: {summary['macro_f1'].min():.3f} - {summary['macro_f1'].max():.3f}, std: {summary['macro_f1'].std():.3f}")
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    from build_points_table import build_and_save_points_table
    from mlp_classifier import apply_mlp_class_groups
    from taxonomy_separability import add_relative_features

    df = build_and_save_points_table()
    df = add_relative_features(df)
    df = apply_mlp_class_groups(df)

    run_split_sensitivity(df)
