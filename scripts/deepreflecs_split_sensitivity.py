"""DeepReflecs counterpart to split_sensitivity.py: how much DeepReflecs' macro F1 depends on
split choice alone, using the exact same 6 candidate splits scripts/split_sensitivity.py
measured the baseline MLP's noise floor on (0.651-0.734 macro F1 range,
MLP_Decisions_and_Findings.md split selection finding) - select_best_split's fold identity
only depends on classes/df/seeds, not on which model or feature set consumes the split, so
calling it with the same classes/n_seeds/base_random_state reproduces those same 6 folds
bit-for-bit, letting the two architectures' noise floors be compared directly rather than
each measured against a different set of splits.

Cached to results/deepreflecs/split_search/fold_<n>/, same caching as any
deepreflecs_classifier.py run (run_training/evaluate_val_metrics)."""
import pandas as pd

from deepreflecs_classifier import DEEPREFLECS_DIR, evaluate_val_metrics, run_training
from feature_distributions import HISTOGRAM_FEATURES, MLP_CLASSES
from sequence_split import select_best_split

SPLIT_SEARCH_DIR = DEEPREFLECS_DIR / "split_search"


def run_split_sensitivity(
    df: pd.DataFrame, n_seeds: int = 10, base_random_state: int = 0, output_subdir: str = "split_search",
) -> pd.DataFrame:
    """Generates the same 6 distinct val splits split_sensitivity.py used (see select_best_split)
    and trains DeepReflecs' standard config on each, only the split changes. Returns one row per
    fold: distributional match score (max_ks) and macro F1."""
    candidates = select_best_split(
        df, classes=MLP_CLASSES, features=HISTOGRAM_FEATURES, n_seeds=n_seeds, base_random_state=base_random_state
    ).drop_duplicates("fold").sort_values("fold")

    split_search_dir = DEEPREFLECS_DIR / output_subdir
    rows = []
    for _, row in candidates.iterrows():
        fold = row["fold"]
        splits = {"train": row["train_sequences"], "val": row["val_sequences"], "test": row["test_sequences"]}
        output_dir = split_search_dir / f"fold_{fold}"
        print(f"=== fold {fold} (max_ks={row['max_ks']:.4f}) ===")
        run_training(df, output_dir=output_dir, splits=splits)
        metrics_df, _, _ = evaluate_val_metrics(df, output_dir=output_dir, splits=splits)
        macro_f1 = metrics_df["f1"].mean()
        print(f"fold {fold} macro F1: {macro_f1:.4f}")
        rows.append({"fold": fold, "max_ks": row["max_ks"], "macro_f1": macro_f1})

    summary = pd.DataFrame(rows)
    print()
    print(f"macro F1 range: {summary['macro_f1'].min():.3f} - {summary['macro_f1'].max():.3f}, std: {summary['macro_f1'].std():.3f}")
    print(summary.to_string(index=False))

    split_search_dir.mkdir(parents=True, exist_ok=True)
    summary_path = split_search_dir / "split_sensitivity_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")
    return summary


if __name__ == "__main__":
    from build_points_table import build_and_save_points_table
    from mlp_classifier import apply_mlp_class_groups
    from taxonomy_separability import add_relative_features

    df = build_and_save_points_table()
    df = add_relative_features(df)
    df = apply_mlp_class_groups(df)

    run_split_sensitivity(df)
