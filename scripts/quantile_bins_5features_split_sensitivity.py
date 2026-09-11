"""One-off noise-floor check for the quantile_bins_5features MLP variant (MLP_CONFIG.json),
resolving model_comparison.md's open question: does the 5-feature (x_rel/y_rel/vr_compensated/
range_sc/rcs) gain over baseline (0.708 vs 0.686, single split) survive the same 6-fold
split-sensitivity check that already showed section 5's near-identical range_sc addition
(MLP_Decisions_and_Findings.md) to be noise, not a real gain?

Mirrors split_sensitivity.py's loop exactly (same 6 candidate splits via select_best_split),
but calls run_training/evaluate_val_metrics with quantile_bins_5features's own config
(features, extra_features, bin_range="quantile", epochs=200) instead of baseline's, since
split_sensitivity.run_split_sensitivity doesn't forward bin_range/epochs. Not folded into that
function: this is a one-off diagnostic for one variant, not a reusable feature-set noise-floor
tool like split_sensitivity.py itself is.

Cached to results/mlp/split_search_quantile_bins_5features/fold_<n>/."""
import pandas as pd

from feature_distributions import HISTOGRAM_FEATURES, MLP_CLASSES
from mlp_classifier import MLP_DIR, evaluate_val_metrics, run_training
from sequence_split import select_best_split

FEATURES = ["x_rel", "y_rel", "vr_compensated", "range_sc", "rcs"]
EXTRA_FEATURES = ["doppler_spread"]
SPLIT_SEARCH_DIR = MLP_DIR / "split_search_quantile_bins_5features"


def run_split_sensitivity(df: pd.DataFrame, n_seeds: int = 10, base_random_state: int = 0) -> pd.DataFrame:
    candidates = select_best_split(
        df, classes=MLP_CLASSES, features=HISTOGRAM_FEATURES, n_seeds=n_seeds, base_random_state=base_random_state
    ).drop_duplicates("fold").sort_values("fold")

    rows = []
    for _, row in candidates.iterrows():
        fold = row["fold"]
        splits = {"train": row["train_sequences"], "val": row["val_sequences"], "test": row["test_sequences"]}
        output_dir = SPLIT_SEARCH_DIR / f"fold_{fold}"
        print(f"=== fold {fold} (max_ks={row['max_ks']:.4f}) ===")
        run_training(
            df, output_dir=output_dir, splits=splits,
            features=FEATURES, extra_features=EXTRA_FEATURES, bin_range="quantile", epochs=200,
        )
        metrics_df, _, _ = evaluate_val_metrics(
            df, output_dir=output_dir, splits=splits,
            features=FEATURES, extra_features=EXTRA_FEATURES, bin_range="quantile",
        )
        macro_f1 = metrics_df["f1"].mean()
        print(f"fold {fold} macro F1: {macro_f1:.4f}")
        rows.append({"fold": fold, "max_ks": row["max_ks"], "macro_f1": macro_f1})

    summary = pd.DataFrame(rows)
    print()
    print(f"macro F1 range: {summary['macro_f1'].min():.3f} - {summary['macro_f1'].max():.3f}, std: {summary['macro_f1'].std():.3f}")
    print(summary.to_string(index=False))
    SPLIT_SEARCH_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = SPLIT_SEARCH_DIR / "split_sensitivity_summary.csv"
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
