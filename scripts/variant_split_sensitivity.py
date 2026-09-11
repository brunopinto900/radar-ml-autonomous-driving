"""Generic noise-floor check for any MLP_CONFIG.json variant, not just baseline
(split_sensitivity.py) or one specific feature set (the now-superseded
quantile_bins_5features_split_sensitivity.py). Resolves model_comparison.md's open question for
the whole 5-feature encoding group at once: histogram_5features, stat_descriptors_5features,
maxad_stats_5features, and quantile_bins_5features all reduce the identical x_rel/y_rel/
vr_compensated/range_sc/rcs feature set, so running each through the same 6 candidate splits
(select_best_split, same call as split_sensitivity.py/deepreflecs_split_sensitivity.py) checks
whether any of the apparent gain over baseline (itself confounded with range_sc, never
isolated before this) survives split-sensitivity, not just single-split noise.

Cached to results/mlp/split_search_<variant>/fold_<n>/, same caching as any mlp_classifier.py
run, reusing MLP_CONFIG.json/mlp_variants.py as the single source of truth for each variant's
classes/features/feature_stats/bin_range/run_kwargs rather than duplicating them per script."""
import sys

import pandas as pd

from feature_distributions import HISTOGRAM_FEATURES
from mlp_classifier import MLP_DIR, evaluate_val_metrics, run_training
from mlp_variants import MLP_VARIANTS, build_variant_df
from sequence_split import select_best_split


def run_split_sensitivity_for_variant(
    raw_df: pd.DataFrame, variant: str, n_seeds: int = 10, base_random_state: int = 0,
) -> pd.DataFrame:
    config = MLP_VARIANTS[variant]
    df = build_variant_df(raw_df, variant)
    feature_kwargs = {
        k: config[k]
        for k in ("features", "extra_features", "normalize", "feature_stats", "bin_range", "standardize_extra")
        if k in config
    }
    arch_kwargs = {
        k: config["run_kwargs"][k] for k in ("hidden_dim", "n_hidden_layers", "dropout", "batch_norm")
        if k in config["run_kwargs"]
    }
    other_run_kwargs = {k: v for k, v in config["run_kwargs"].items() if k not in arch_kwargs}

    candidates = select_best_split(
        df, classes=config["classes"], features=HISTOGRAM_FEATURES, n_seeds=n_seeds,
        base_random_state=base_random_state,
    ).drop_duplicates("fold").sort_values("fold")

    split_search_dir = MLP_DIR / f"split_search_{variant}"
    rows = []
    for _, row in candidates.iterrows():
        fold = row["fold"]
        splits = {"train": row["train_sequences"], "val": row["val_sequences"], "test": row["test_sequences"]}
        output_dir = split_search_dir / f"fold_{fold}"
        print(f"=== {variant} fold {fold} (max_ks={row['max_ks']:.4f}) ===")
        run_training(
            df, classes=config["classes"], output_dir=output_dir, splits=splits,
            **feature_kwargs, **arch_kwargs, **other_run_kwargs,
        )
        metrics_df, _, _ = evaluate_val_metrics(
            df, classes=config["classes"], output_dir=output_dir, splits=splits, **feature_kwargs, **arch_kwargs,
        )
        macro_f1 = metrics_df["f1"].mean()
        print(f"{variant} fold {fold} macro F1: {macro_f1:.4f}")
        rows.append({"fold": fold, "max_ks": row["max_ks"], "macro_f1": macro_f1})

    summary = pd.DataFrame(rows)
    print()
    print(f"{variant} macro F1 range: {summary['macro_f1'].min():.3f} - {summary['macro_f1'].max():.3f}, "
          f"std: {summary['macro_f1'].std():.3f}")
    print(summary.to_string(index=False))
    split_search_dir.mkdir(parents=True, exist_ok=True)
    summary_path = split_search_dir / "split_sensitivity_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")
    return summary


if __name__ == "__main__":
    from build_points_table import build_and_save_points_table
    from taxonomy_separability import add_relative_features

    raw_df = build_and_save_points_table()
    raw_df = add_relative_features(raw_df)

    variant = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    run_split_sensitivity_for_variant(raw_df, variant)
