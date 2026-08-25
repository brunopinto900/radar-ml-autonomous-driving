"""Ablation informing Design_Decisions.md decision 1: does merging bus into large_vehicle
(mlp_variants.py's `baseline`) actually help the MLP, vs. keeping it separate or also
splitting truck back out? Downstream sanity check on the LR/RF probe evidence, same
architecture/training config as mlp_classifier.py's standing model.

Variants (defined in mlp_variants.py), both compared against `baseline`:
- `bus_separate`: original, pre-merge taxonomy (6 classes).
- `truck_separate`: also undoes the truck merge, 3 separate vehicle classes (7 total);
  retests decision 1's truck merge through the MLP instead of the LR/RF probe.

Both reuse the fixed sequence-level split unchanged (relabeling doesn't affect split
membership) and mlp_classifier's standing hyperparameters. Findings:
MLP_Decisions_and_Findings.md section 4.
"""
TAXONOMY_VARIANTS = ["bus_separate", "truck_separate"]


if __name__ == "__main__":
    import sys

    from build_points_table import build_and_save_points_table
    from mlp_variants import run_variant
    from taxonomy_separability import add_relative_features

    raw_df = build_and_save_points_table()
    raw_df = add_relative_features(raw_df)

    variant = sys.argv[1] if len(sys.argv) > 1 else None
    if variant:
        run_variant(raw_df, variant)
    else:
        for name in TAXONOMY_VARIANTS:
            run_variant(raw_df, name)
