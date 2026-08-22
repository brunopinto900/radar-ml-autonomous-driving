"""One off ablation: does Design_Decisions.md decision 1's class taxonomy (merge truck/train into
large_vehicle, keep bus separate) actually help the downstream MLP, versus two alternatives, run
through the same architecture/training config as mlp_classifier.py's standing model? Not meant to
replace decision 1's LR/RF probe evidence, this is a downstream sanity check now that a real
classifier exists, motivated by MLP_Decisions_and_Findings.md's bus/large_vehicle confusion.

Two variants, defined in mlp_variants.py and run together here:

    bus_merged: fold bus into large_vehicle too (5 classes). Tests whether the bus/large_vehicle
    confusion in the confusion matrix is better handled by not distinguishing them at all.

    truck_separate: undo the truck merge, so large_vehicle/truck/bus are three separate classes
    (7 classes). Re tests decision 1's original merge call, through the MLP instead of the LR/RF
    probe.

Both reuse the fixed sequence level split (results/data/sequence_split.json) unchanged, since
relabeling instances doesn't touch which sequences are in which split, and both train with
mlp_classifier's standing hyperparameters (N_BINS, HIDDEN_DIM, LEARNING_RATE, EPOCHS,
BATCH_SIZE). Findings: MLP_Decisions_and_Findings.md section 4.
"""
TAXONOMY_VARIANTS = ["bus_merged", "truck_separate"]


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
