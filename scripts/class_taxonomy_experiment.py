"""One off ablation that informed Design_Decisions.md decision 1's final resolution: does
keeping bus separate from large_vehicle actually help the MLP, versus merging it in (the
standing choice, mlp_variants.py's `baseline`), or splitting truck back out instead? Not meant
to replace the LR/RF probe evidence, this is a downstream sanity check run through the same
architecture/training config as mlp_classifier.py's standing model.

Two variants, defined in mlp_variants.py and run together here, both compared against the
`baseline` (bus merged) variant:

    bus_separate: the original, pre-merge taxonomy (6 classes) - was bus actually better off on
    its own?

    truck_separate: undo the truck merge too, so large_vehicle/truck/bus are three separate
    classes (7 classes). Re tests decision 1's original truck merge call, through the MLP
    instead of the LR/RF probe.

Both reuse the fixed sequence level split (results/data/sequence_split.json) unchanged, since
relabeling instances doesn't touch which sequences are in which split, and both train with
mlp_classifier's standing hyperparameters (N_BINS, HIDDEN_DIM, LEARNING_RATE, EPOCHS,
BATCH_SIZE). Findings: MLP_Decisions_and_Findings.md section 4.
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
