"""Single source of truth for every trained MLP variant produced so far, run or load one by
name instead of hand assembling classes/class_groups/output_dir per call. `baseline` is the
current working model (bus merged into large_vehicle, Design_Decisions.md decision 1's final
resolution) and lives at MLP_DIR, mlp_classifier.py's own default. `bus_separate` is the
original, pre-merge taxonomy, kept for reference/comparison, not the standing model anymore.
`epochs_1000` and `truck_separate` are the two ablations that informed the merge decision
(MLP_Decisions_and_Findings.md sections 2 and 4).
"""
from dataloader import CLASS_GROUPS, MLP_CLASS_GROUPS, RESULTS_DIR
from feature_distributions import FINAL_CLASSES, MLP_CLASSES
from mlp_classifier import MLP_DIR, evaluate_val_metrics, run_training

EXPERIMENT_DIR = RESULTS_DIR / "mlp" / "class_taxonomy_experiment"

MLP_VARIANTS = {
    "baseline": {
        "classes": MLP_CLASSES,
        "class_groups": MLP_CLASS_GROUPS,
        "output_dir": MLP_DIR,
        "run_kwargs": {},
    },
    "bus_separate": {
        "classes": FINAL_CLASSES,
        "class_groups": CLASS_GROUPS,
        "output_dir": MLP_DIR / "bus_separate",
        "run_kwargs": {},
    },
    "epochs_1000": {
        "classes": FINAL_CLASSES,
        "class_groups": CLASS_GROUPS,
        "output_dir": MLP_DIR / "epochs_1000",
        "run_kwargs": {"epochs": 1000},
    },
    "truck_separate": {
        "classes": ["car", "large_vehicle", "truck", "bus", "two_wheeler", "pedestrian", "pedestrian_group"],
        "class_groups": {**CLASS_GROUPS, "truck": "truck"},
        "output_dir": EXPERIMENT_DIR / "truck_separate",
        "run_kwargs": {},
    },
}


def build_variant_df(raw_df, variant: str):
    """raw_df must already have x_rel/y_rel/doppler_spread (taxonomy_separability.add_relative_
    features) but not yet have `group` applied, each variant maps label_name to group itself."""
    class_groups = MLP_VARIANTS[variant]["class_groups"]
    df = raw_df.copy()
    df["group"] = df["label_name"].map(class_groups)
    return df.loc[df["group"].notna()]


def run_variant(raw_df, variant: str):
    """Trains (or loads from cache) and evaluates val metrics for the named variant. See
    MLP_VARIANTS for what's available. Returns (model, history, metrics_df, cm_fig, bar_fig)."""
    config = MLP_VARIANTS[variant]
    df = build_variant_df(raw_df, variant)
    model, history, X_test, y_test = run_training(
        df, classes=config["classes"], output_dir=config["output_dir"], **config["run_kwargs"]
    )
    metrics_df, cm_fig, bar_fig = evaluate_val_metrics(df, classes=config["classes"], output_dir=config["output_dir"])
    return model, history, metrics_df, cm_fig, bar_fig


if __name__ == "__main__":
    import sys

    from build_points_table import build_and_save_points_table
    from taxonomy_separability import add_relative_features

    raw_df = build_and_save_points_table()
    raw_df = add_relative_features(raw_df)

    variant = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    run_variant(raw_df, variant)
