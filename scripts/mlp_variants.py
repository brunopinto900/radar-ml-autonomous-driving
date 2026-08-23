"""Loads and resolves MLP_CONFIG.json (project root) into MLP_VARIANTS, run or load a variant by
name instead of hand assembling classes/class_groups/output_dir per call. See MLP_CONFIG.json's
own "_comment"/"_note" fields for what each variant is and why - this module only resolves the
config into runnable form, it isn't the place to read variant descriptions anymore.

Resolution: each entry's "class_groups_base" (CLASS_GROUPS or MLP_CLASS_GROUPS, dataloader.py's
real taxonomy dicts) is merged with its optional "class_groups_overrides" - the mapping itself is
never duplicated in the JSON, only referenced by name, so it always has exactly one source of
truth. "output_dir" is relative to MLP_DIR (results/mlp/); null means MLP_DIR itself.
"""
import json
from pathlib import Path

from dataloader import CLASS_GROUPS, MLP_CLASS_GROUPS
from mlp_classifier import MLP_DIR, evaluate_val_metrics, run_training

CONFIG_PATH = Path(__file__).resolve().parent.parent / "MLP_CONFIG.json"
_CLASS_GROUPS_BASES = {"CLASS_GROUPS": CLASS_GROUPS, "MLP_CLASS_GROUPS": MLP_CLASS_GROUPS}


def _load_variants() -> dict:
    raw = json.loads(CONFIG_PATH.read_text())
    variants = {}
    for name, entry in raw["variants"].items():
        class_groups = {**_CLASS_GROUPS_BASES[entry["class_groups_base"]], **entry.get("class_groups_overrides", {})}
        output_dir = MLP_DIR if entry["output_dir"] is None else MLP_DIR / entry["output_dir"]
        config = {
            "classes": entry["classes"],
            "class_groups": class_groups,
            "output_dir": output_dir,
            "run_kwargs": entry.get("run_kwargs", {}),
        }
        for optional_key in ("features", "extra_features"):
            if optional_key in entry:
                config[optional_key] = entry[optional_key]
        variants[name] = config
    return variants


MLP_VARIANTS = _load_variants()


def build_variant_df(raw_df, variant: str):
    """raw_df must already have x_rel/y_rel/doppler_spread (taxonomy_separability.add_relative_
    features) but not yet have `group` applied, each variant maps label_name to group itself."""
    class_groups = MLP_VARIANTS[variant]["class_groups"]
    df = raw_df.copy()
    df["group"] = df["label_name"].map(class_groups)
    return df.loc[df["group"].notna()]


def run_variant(raw_df, variant: str):
    """Trains (or loads from cache) and evaluates val metrics for the named variant. See
    MLP_CONFIG.json for what's available. A variant may optionally set "features"/"extra_features"
    to swap the feature set instead of (or alongside) the taxonomy/hyperparameters - falls back
    to mlp_classifier.py's own defaults (POINT_LEVEL_FEATURES / doppler_spread) if absent.
    `hidden_dim`, if set in "run_kwargs", also has to reach evaluate_val_metrics (not just
    run_training), since reloading the cached model needs to reconstruct the exact same
    architecture. Returns (model, history, metrics_df, cm_fig, bar_fig)."""
    config = MLP_VARIANTS[variant]
    df = build_variant_df(raw_df, variant)
    feature_kwargs = {k: config[k] for k in ("features", "extra_features") if k in config}
    arch_kwargs = {k: config["run_kwargs"][k] for k in ("hidden_dim",) if k in config["run_kwargs"]}
    model, history, X_test, y_test = run_training(
        df, classes=config["classes"], output_dir=config["output_dir"], **feature_kwargs, **config["run_kwargs"]
    )
    metrics_df, cm_fig, bar_fig = evaluate_val_metrics(
        df, classes=config["classes"], output_dir=config["output_dir"], **feature_kwargs, **arch_kwargs
    )
    return model, history, metrics_df, cm_fig, bar_fig


if __name__ == "__main__":
    import sys

    from build_points_table import build_and_save_points_table
    from taxonomy_separability import add_relative_features

    raw_df = build_and_save_points_table()
    raw_df = add_relative_features(raw_df)

    variant = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    run_variant(raw_df, variant)
