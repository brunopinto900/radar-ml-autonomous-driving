"""Day 5: per-feature value distributions across the final (grouped) class set, to inform
the per-instance histogram encoding's bin range/count.

Different job than scripts/taxonomy_separability.py's histograms: that was (and failed at)
judging class separability from 1D overlays. This is about understanding each feature's own
value range and shape before picking bin edges for it - a legitimate use of a histogram even
though eyeballing one is the wrong tool for a separability call.
"""
import matplotlib.pyplot as plt
import pandas as pd

from dataloader import CLASS_GROUPS, RESULTS_DIR
from taxonomy_separability import add_relative_features

FINAL_CLASSES = ["car", "large_vehicle", "bus", "two_wheeler", "pedestrian", "pedestrian_group"]
POINT_LEVEL_FEATURES = ["rcs", "vr_compensated", "x_rel", "y_rel"]
INSTANCE_LEVEL_FEATURES = ["doppler_spread"]
HISTOGRAM_FEATURES = POINT_LEVEL_FEATURES + INSTANCE_LEVEL_FEATURES


def apply_class_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw label_name to the final training class (Design_Decisions.md decision 1),
    dropping rows whose class isn't trained on (mapped to None)."""
    df = df.copy()
    df["group"] = df["label_name"].map(CLASS_GROUPS)
    return df.loc[df["group"].notna()]


def feature_values(df: pd.DataFrame, instances: pd.DataFrame, feature: str) -> pd.Series:
    """Point-level features pool every point; doppler_spread is one value per instance
    (broadcast across all its points by add_relative_features), so it's pulled from the
    deduped instance table instead - otherwise busier instances would be overrepresented."""
    return instances[feature] if feature in INSTANCE_LEVEL_FEATURES else df[feature]


def feature_percentiles(
    df: pd.DataFrame, features: list[str] = HISTOGRAM_FEATURES, classes: list[str] = FINAL_CLASSES
) -> pd.DataFrame:
    """1st/50th/99th percentile per feature per class - a numeric complement to the
    histograms, since eyeballing bin range off a chart alone is the same kind of visual
    judgment call already found unreliable for separability. Returns one row per class."""
    instances = df.drop_duplicates(["sequence_name", "timestamp", "track_id"])

    rows = []
    for cls in classes:
        sub = df.loc[df["group"] == cls]
        sub_instances = instances.loc[instances["group"] == cls]
        row = {"class": cls, "n_points": len(sub)}
        for feature in features:
            values = feature_values(sub, sub_instances, feature)
            row[f"{feature}_p1"] = values.quantile(0.01)
            row[f"{feature}_p50"] = values.quantile(0.50)
            row[f"{feature}_p99"] = values.quantile(0.99)
        rows.append(row)

    result = pd.DataFrame(rows).set_index("class")
    print(result.round(2).to_string())
    return result


def plot_bin_sweep(
    df: pd.DataFrame, features: list[str] = HISTOGRAM_FEATURES, bin_counts: tuple[int, ...] = (8, 16, 32, 64)
):
    """One figure per feature: the same pooled distribution (all final classes combined) shown
    at a few candidate bin counts side by side, to eyeball the resolution/sparsity tradeoff
    directly - not a per-class comparison. Point-level features pool every point;
    doppler_spread pools one value per instance (see feature_values). Saves each to
    results/bin_sweep_<feature>.png. Returns {feature: fig}."""
    instances = df.drop_duplicates(["sequence_name", "timestamp", "track_id"])

    figs = {}
    for feature in features:
        values = feature_values(df, instances, feature)
        lo, hi = values.quantile(0.01), values.quantile(0.99)
        fig, axes = plt.subplots(1, len(bin_counts), figsize=(4 * len(bin_counts), 4))
        for ax, n_bins in zip(axes, bin_counts):
            ax.hist(values, bins=n_bins, range=(lo, hi), color="tab:blue", edgecolor="k", linewidth=0.3)
            ax.set_title(f"{n_bins} bins")
            ax.set_xlabel(feature)
        fig.suptitle(f"{feature} - bin count sweep (pooled, {len(values)} values, range=[p1,p99])")
        fig.tight_layout()
        figs[feature] = fig

    feature_dist_dir = RESULTS_DIR / "feature_distributions"
    feature_dist_dir.mkdir(parents=True, exist_ok=True)
    for feature, fig in figs.items():
        path = feature_dist_dir / f"bin_sweep_{feature}.png"
        fig.savefig(path, dpi=150)
        print(f"Saved {path}")

    return figs


if __name__ == "__main__":
    from build_points_table import build_and_save_points_table

    df = build_and_save_points_table()
    df = add_relative_features(df)
    df = apply_class_groups(df)

    feature_percentiles(df)
    plot_bin_sweep(df)
