"""Per-feature value distributions across the final (grouped) class set, informing the
per-instance histogram encoding's bin range/count.

Different job than scripts/taxonomy_separability.py's histograms, which judged (and failed
at) class separability from 1D overlays. This is about each feature's own value range and
shape before picking bin edges, visual inspection is a legitimate tool here even though it's
the wrong one for a separability call."""
import matplotlib.pyplot as plt
import pandas as pd

from dataloader import CLASS_GROUPS, RESULTS_DIR
from taxonomy_separability import add_relative_features

FINAL_CLASSES = ["car", "large_vehicle", "bus", "two_wheeler", "pedestrian", "pedestrian_group"]

# current working taxonomy (Design_Decisions.md decision 1's final resolution: bus merged into
# large_vehicle too), matches dataloader.MLP_CLASS_GROUPS. Kept separate from FINAL_CLASSES for
# the same reason MLP_CLASS_GROUPS is kept separate from CLASS_GROUPS.
MLP_CLASSES = ["car", "large_vehicle", "two_wheeler", "pedestrian", "pedestrian_group"]

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
    deduped instance table instead, otherwise busier instances would be overrepresented."""
    return instances[feature] if feature in INSTANCE_LEVEL_FEATURES else df[feature]


def feature_percentiles(
    df: pd.DataFrame, features: list[str] = HISTOGRAM_FEATURES, classes: list[str] = FINAL_CLASSES
) -> pd.DataFrame:
    """1st/50th/99th percentile per feature per class, a numeric complement to the histograms,
    since reading bin range off a chart alone is the same kind of visual judgment call already
    found unreliable for separability. Returns one row per class."""
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


def plot_per_class_distributions(
    df: pd.DataFrame,
    features: list[str] = HISTOGRAM_FEATURES,
    classes: list[str] = FINAL_CLASSES,
    n_bins: int = 32,
):
    """One figure per feature: that feature's distribution split out per class (own subplot,
    own p1/p99 range each). A distribution pooled across classes can look multimodal purely
    from mixing classes with different central tendencies, this isolates whether a single
    class's own distribution has that shape. Saves each to
    results/feature_distributions/per_class_<feature>.png. Returns {feature: fig}."""
    instances = df.drop_duplicates(["sequence_name", "timestamp", "track_id"])

    figs = {}
    for feature in features:
        n_cols = 3
        n_rows = -(-len(classes) // n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
        axes = axes.flatten()
        for ax, cls in zip(axes, classes):
            sub = df.loc[df["group"] == cls]
            sub_instances = instances.loc[instances["group"] == cls]
            values = feature_values(sub, sub_instances, feature)
            lo, hi = values.quantile(0.01), values.quantile(0.99)
            ax.hist(values, bins=n_bins, range=(lo, hi), color="tab:blue", edgecolor="k", linewidth=0.3)
            ax.set_title(f"{cls} (n={len(values)})")
            ax.set_xlabel(feature)
        for ax in axes[len(classes):]:
            ax.axis("off")
        fig.suptitle(f"{feature}: per-class distribution ({n_bins} bins, range=[p1,p99] per class)")
        fig.tight_layout()
        figs[feature] = fig

    feature_dist_dir = RESULTS_DIR / "feature_distributions"
    feature_dist_dir.mkdir(parents=True, exist_ok=True)
    for feature, fig in figs.items():
        path = feature_dist_dir / f"per_class_{feature}.png"
        fig.savefig(path, dpi=150)
        print(f"Saved {path}")

    return figs


if __name__ == "__main__":
    from build_points_table import build_and_save_points_table

    df = build_and_save_points_table()
    df = add_relative_features(df)
    df = apply_class_groups(df)

    feature_percentiles(df)
    plot_per_class_distributions(df)
