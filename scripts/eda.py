"""Day 4 EDA on results/train_points.parquet. See EDA.md for the plan/findings.

One function per EDA.md item, in the same order. Never touches val/test, per
DESIGN_DECISIONS.md's split discipline.
"""
import numpy as np

import matplotlib
import pandas as pd

matplotlib.use("Agg")  # headless: this script only ever saves plots, never shows them
import matplotlib.pyplot as plt  # noqa: E402

from dataloader import GROUP_COLORS, RESULTS_DIR  # noqa: E402

EDA_DIR = RESULTS_DIR / "eda"

SUBCLASS_COLORS = {
    "train": "tab:pink",
    "truck": "gold",
    "bus": "tab:brown",
    "large_vehicle": "tab:orange",
}


def _mad(x: pd.Series) -> float:
    """Median absolute deviation - robust to a single extreme (e.g. aliased) point."""
    return (x - x.median()).abs().median()


def instance_features(df: pd.DataFrame) -> pd.DataFrame:
    """One row per object instance: label/class, point count, Doppler spread (std, MAD, IQR -
    see EDA.md item #4 for why std alone is aliasing-contaminated), spatial extent, range."""
    g = df.groupby(["sequence_name", "timestamp", "track_id"])
    feats = g.agg(
        label_name=("label_name", "first"),
        class_name=("class_name", "first"),
        n_points=("rcs", "size"),
        vr_std=("vr_compensated", "std"),
        # TEMP: commented out for speed, revert after
        # vr_mad=("vr_compensated", _mad),
        # vr_q25=("vr_compensated", lambda x: x.quantile(0.25)),
        # vr_q75=("vr_compensated", lambda x: x.quantile(0.75)),
        mean_range=("range_sc", "mean"),
        x_min=("x_cc", "min"),
        x_max=("x_cc", "max"),
        y_min=("y_cc", "min"),
        y_max=("y_cc", "max"),
    ).reset_index()
    feats["extent"] = np.hypot(feats["x_max"] - feats["x_min"], feats["y_max"] - feats["y_min"])
    # feats["vr_iqr"] = feats["vr_q75"] - feats["vr_q25"]

    # std of 1 point is NaN already; MAD/IQR of 1-2 points are ~0 but not NaN (a point can't
    # disagree with itself) - mask all three the same way so "too little data to measure
    # dispersion" isn't silently conflated with "measured zero dispersion" in MAD/IQR only.
    too_few_points = feats["n_points"] < 3
    feats.loc[too_few_points, ["vr_std"]] = np.nan
    return feats


def eda_1_large_vehicle_merge(df: pd.DataFrame, feats: pd.DataFrame) -> None:
    """Overlay the 4 large_vehicle sub-classes (RCS, Doppler spread, spatial extent), then
    check the merged large_vehicle group stays separated from the other final classes."""
    subclasses = ["train", "truck", "bus", "large_vehicle"]

    subclass_rcs = df.loc[df["label_name"].isin(subclasses), "rcs"]
    rcs_range = (subclass_rcs.min(), subclass_rcs.max())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for name in subclasses:
        color = SUBCLASS_COLORS[name]
        sub_points = df[df["label_name"] == name]
        axes[0].hist(sub_points["rcs"], bins=30, range=rcs_range, density=True,
                     histtype="step", linewidth=1.8, color=color, label=name)

    axes[0].set_xlabel("RCS [dBsm]")
    axes[0].set_ylabel("density")
    axes[0].set_title("RCS by sub-class (per point)")
    axes[0].legend(fontsize=8)

    vr_std_data = [feats.loc[feats["label_name"] == name, "vr_std"].dropna() for name in subclasses]
    axes[1].boxplot(vr_std_data, tick_labels=subclasses, showfliers=False)
    axes[1].set_ylabel("scan Doppler spread, std(vr_compensated) [m/s]")
    axes[1].set_title("Doppler spread by sub-class (per scan)")
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right")

    extent_data = [feats.loc[feats["label_name"] == name, "extent"] for name in subclasses]
    axes[2].boxplot(extent_data, tick_labels=subclasses)
    axes[2].set_ylabel("scan spatial extent [m]")
    axes[2].set_title("Spatial extent by sub-class (per scan)")
    plt.setp(axes[2].get_xticklabels(), rotation=45, ha="right")

    fig.suptitle("Do the large_vehicle sub-classes actually look alike to radar?")
    fig.tight_layout()
    path = EDA_DIR / "01a_large_vehicle_subclasses.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")

    classes = ["car", "large_vehicle", "pedestrian", "pedestrian_group", "two_wheeler"]
    class_rcs = df.loc[df["class_name"].isin(classes), "rcs"]
    rcs_range = (class_rcs.min(), class_rcs.max())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for name in classes:
        color = GROUP_COLORS[name]
        class_points = df[df["class_name"] == name]
        axes[0].hist(class_points["rcs"], bins=30, range=rcs_range, density=True,
                     histtype="step", linewidth=1.8, color=color, label=name)

    axes[0].set_xlabel("RCS [dBsm]")
    axes[0].set_ylabel("density")
    axes[0].set_title("RCS by final class (per point)")
    axes[0].legend(fontsize=8)

    vr_std_data = [feats.loc[feats["class_name"] == name, "vr_std"].dropna() for name in classes]
    axes[1].boxplot(vr_std_data, tick_labels=classes, showfliers=False)
    axes[1].set_ylabel("scan Doppler spread, std(vr_compensated) [m/s]")
    axes[1].set_title("Doppler spread by final class (per scan)")
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right")

    extent_data = [feats.loc[feats["class_name"] == name, "extent"] for name in classes]
    axes[2].boxplot(extent_data, tick_labels=classes)
    axes[2].set_ylabel("scan spatial extent [m]")
    axes[2].set_title("Spatial extent by final class (per scan)")
    plt.setp(axes[2].get_xticklabels(), rotation=45, ha="right")

    fig.suptitle("Is merged large_vehicle still separated from the other final classes?")
    fig.tight_layout()
    path = EDA_DIR / "01b_merged_vs_other_classes.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")


def eda_vr_std_shape(feats: pd.DataFrame) -> None:
    """Histogram of per-scan Doppler spread (vr_std) by final class - checks whether the
    boxplot's quartile positions actually reflect a Gaussian-like vs. skewed-near-zero shape,
    and whether the aliasing outliers form a visible separate cluster."""
    classes = ["car", "large_vehicle", "pedestrian", "pedestrian_group", "two_wheeler"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name in classes:
        color = GROUP_COLORS[name]
        vr_std = feats.loc[feats["class_name"] == name, "vr_std"].dropna()
        axes[0].hist(vr_std, bins=60, density=True, histtype="step", linewidth=1.8,
                     color=color, label=name)
        axes[1].hist(vr_std, bins=40, range=(0, 2), density=True, histtype="step",
                     linewidth=1.8, color=color, label=name)

    axes[0].set_yscale("log")
    axes[0].set_xlabel("scan Doppler spread, std(vr_compensated) [m/s]")
    axes[0].set_ylabel("density (log scale)")
    axes[0].set_title("Full range - outlier tail visible")
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("scan Doppler spread, std(vr_compensated) [m/s]")
    axes[1].set_title("Zoomed to 0-2 m/s - shape of the 'normal' bulk")
    axes[1].legend(fontsize=8)

    fig.suptitle("Is per-scan Doppler spread Gaussian-shaped, or skewed near zero?")
    fig.tight_layout()
    path = EDA_DIR / "01c_vr_std_histograms.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")


def eda_vr_dispersion_robust(feats: pd.DataFrame) -> None:
    """Same 'is large_vehicle separated from the other final classes' question as eda_1's
    Doppler-spread panel, but with std(vr_compensated) replaced by MAD and IQR - robust
    dispersion measures that resist a single aliased point dominating a scan's value."""
    classes = ["car", "large_vehicle", "pedestrian", "pedestrian_group", "two_wheeler"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for col, ax, ylabel, title in [
        ("vr_mad", axes[0], "scan vr MAD [m/s]", "Doppler dispersion (MAD) by final class"),
        ("vr_iqr", axes[1], "scan vr IQR [m/s]", "Doppler dispersion (IQR) by final class"),
    ]:
        data = [feats.loc[feats["class_name"] == name, col].dropna() for name in classes]
        ax.boxplot(data, tick_labels=classes, showfliers=False)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    fig.suptitle("Doppler dispersion by final class - robust alternatives to std(vr_compensated)")
    fig.tight_layout()
    path = EDA_DIR / "01d_vr_dispersion_robust.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")

    print("\nmedian per-scan dispersion by final class:")
    summary = feats.loc[feats["class_name"].isin(classes)].groupby("class_name")[
        ["vr_std", "vr_mad", "vr_iqr"]
    ].median().reindex(classes)
    print(summary)


AZIMUTH_BINS = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 90]


def eda_2_rcs_vs_azimuth(df: pd.DataFrame) -> None:
    """Median RCS vs. |azimuth_sc| (distance from sensor boresight), one line per final
    class - does the boresight-fading pattern hold for every class, or is it class-specific
    (i.e. a scene-geometry/class-mix confound rather than a sensor gain effect)?"""
    classes = ["car", "large_vehicle", "pedestrian", "pedestrian_group", "two_wheeler"]
    abs_az_deg = df["azimuth_sc"].abs() * 180 / np.pi
    az_bin = pd.cut(abs_az_deg, AZIMUTH_BINS)
    bin_mids = [interval.mid for interval in az_bin.cat.categories]

    fig, ax = plt.subplots(figsize=(9, 5))
    for name in classes:
        mask = df["class_name"] == name
        median_rcs = df.loc[mask, "rcs"].groupby(az_bin[mask], observed=True).median()
        median_rcs = median_rcs.reindex(az_bin.cat.categories)
        ax.plot(bin_mids, median_rcs.to_numpy(), marker="o", color=GROUP_COLORS[name], label=name)

    ax.set_xlabel("|azimuth_sc| bin midpoint [deg]")
    ax.set_ylabel("median RCS [dBsm]")
    ax.set_title("Median RCS vs. distance from sensor boresight, by final class (per point)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = EDA_DIR / "02a_rcs_vs_azimuth_by_class.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")


def eda_2_point_count_and_doppler_center(df: pd.DataFrame, feats: pd.DataFrame) -> None:
    """Item #2 baseline: per-class point count (per scan) and Doppler center (pooled,
    point-wise). RCS baseline already covered by 01b's first panel - not repeated here."""
    classes = ["car", "large_vehicle", "pedestrian", "pedestrian_group", "two_wheeler"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    n_points_data = [feats.loc[feats["class_name"] == name, "n_points"] for name in classes]
    axes[0].boxplot(n_points_data, tick_labels=classes, showfliers=False)
    axes[0].set_ylabel("n_points per scan")
    axes[0].set_title("Point count by final class (per scan)")
    plt.setp(axes[0].get_xticklabels(), rotation=45, ha="right")

    # min/max would be blown out by the same aliasing outliers as EDA.md item #4 - use the
    # 0.5/99.5 percentile range instead so the plotted bulk isn't squashed to a sliver.
    class_vr = df.loc[df["class_name"].isin(classes), "vr_compensated"]
    vr_range = (class_vr.quantile(0.005), class_vr.quantile(0.995))
    for name in classes:
        color = GROUP_COLORS[name]
        class_points = df[df["class_name"] == name]
        axes[1].hist(class_points["vr_compensated"], bins=40, range=vr_range, density=True,
                     histtype="step", linewidth=1.8, color=color, label=name)
    axes[1].set_xlabel("radial velocity, ego-compensated [m/s]")
    axes[1].set_ylabel("density")
    axes[1].set_title("Doppler center by final class (per point, pooled)")
    axes[1].legend(fontsize=8)

    fig.suptitle("Item #2 baseline: point count and Doppler center by final class")
    fig.tight_layout()
    path = EDA_DIR / "02b_point_count_and_doppler_center.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")


RANGE_BINS = [0, 10, 20, 30, 40, 50, 60, 80, 100]


def eda_3_point_count_and_extent_vs_range(feats: pd.DataFrame) -> None:
    """Point count and spatial extent vs. mean_range, one line per final class - per EDA.md
    item #3, checks whether class differences in n_points/extent hold up at fixed range, or
    are mostly explained by different classes sitting at different typical ranges."""
    classes = ["car", "large_vehicle", "pedestrian", "pedestrian_group", "two_wheeler"]
    range_bin = pd.cut(feats["mean_range"], RANGE_BINS)
    bin_mids = [interval.mid for interval in range_bin.cat.categories]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name in classes:
        color = GROUP_COLORS[name]
        mask = feats["class_name"] == name
        median_n = feats.loc[mask, "n_points"].groupby(range_bin[mask], observed=True).median()
        median_n = median_n.reindex(range_bin.cat.categories)
        axes[0].plot(bin_mids, median_n.to_numpy(), marker="o", color=color, label=name)

        median_extent = feats.loc[mask, "extent"].groupby(range_bin[mask], observed=True).median()
        median_extent = median_extent.reindex(range_bin.cat.categories)
        axes[1].plot(bin_mids, median_extent.to_numpy(), marker="o", color=color, label=name)

    axes[0].set_xlabel("mean_range bin midpoint [m]")
    axes[0].set_ylabel("median n_points")
    axes[0].set_title("Point count vs. range, by final class")
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("mean_range bin midpoint [m]")
    axes[1].set_ylabel("median extent [m]")
    axes[1].set_title("Spatial extent vs. range, by final class")
    axes[1].legend(fontsize=8)

    fig.suptitle("Point-count-vs-range confound (EDA.md item #3)")
    fig.tight_layout()
    path = EDA_DIR / "03a_point_count_extent_vs_range.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")


if __name__ == "__main__":
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(RESULTS_DIR / "train_points.parquet")
    print(f"Loaded {len(df)} points from train_points.parquet")

    feats = instance_features(df)
    print(f"{len(feats)} instances")

    eda_1_large_vehicle_merge(df, feats)
    eda_vr_std_shape(feats)
    # eda_vr_dispersion_robust(feats)  # TEMP: commented out for speed, revert after
    eda_2_rcs_vs_azimuth(df)
    eda_2_point_count_and_doppler_center(df, feats)
    eda_3_point_count_and_extent_vs_range(feats)
