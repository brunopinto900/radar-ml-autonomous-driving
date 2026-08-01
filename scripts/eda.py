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
        mean_rcs=("rcs", "mean"),
        vr_std=("vr_compensated", "std"),
        # vr_mad=("vr_compensated", _mad),  # TEMP: commented out for speed, revert after
        # vr_q25=("vr_compensated", lambda x: x.quantile(0.25)),
        # vr_q75=("vr_compensated", lambda x: x.quantile(0.75)),
        mean_range=("range_sc", "mean"),
        x_min=("x_cc", "min"),
        x_max=("x_cc", "max"),
        y_min=("y_cc", "min"),
        y_max=("y_cc", "max"),
    ).reset_index()
    feats["extent"] = np.hypot(feats["x_max"] - feats["x_min"], feats["y_max"] - feats["y_min"])
    # feats["vr_iqr"] = feats["vr_q75"] - feats["vr_q25"]  # TEMP: commented out with vr_mad above

    # std of 1 point is NaN already; MAD/IQR of 1-2 points are ~0 but not NaN (a point can't
    # disagree with itself) - mask all three the same way so "too little data to measure
    # dispersion" isn't silently conflated with "measured zero dispersion" in MAD/IQR only.
    too_few_points = feats["n_points"] < 3
    feats.loc[too_few_points, "vr_std"] = np.nan
    return feats


def eda_1_large_vehicle_merge(df: pd.DataFrame, feats: pd.DataFrame, rcs_col: str = "rcs",
                               suffix: str = "") -> None:
    """Overlay the 4 large_vehicle sub-classes (RCS, Doppler spread, spatial extent), then
    check the merged large_vehicle group stays separated from the other final classes.
    rcs_col/suffix let this be re-run on range-compensated RCS (see FEATURE_MAP.md) without
    duplicating the function - pass rcs_col="rcs_range_comp", suffix="_range_comp_rcs"."""
    subclasses = ["train", "truck", "bus", "large_vehicle"]
    rcs_label = "range-compensated RCS [dB]" if suffix else "RCS [dBsm]"

    subclass_rcs = df.loc[df["label_name"].isin(subclasses), rcs_col]
    rcs_range = (subclass_rcs.min(), subclass_rcs.max())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for name in subclasses:
        color = SUBCLASS_COLORS[name]
        sub_points = df[df["label_name"] == name]
        axes[0].hist(sub_points[rcs_col], bins=30, range=rcs_range, density=True,
                     histtype="step", linewidth=1.8, color=color, label=name)

    axes[0].set_xlabel(rcs_label)
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
    path = EDA_DIR / f"01a_large_vehicle_subclasses{suffix}.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")

    classes = ["car", "large_vehicle", "pedestrian", "pedestrian_group", "two_wheeler"]
    class_rcs = df.loc[df["class_name"].isin(classes), rcs_col]
    rcs_range = (class_rcs.min(), class_rcs.max())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for name in classes:
        color = GROUP_COLORS[name]
        class_points = df[df["class_name"] == name]
        axes[0].hist(class_points[rcs_col], bins=30, range=rcs_range, density=True,
                     histtype="step", linewidth=1.8, color=color, label=name)

    axes[0].set_xlabel(rcs_label)
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
    path = EDA_DIR / f"01b_merged_vs_other_classes{suffix}.png"
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
RANGE_BINS = [0, 10, 20, 30, 40, 50, 60, 80, 100]


def add_range_compensated_rcs(df: pd.DataFrame) -> pd.DataFrame:
    """Adds rcs_range_comp: a shared (pooled, all-classes) non-parametric range trend -
    pooled median RCS per range bin, used as "expected RCS at this range" - subtracted per
    point. See FEATURE_MAP.md 02e/02f for why (RCS trends with range even within one class,
    likely CFAR detection-threshold bias) and the caveat (a single pooled trend doesn't fully
    flatten every class - large_vehicle specifically behaves differently)."""
    range_bin = pd.cut(df["range_sc"], RANGE_BINS)
    pooled_median = df["rcs"].groupby(range_bin, observed=True).median()
    expected_rcs = range_bin.map(pooled_median).astype(float)
    df = df.copy()
    df["rcs_range_comp"] = df["rcs"] - expected_rcs
    return df


def eda_2_rcs_vs_azimuth(df: pd.DataFrame, rcs_col: str = "rcs", suffix: str = "") -> None:
    """Median RCS vs. |azimuth_sc| (distance from sensor boresight), one line per final
    class - does the boresight-fading pattern hold for every class, or is it class-specific
    (i.e. a scene-geometry/class-mix confound rather than a sensor gain effect)?"""
    classes = ["car", "large_vehicle", "pedestrian", "pedestrian_group", "two_wheeler"]
    rcs_label = "median range-compensated RCS [dB]" if suffix else "median RCS [dBsm]"
    abs_az_deg = df["azimuth_sc"].abs() * 180 / np.pi
    az_bin = pd.cut(abs_az_deg, AZIMUTH_BINS)
    bin_mids = [interval.mid for interval in az_bin.cat.categories]

    fig, ax = plt.subplots(figsize=(9, 5))
    for name in classes:
        mask = df["class_name"] == name
        median_rcs = df.loc[mask, rcs_col].groupby(az_bin[mask], observed=True).median()
        median_rcs = median_rcs.reindex(az_bin.cat.categories)
        ax.plot(bin_mids, median_rcs.to_numpy(), marker="o", color=GROUP_COLORS[name], label=name)

    ax.set_xlabel("|azimuth_sc| bin midpoint [deg]")
    ax.set_ylabel(rcs_label)
    ax.set_title("Median RCS vs. distance from sensor boresight, by final class (per point)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = EDA_DIR / f"02a_rcs_vs_azimuth_by_class{suffix}.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")


def eda_2_rcs_vs_range_by_class(df: pd.DataFrame) -> None:
    """Median RCS vs. range_sc, one line per final class (per point) - FEATURE_MAP.md
    follow-up: does RCS still trend with range *within* a single class, or is the pooled
    range-RCS gradient (02d) fully explained by different classes sitting at different
    typical ranges? A flat within-class line means class-mix explains it; a trending line
    means there's a real per-object measurement effect on top of the class-mix."""
    classes = ["car", "large_vehicle", "pedestrian", "pedestrian_group", "two_wheeler"]
    range_bin = pd.cut(df["range_sc"], RANGE_BINS)
    bin_mids = [interval.mid for interval in range_bin.cat.categories]

    fig, ax = plt.subplots(figsize=(9, 5))
    for name in classes:
        mask = df["class_name"] == name
        median_rcs = df.loc[mask, "rcs"].groupby(range_bin[mask], observed=True).median()
        median_rcs = median_rcs.reindex(range_bin.cat.categories)
        ax.plot(bin_mids, median_rcs.to_numpy(), marker="o", color=GROUP_COLORS[name], label=name)

    ax.set_xlabel("range_sc bin midpoint [m]")
    ax.set_ylabel("median RCS [dBsm]")
    ax.set_title("Median RCS vs. range, by final class (per point) - within-class trend check")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = EDA_DIR / "02e_rcs_vs_range_by_class.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")


def eda_2_rcs_range_detrend(df: pd.DataFrame) -> None:
    """Fit one shared (pooled, all-classes) range trend and subtract it per point, then
    check: (a) the within-class trend is now flat (confirms the detrend worked), and (b)
    class separation survives on the residual (confirms detrending didn't destroy signal).
    Non-parametric on purpose - 02e's curve isn't linear (steep at short range, flattening
    at long range), so an empirical per-bin median is used as the "expected RCS at this
    range" instead of assuming a functional form."""
    classes = ["car", "large_vehicle", "pedestrian", "pedestrian_group", "two_wheeler"]
    range_bin = pd.cut(df["range_sc"], RANGE_BINS)
    df = add_range_compensated_rcs(df)
    residual_rcs = df["rcs_range_comp"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    bin_mids = [interval.mid for interval in range_bin.cat.categories]
    for name in classes:
        mask = df["class_name"] == name
        median_resid = residual_rcs[mask].groupby(range_bin[mask], observed=True).median()
        median_resid = median_resid.reindex(range_bin.cat.categories)
        axes[0].plot(bin_mids, median_resid.to_numpy(), marker="o", color=GROUP_COLORS[name], label=name)
    axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_xlabel("range_sc bin midpoint [m]")
    axes[0].set_ylabel("median residual RCS [dB]")
    axes[0].set_title("Residual RCS vs. range, by class - should now be flat")
    axes[0].legend(fontsize=8)

    resid_range = (residual_rcs.quantile(0.005), residual_rcs.quantile(0.995))
    for name in classes:
        color = GROUP_COLORS[name]
        mask = df["class_name"] == name
        axes[1].hist(residual_rcs[mask], bins=30, range=resid_range, density=True,
                     histtype="step", linewidth=1.8, color=color, label=name)
    axes[1].set_xlabel("residual RCS [dB]")
    axes[1].set_ylabel("density")
    axes[1].set_title("Residual RCS by final class - does separation survive?")
    axes[1].legend(fontsize=8)

    fig.suptitle("Range-detrended RCS (shared pooled trend subtracted per point)")
    fig.tight_layout()
    path = EDA_DIR / "02f_rcs_range_detrended.png"
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


def eda_2_rcs_vs_point_count_scatter(feats: pd.DataFrame, rcs_col: str = "mean_rcs",
                                      suffix: str = "") -> None:
    """2D scatter: mean RCS vs. point count, colored by class, per scan - the actual
    deliverable named in TODO.md's Day 4 line, to eyeball joint separability rather than
    each feature's marginal distribution alone (which is all the other item #2 plots show)."""
    classes = ["car", "large_vehicle", "pedestrian", "pedestrian_group", "two_wheeler"]
    rcs_label = "mean range-compensated RCS per scan [dB]" if suffix else "mean RCS per scan [dBsm]"
    rng = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=(9, 7))
    for name in classes:
        color = GROUP_COLORS[name]
        sub = feats[feats["class_name"] == name]
        jitter = rng.uniform(-0.3, 0.3, size=len(sub))  # n_points is discrete - jitter for visibility
        ax.scatter(sub["n_points"] + jitter, sub[rcs_col], s=6, alpha=0.15,
                   color=color, label=name, edgecolors="none")

    ax.set_xlabel("n_points per scan (jittered for visibility)")
    ax.set_ylabel(rcs_label)
    ax.set_title("Mean RCS vs. point count, by final class (per scan)")
    legend = ax.legend(fontsize=9, markerscale=4)
    for lh in legend.legend_handles:
        lh.set_alpha(1)
    fig.tight_layout()
    path = EDA_DIR / f"02c_rcs_vs_point_count_scatter{suffix}.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")


def eda_2_rcs_vs_point_count_scatter_by_range(feats: pd.DataFrame, rcs_col: str = "mean_rcs",
                                               suffix: str = "") -> None:
    """Same scatter as eda_2_rcs_vs_point_count_scatter (mean RCS vs. point count, per scan),
    but colored by mean_range instead of class - still a flat 2D scatter, just swapping what
    the color encodes, to check how much of the class-scatter's overlap is a range effect."""
    rcs_label = "mean range-compensated RCS per scan [dB]" if suffix else "mean RCS per scan [dBsm]"
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.3, 0.3, size=len(feats))

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(feats["n_points"] + jitter, feats[rcs_col], c=feats["mean_range"],
                     s=6, alpha=0.3, cmap="viridis", edgecolors="none")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("mean_range [m]")

    ax.set_xlabel("n_points per scan (jittered for visibility)")
    ax.set_ylabel(rcs_label)
    ax.set_title("Mean RCS vs. point count, colored by range (per scan, all classes pooled)")
    fig.tight_layout()
    path = EDA_DIR / f"02d_rcs_vs_point_count_by_range{suffix}.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")


def eda_2_rcs_vs_vr_std_scatter(feats: pd.DataFrame) -> None:
    """2D scatter: mean RCS vs. Doppler spread (vr_std), colored by class, per scan - the two
    strongest features found across the whole EDA (item #2/FEATURE_MAP.md for RCS, item #4
    for Doppler spread), checked jointly for the first time. Only scans with n_points >= 3
    have a defined vr_std (instance_features masks the rest to NaN - see item #4), so this
    necessarily has fewer points than the RCS-only scatter (02c) - pedestrian especially,
    given >54% of its scans are single-point."""
    classes = ["car", "large_vehicle", "pedestrian", "pedestrian_group", "two_wheeler"]
    # vr_std > 3.0 m/s is essentially all aliasing artifact (item #4/01c) - a real per-scan
    # spread that high isn't physically plausible for any of these classes, and it stretches
    # the axis enough to make the actual bulk of the data look misleadingly compressed.
    valid = feats.dropna(subset=["vr_std"])
    valid = valid[valid["vr_std"] <= 3.0]

    fig, ax = plt.subplots(figsize=(9, 7))
    for name in classes:
        color = GROUP_COLORS[name]
        sub = valid[valid["class_name"] == name]
        ax.scatter(sub["vr_std"], sub["mean_rcs"], s=6, alpha=0.15,
                   color=color, label=name, edgecolors="none")

    ax.set_xlabel("scan Doppler spread, std(vr_compensated) [m/s]")
    ax.set_ylabel("mean RCS per scan [dBsm]")
    ax.set_title("Mean RCS vs. Doppler spread, by final class (per scan, n_points >= 3, vr_std <= 3.0 m/s)")
    legend = ax.legend(fontsize=9, markerscale=4)
    for lh in legend.legend_handles:
        lh.set_alpha(1)
    fig.tight_layout()
    path = EDA_DIR / "02g_rcs_vs_vr_std_scatter.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")


def eda_2_rcs_vs_vr_mad_scatter(feats: pd.DataFrame) -> None:
    """Same as eda_2_rcs_vs_vr_std_scatter, but vr_mad instead of vr_std - the actual
    recommended feature (item #4: MAD resists a single aliased point far better than std),
    checked jointly with RCS the same way. No hardcoded upper cutoff needed here in
    principle (that was specifically an std/aliasing problem), but still uses a percentile
    xlim so a handful of remaining extreme values don't squash the axis."""
    classes = ["car", "large_vehicle", "pedestrian", "pedestrian_group", "two_wheeler"]
    valid = feats.dropna(subset=["vr_mad"])
    x_max = valid["vr_mad"].quantile(0.995)

    fig, ax = plt.subplots(figsize=(9, 7))
    for name in classes:
        color = GROUP_COLORS[name]
        sub = valid[valid["class_name"] == name]
        ax.scatter(sub["vr_mad"], sub["mean_rcs"], s=6, alpha=0.15,
                   color=color, label=name, edgecolors="none")

    ax.set_xlim(-0.05, x_max)
    ax.set_xlabel("scan Doppler MAD, MAD(vr_compensated) [m/s]")
    ax.set_ylabel("mean RCS per scan [dBsm]")
    ax.set_title("Mean RCS vs. Doppler MAD, by final class (per scan, n_points >= 3 only)")
    legend = ax.legend(fontsize=9, markerscale=4)
    for lh in legend.legend_handles:
        lh.set_alpha(1)
    fig.tight_layout()
    path = EDA_DIR / "02h_rcs_vs_vr_mad_scatter.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")


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

    df = add_range_compensated_rcs(df)
    comp_means = df.groupby(["sequence_name", "timestamp", "track_id"])["rcs_range_comp"] \
        .mean().rename("mean_rcs_comp")
    feats = feats.merge(comp_means, on=["sequence_name", "timestamp", "track_id"], how="left")

    eda_1_large_vehicle_merge(df, feats)
    eda_vr_std_shape(feats)
    # eda_vr_dispersion_robust(feats)  # TEMP: commented out for speed, revert after
    eda_2_rcs_vs_azimuth(df)
    eda_2_rcs_vs_range_by_class(df)
    eda_2_rcs_range_detrend(df)
    eda_2_point_count_and_doppler_center(df, feats)
    eda_2_rcs_vs_point_count_scatter(feats)
    eda_2_rcs_vs_point_count_scatter_by_range(feats)
    eda_2_rcs_vs_vr_std_scatter(feats)
    # eda_2_rcs_vs_vr_mad_scatter(feats)  # TEMP: commented out for speed, revert after (needs vr_mad above)
    eda_3_point_count_and_extent_vs_range(feats)

    print("\n--- regenerating RCS plots with range-compensated RCS (_range_comp_rcs) ---")
    eda_1_large_vehicle_merge(df, feats, rcs_col="rcs_range_comp", suffix="_range_comp_rcs")
    eda_2_rcs_vs_azimuth(df, rcs_col="rcs_range_comp", suffix="_range_comp_rcs")
    eda_2_rcs_vs_point_count_scatter(feats, rcs_col="mean_rcs_comp", suffix="_range_comp_rcs")
    eda_2_rcs_vs_point_count_scatter_by_range(feats, rcs_col="mean_rcs_comp", suffix="_range_comp_rcs")
