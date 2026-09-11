## MaxAD and the 5-feature encoding: what actually drove the gain

Follow-up to `MLP_Decisions_and_Findings.md` section 11 (`stat_descriptors`: explicit per-instance statistics instead of histograms). Started as a test of `maxAD` (max absolute deviation from the instance's own median: `max(|x - median(x)|)`) as a replacement for `std`/`mean`, added to `histogram_separability.build_stat_features` (`scripts/histogram_separability.py`). Ended up showing that the statistic barely mattered, the feature set did.

### First attempt: polar features, underperformed baseline

Two variants (`scripts/mlp_variants.py`/`MLP_CONFIG.json`), both on `rcs`/`vr_compensated`/`radial`/`azimuth_sc` (polar position, not Cartesian):

| Variant | Statistics | Macro F1 (single split) |
|---|---|---|
| Baseline | 16-bin histogram, `x_rel`/`y_rel` | 0.686 |
| `stat_descriptors` | mean/median/std | 0.658 |
| `maxad_stats` | median/maxAD | 0.650 |

Both underperformed baseline, `maxad_stats` slightly worse than `stat_descriptors`. Concentrated in `large_vehicle` and `pedestrian`, the two classes where central tendency (mean/median of `rcs`/`vr_compensated`) plausibly carries more signal than an outlier-driven deviation stat.

### Second attempt: swap in Cartesian position plus range_sc

Same statistics, but on `x_rel`/`y_rel`/`vr_compensated`/`range_sc`/`rcs`, baseline's own Cartesian position features plus `range_sc` added, instead of the polar pair. 200 epochs (vs 50 for the variants above, ruling out undertraining). Single-split result: macro F1 jumped to 0.706 (`maxad_stats_5features`) and 0.712 (`stat_descriptors_5features`), both well above baseline. That raised the obvious question: was this `maxAD` specifically, or just the feature swap? Two more variants isolated it: `quantile_bins_5features` (16-bin quantile histogram, same 5 features) and `histogram_5features` (16-bin equal-width histogram, same 5 features) both landed in the same 0.71-ish neighborhood on a single split. Conclusion at that point: it's the feature set, not `maxAD`, not discretization vs continuous statistics.

### Third attempt: is any of this real, or is it the same noise-floor artifact as section 5's range_sc test?

`MLP_Decisions_and_Findings.md` section 5 already tested a near-identical idea (add `range_sc` to baseline's own Cartesian features, no other change) and found the resulting +0.018 macro F1 gap sat inside baseline's split-choice noise floor, not a real gain. All four single-split numbers above add `range_sc` the same way and show gains of similar magnitude (+0.02 to +0.03), raising the same concern.

Resolved by running all four 5-feature variants through the actual 6-fold split-sensitivity check (`scripts/variant_split_sensitivity.py`, same 6 candidate splits as baseline/DeepReflecs), rather than trusting a single split either way:

| Variant | Mean macro F1 (6-fold) | Mean Δ vs baseline | Δ std | Wins |
|---|---|---|---|---|
| Baseline | 0.692 | -- | -- | -- |
| `histogram_5features` | 0.714 | +0.023 | 0.007 | 6/6 |
| `quantile_bins_5features` | 0.723 | +0.031 | 0.009 | 6/6 |
| `stat_descriptors_5features` | 0.725 | +0.034 | 0.007 | 6/6 |
| `maxad_stats_5features` | 0.723 | +0.032 | 0.008 | 6/6 |

All four win every one of the 6 folds, with tight, consistent deltas (std 0.007 to 0.009), far tighter than baseline's own raw between-fold std (0.027). That's stronger evidence than section 5's original `range_sc` test had: that test compared one single-split number against the overall range of unrelated baseline runs, not a real paired per-fold comparison across multiple independently-built encodings. The paired check says the gain is real: Cartesian position plus `range_sc` beats baseline's original 4 features by roughly +0.02 to +0.03 macro F1, confirmed, not noise.

### Final finding

`maxAD` doesn't beat `mean`/`median`/`std` on the same features (0.723 vs 0.725, 6-fold mean), and a quantile histogram doesn't lose to either (0.723). Encoding scheme doesn't matter once the feature set is right. What mattered, the entire validated effect, is `range_sc`, added to baseline's existing Cartesian `x_rel`/`y_rel`, not the specific statistic used to summarize it. See `model_comparison.md` for how this fits alongside DeepReflecs, and `notebooks/model_comparison.ipynb` for the aggregation (cached results only, no retraining).
