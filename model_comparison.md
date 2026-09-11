# Model Comparison: Baseline vs Range-Extended Encodings vs DeepReflecs

Compares the baseline histogram MLP against a validated 5-feature encoding, `x_rel`/`y_rel`/`vr_compensated`/`range_sc`/`rcs` (baseline's own Cartesian features plus `range_sc`), tested across four different reductions, and DeepReflecs, the point-set network (`deepreflecs.md`). All numbers here are 6-fold split-sensitivity means, not single-split: an earlier draft of this document used single-split numbers, mistook the resulting gap for a probable noise-floor artifact (matching an earlier, weaker single-split test of the same idea, `range_sc` in `MLP_Decisions_and_Findings.md` section 5), and was wrong. The properly cross-validated numbers below settle it.

An even earlier attempt used polar position features (`radial`/`azimuth_sc`) instead of Cartesian `x_rel`/`y_rel`. That underperformed baseline outright (macro F1 0.65 to 0.66, single split, `MLP_MaxAD_Findings.md`) and isn't carried forward here, it's superseded entirely by the Cartesian-plus-`range_sc` feature set below.

## Feature sets

| Method | Point-level features | Reduction |
|---|---|---|
| Baseline | `rcs`, `vr_compensated`, `x_rel`, `y_rel` | 16-bin equal-width histogram + `doppler_spread` unbinned |
| `histogram_5features` | `x_rel`, `y_rel`, `vr_compensated`, `range_sc`, `rcs` | 16-bin equal-width histogram + `doppler_spread` unbinned |
| `quantile_bins_5features` | `x_rel`, `y_rel`, `vr_compensated`, `range_sc`, `rcs` | 16-bin quantile histogram + `doppler_spread` unbinned |
| `stat_descriptors_5features` | `x_rel`, `y_rel`, `vr_compensated`, `range_sc`, `rcs` | mean/median/std per feature + `doppler_spread` unbinned |
| `maxad_stats_5features` | `x_rel`, `y_rel`, `vr_compensated`, `range_sc`, `rcs` | median/maxAD per feature + `doppler_spread` unbinned |
| DeepReflecs | `x_rel`, `y_rel`, `rcs`, `vr_compensated`, `range_sc` | no reduction, raw point list, shared-weight point-set network |

The only feature-set difference between baseline and everything else in this table is `range_sc`. Once that's added, matching DeepReflecs' own feature set, all four MLP encodings beat baseline. None of the four beat each other.

## Results: 6-fold split-sensitivity (validated)

| Method | Mean macro F1 | Mean Δ vs baseline | Δ std | Wins |
|---|---|---|---|---|
| Baseline | 0.692 | -- | -- | -- |
| `histogram_5features` | 0.714 | +0.023 | 0.007 | 6/6 |
| `quantile_bins_5features` | 0.723 | +0.031 | 0.009 | 6/6 |
| `stat_descriptors_5features` | 0.725 | +0.034 | 0.007 | 6/6 |
| `maxad_stats_5features` | 0.723 | +0.032 | 0.008 | 6/6 |
| DeepReflecs | 0.735 | +0.044 | -- | 6/6 |

All four 5-feature MLP encodings beat baseline in every one of the 6 folds, with tight, consistent deltas (std 0.007 to 0.009), far tighter than baseline's own raw between-fold std (0.027). That consistency across four genuinely different reductions of the same features, not just one encoding's single-split number, is what makes this a validated result rather than the noise-floor artifact a weaker single-split comparison suggested.

Per class F1, 6-fold mean ± std, best per row in bold:

| Class | Baseline | `histogram_5features` | `quantile_bins_5features` | `stat_descriptors_5features` | `maxad_stats_5features` | DeepReflecs |
|---|---|---|---|---|---|---|
| `car` | 0.863 ± 0.020 | 0.866 ± 0.017 | 0.865 ± 0.020 | 0.868 ± 0.020 | 0.864 ± 0.022 | **0.877 ± 0.016** |
| `large_vehicle` | 0.719 ± 0.065 | 0.713 ± 0.055 | 0.696 ± 0.037 | 0.719 ± 0.049 | 0.714 ± 0.045 | **0.735 ± 0.049** |
| `two_wheeler` | 0.571 ± 0.133 | 0.614 ± 0.106 | 0.644 ± 0.097 | 0.647 ± 0.092 | 0.635 ± 0.091 | **0.650 ± 0.097** |
| `pedestrian` | 0.674 ± 0.030 | 0.684 ± 0.030 | 0.694 ± 0.036 | 0.690 ± 0.028 | 0.694 ± 0.029 | **0.703 ± 0.030** |
| `pedestrian_group` | 0.630 ± 0.045 | 0.693 ± 0.028 | **0.716 ± 0.038** | 0.702 ± 0.034 | 0.708 ± 0.037 | 0.710 ± 0.042 |

`large_vehicle` is flat across every method, essentially no real effect from feature set or architecture. `two_wheeler` and `pedestrian_group` carry almost the entire gain, for both the feature-set effect and DeepReflecs' further architecture effect. DeepReflecs wins 4 of 5 classes outright; `pedestrian_group` is the one exception, `quantile_bins_5features` edges it out, within noise (see verdict below).

**Per-class verdict, real or noise.** The table above mixes two different effects (feature choice: any 5-feature variant vs baseline; architecture: DeepReflecs vs the best 5-feature MLP variant), so "best in the row" isn't the same question as "is that difference real." Splitting them, using wins across the 6 folds and whether the mean delta clearly exceeds its own fold-to-fold std:

| Class | DeepReflecs vs baseline | DeepReflecs vs best MLP variant (architecture only) |
|---|---|---|
| `car` | Real. 6/6 folds, mean +0.014, std 0.006 | Real, small. 6/6 folds, mean +0.012, std 0.006 |
| `large_vehicle` | Noise. Only 4/6 folds, mean +0.016, std 0.027, delta smaller than its own spread | Real, but not what it looks like. 6/6 folds, mean +0.039, this is DeepReflecs recovering ground `quantile_bins_5features` lost on this class (0.696 vs baseline's 0.719), not DeepReflecs beating baseline |
| `two_wheeler` | Real. 6/6 folds, mean +0.079, std 0.043 | Real but negligible. 6/6 folds, mean +0.006, std 0.003, almost all of this class's gain is the feature-set effect |
| `pedestrian` | Real. 6/6 folds, mean +0.029, std 0.014 | Borderline. 6/6 folds, mean +0.009, std 0.010, right at the edge of its own noise |
| `pedestrian_group` | Real, strongest of all. 6/6 folds, mean +0.081, std 0.017 | Not real. Only 2/6 folds, mean −0.006, `quantile_bins_5features` is at least as good here |

Net: DeepReflecs' advantage over baseline is real for 4 of 5 classes. Once feature choice is separated out, the architecture effect on its own is only clearly real for `car`, negligible for `two_wheeler`/`pedestrian`, and slightly reversed for `pedestrian_group`. Most of the per-class story belongs to `range_sc`, not the network.

## Comparison

Encoding scheme genuinely doesn't matter: the four 5-feature MLP variants land within 0.011 macro F1 of each other (0.714 to 0.725), tighter than any one of their own fold-to-fold spreads. Equal-width bins, quantile bins, raw mean/median/std, raw median/maxAD, computed from the same 5 features, all end up in the same place.

What separates baseline from all four is exactly one feature, `range_sc`. That's now a validated effect (+0.02 to +0.03 macro F1, confirmed across 6 folds), not noise, reversing what this document said before running the fold check.

DeepReflecs adds a further, smaller effect on top: paired directly against `quantile_bins_5features` (identical feature set, same 6 folds), mean +0.012 macro F1, wins 6/6, delta std 0.004. That's a real architecture effect, distinct from and additive to the feature-set effect.

Net: DeepReflecs' total 6-fold advantage over baseline (+0.044) decomposes almost exactly into feature choice (+0.031, using `quantile_bins_5features` as the representative encoding) plus architecture (+0.012). Both pieces are validated, not one real and one assumed.

## References

`MLP_Decisions_and_Findings.md` (baseline), `MLP_MaxAD_Findings.md` (the 5-feature encoding variants and how the polar-feature dead end led to them), `deepreflecs.md` (DeepReflecs). Aggregated from cached results, no retraining, in `notebooks/model_comparison.ipynb`.
