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

Resolved by running all four 5-feature variants through the actual 6-fold split-sensitivity check (`scripts/variant_split_sensitivity.py`, same 6 candidate splits as baseline/DeepReflecs), rather than trusting a single split either way. All four win every one of the 6 folds, with tight, consistent deltas (std 0.007 to 0.009), far tighter than baseline's own raw between-fold std (0.027). Full table: `model_comparison.md`'s Results section.

That's stronger evidence than section 5's original `range_sc` test had: that test compared one single-split number against the overall range of unrelated baseline runs, not a real paired per-fold comparison across multiple independently-built encodings. The paired check says the gain is real: Cartesian position plus `range_sc` beats baseline's original 4 features by roughly +0.02 to +0.03 macro F1, confirmed, not noise.

### Final finding

`maxAD` doesn't beat `mean`/`median`/`std` on the same features (0.723 vs 0.725, 6-fold mean), and a quantile histogram doesn't lose to either (0.723). Encoding scheme doesn't matter once the feature set is right. What mattered, the entire validated effect, is `range_sc`, added to baseline's existing Cartesian `x_rel`/`y_rel`, not the specific statistic used to summarize it. See `model_comparison.md` for how this fits alongside DeepReflecs, and `notebooks/model_comparison.ipynb` for the aggregation (cached results only, no retraining).

### Considered, not run

Three follow-up questions came up after the finding above, each reasoned through against existing evidence rather than retrained, since none of them looked likely enough to justify another 6-fold pass (1.3-1.5h each).

- **Does removing `doppler_spread` improve the 5-feature variants?** No expected effect. Zero-out importance on the `combined_features` model already measured `doppler_spread`'s contribution at -0.006 macro F1 (`MLP_Decisions_and_Findings.md` section 14a), second smallest of twelve feature groups. The `standardized` variant separately showed even a much larger perturbation, fixing its scale mismatch against the [0,1] histogram bins, landed at 0.688 vs baseline's 0.686, inside noise. Nothing suggests removing it entirely would move the needle either direction.
- **Is the `range_sc` effect position (scene composition) or shape?** Proposed test: swap the 16-bin `range_sc` histogram for a single unbinned scalar, the same treatment `doppler_spread` gets, and check whether that alone recovers most of the gain. Not run: a single central-tendency scalar conflates two things at once, dropping within-instance shape and giving the network a lossier location estimate, so a negative result wouldn't distinguish "it's shape" from "the network just needs the extra resolution to pin down the same position." A clean test would need a location-only statistic (e.g. median `range_sc`) compared separately against a shape-only statistic with location removed (`range_sc` demeaned per instance, then dispersion), and even then the effect being split is already only +0.02 to +0.03 macro F1, splitting it further risks landing both halves inside the fold-to-fold noise (std 0.007-0.009) without resolving anything. Also wouldn't settle "artifact vs real signal" on its own either way, that needs checking the range-class relationship across different recording sequences, a separate and more expensive question.
- **Would bigger models help now that `range_sc` is in the feature set?** No expected effect. The capacity ablation (`hidden8`/`hidden32`/`hidden64`, an 8x range on `HIDDEN_DIM`) was already flat, even per-class, on baseline's 65-dim input (`MLP_Decisions_and_Findings.md` section 6), consistent with section 10's sparsity ceiling: the bottleneck is information per instance, not the model's capacity to combine it. The encoding-invariance result above is independent evidence pointing the same way, if the true feature-to-class relationship needed more capacity to fit, different encodings (equal-width vs quantile bins vs raw statistics) would expose that complexity differently, and they didn't, all four landed within 0.011 of each other. `quantile_bins_5features`'s 81-dim input is only 25% wider than baseline's 65, a much smaller perturbation than the 8x capacity range that already found nothing.
