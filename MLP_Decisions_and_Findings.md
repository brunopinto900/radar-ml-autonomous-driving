## Summary

Ranked by size of effect, not the order tested. Each item points at the numbered section below with the full methodology, numbers, and caveats.

**0. Histogram confound**
Binning noise, not normalization, is the real problem at this n: with a few dozen instances split across 16 bins per feature, each bin's height reflects which handful of points happened to land there, not the population, and that happens whether or not the histogram is normalized.
At this sample size, don't bin: use per instance scatter/strip plots or ECDFs instead, and pool misclassified two_wheeler across the 6 existing split sensitivity folds (already trained, sections 15 to 18) to grow n before reading anything off a single val confusion matrix.

**1. Sparsity is the ceiling, not the model or the features (section 10).** Slicing the trained model's own val predictions by instance point count, no retraining, shows macro F1 roughly doubling from 1 to 5 points/instance (0.381 -> 0.764). Ten independent axes of model and feature changes landed inside the noise floor and none of them affect this: capacity (section 6), depth (section 7), point count as an input feature (section 8), spatial/azimuth extent (section 9), explicit per-instance statistics instead of histograms (section 11), bin edge placement (section 12). `large_vehicle` is the clearest case, a size question structurally unanswerable from 1-2 points. The fix is points aggregated across scans over time, not a better single-scan feature or a bigger model.

**2. combined_features: the one validated positive result (section 14).** Stacking every previously-tested feature additively on top of `baseline`, nothing removed, gives +0.036 macro F1, mean across all 6 splits of the split-sensitivity check, winning every single fold (sign test: ~1.6% probability of that outcome by chance). Concentrated in `two_wheeler` and `pedestrian_group`; does not help `large_vehicle`, the class sparsity hits hardest, so this is separate, real signal, not a dent in the sparsity ceiling. About a tenth the size of finding 1. Zero-out importance (section 14a) and isolation tests (section 14b) show the gain isn't reducible to one or two standout features, it's an interaction across several.

**3. Splits are lumpy, not just noisy.** `StratifiedGroupKFold` only matches instance-count ratios and can't split a sequence, so hitting the right ratio means moving whole sequences around. If a class's supply is concentrated in a couple of sequences, whichever few land in val define that split's entire picture of the class, regardless of how correct the overall ratio looks. A class's radar signature also isn't one fixed thing, it varies by aspect angle and motion state (a bicycle stopped and front-on reads near-zero Doppler and a narrow RCS cross-section; a bicycle crossing broadside also reads near-zero Doppler but a much larger RCS cross-section, same label, different feature distribution). This is a real part of `two_wheeler`'s instability across runs (top1 sequence share 18.8%, top3 33.8%). More folds don't fix this, they only reveal it as variance. Section 15 traces the mechanism directly: sparsity makes `two_wheeler`'s per-instance `vr_compensated` noisy across folds, and `vr_compensated` is the model's single most load-bearing feature, so that noise actually moves F1; `large_vehicle` has a comparably unstable feature of its own (`spatial_extent`) but the model barely relies on it, so its instability doesn't translate the same way. Section 16 confirms this causally (retraining without `vr_compensated` narrows `two_wheeler`'s spread, both raw and relative); section 17 checks whether the instability itself traces to `two_wheeler` merging two different velocity regimes (`bicycle`/`motorized_two_wheeler`), mixed evidence, not a clean confirmation. This is why `range_sc`'s apparent +0.018 (section 5) and `deep10`'s +0.023 (section 7) both evaporated under the 6-fold split-sensitivity check, while `combined_features`'s gain survived it. Never compare metrics across two different splits, even both stratified by class proportion, a number from outside this project's fixed split is a different measurement, not merely a caveat away from being comparable.

**4. Root cause is data collection, not splitting.** RadarScenes is fixed and already collected: naturalistic driving passively records whatever crosses the sensor's path, so rare classes end up with thin, incidental coverage of their possible presentations while common classes (`car`) receive broad coverage inherently. The practical response isn't a better split, a split-selection test scoring 6 candidate splits by train/val distribution match found no candidate to be a demonstrated improvement (macro F1 ranged 0.651-0.734 purely from split choice, best-matching split scored near the worst). Instead, it's identifying which classes have this thin coverage before trusting a single number for them, instance count, distinct sequence count, share from the single busiest sequence (`taxonomy_separability.plot_taxonomy_class_balance`), a groupby, not a training run.

**5. Everything else tested, ruled out inside noise:** batch size (section 1), capacity (section 6), depth (section 7), the `doppler_spread` scale mismatch (section 7), point count as an input feature (section 8), spatial/azimuth extent alone (section 9), explicit statistics instead of histograms (section 11), bin edge placement (section 12), `rcs_extent`/`spatial_extent` in isolation (section 14b). Dropping sparse instances instead of working around them was also tried and made things much worse (val accuracy 74.1% -> 48.6%), confirming sparse instances carry real, weak-but-usable signal rather than noise.

**6. Process lesson: cheap-diagnostic-first discipline was inconsistent.** Section 10's point count diagnostic (no retraining, slices an existing model's predictions) directly answers the sparsity question and is the cheapest test in the whole program, it should have run second or third, not eighth. Most of sections 6-9 and 11-12 are retrain-heavy proxies for the same question it answers directly, run before it rather than after. The truck_car_regroup detour (an old-project taxonomy idea, tested and reverted, not in this log) was the same mistake in a more expensive form: a full 200-epoch retrain plus a separability probe on an unreplicated hypothesis, before it was abandoned. Run the low-cost diagnostic before committing to an expensive one, not the reverse.

Detailed, chronological log below.

## Model config

**Architecture:** 3 layer MLP (2 hidden layers of 16 units each, ReLU), input 65 features, output 5 classes (`MLP_CLASSES`: `car`, `large_vehicle`, `two_wheeler`, `pedestrian`, `pedestrian_group`; bus merged into large_vehicle per decision 1's final resolution).

**Feature design:** per instance histogram encoding. `rcs`, `vr_compensated`, `x_rel`, `y_rel` each binned into N_BINS=16 fraction of points per bin columns (64 dims total), plus `doppler_spread` appended unbinned (1 dim), for 65 dims total. Bin edges use the 1st to 99th percentile range, fit on train only and reused unchanged for val/test.

**Training:** Adam optimizer, class weighted cross entropy loss (weight = max class count / class count). LEARNING_RATE=4e-5, EPOCHS=50, BATCH_SIZE=128, RANDOM_STATE=0.

**Data:** fixed sequence grouped train/val/test split (`Design_Decisions.md` decision 5).

**Noise floor standard:** established via a 6-fold split-sensitivity check (`split_sensitivity.py`, same 6 valid splits at fixed 70/15/15 proportions throughout this doc): macro F1 across splits ranges 0.651-0.734 (std 0.027) on `baseline`'s exact config, purely from which sequences land in val. Per-class spread ranges from 0.056 (`car`) to 0.386 (`two_wheeler`). Every delta reported below is judged against this: real only if it clears the affected class's own spread. Referenced below as "the noise floor" or "section 5", where the `range_sc` gap first motivated running this check.

## 1. Batch size and learning rate

Batch size affects whether rare classes contribute to each gradient update. bus makes up only 1.9% of the 358,210 train instances (vs. 42.8% for car); at that frequency, bus is absent from 54% of batches at bs=32 and 30% at bs=64.

**Decision:** use batch_size=128, the smallest candidate keeping every class's miss probability below 10%.

Scale the learning rate accordingly:

```
lr = 1e-5 × (128 / 32) = 4e-5
```

This follows the linear scaling rule and avoids slowing training through both larger batches and a reduced learning rate.

See `scripts/batch_size_selection.py`.

**Post-merge recheck:** the miss-probability argument above predates decision 1's final resolution (bus merged into large_vehicle), which changes the class frequencies it was computed from - two_wheeler (7.0% of train) no longer needs bs=128 to stay under 10% miss probability the way bus (1.9%) did. Retested with `bs32` (`batch_size=32`, `lr=1e-5` per the linear scaling rule) against the merged-taxonomy `baseline`: macro F1 0.687 vs 0.686, every per-class delta within 0.005, both comfortably inside the 0.651-0.734 noise floor. `scripts/mlp_variants.py`'s `bs32`, results at `results/mlp/bs32/`.

**Decision confirmed:** batch_size=128 isn't just theoretically justified anymore, an actual smaller-batch run shows no measurable difference either way, so there's no cost to keeping the larger, theoretically-motivated batch size.

## 2. Epoch count: no gain past roughly epoch 200

Increasing training from 50 to 1000 epochs produced only a small macro F1 improvement (0.611 to 0.626 in the original 6 class taxonomy) and mostly added training cost and validation noise, validation accuracy plateaued after roughly 150 to 200 epochs. bus specifically got slightly worse with more training while most other classes improved, a gradient starvation signature (bus remained poorly represented in individual batches) that is now moot, since bus was merged into large_vehicle (see `Design_Decisions.md` decision 1).

**Decision:** retain EPOCHS=50.

## 3. Confusion matrix findings (epochs=50, original 6 class taxonomy, `results/mlp/bus_separate/mlp_confusion_matrix.png`)

Row-normalized, val set:

- `car`: 73% correct, the most accurate label, and the majority class. 10% confused as `large_vehicle`.
- `large_vehicle`: confused as `car` and `bus`.
- `bus`: confused as `large_vehicle`.
- `two_wheeler`: confused as `pedestrian` and `pedestrian_group`.
- `pedestrian`: high recall.
- `pedestrian_group`: frequently confused as `pedestrian`, as expected given point sparsity (an isolated pedestrian and a sparse group can look similar with few points to work with).

**Cross-check against section 2 above:** the confusion isn't scattered, it clusters within car/large_vehicle/bus (all "vehicle" classes) and separately within pedestrian/pedestrian_group, rather than spreading across unrelated classes. That pattern is more consistent with a feature-separability ceiling (the histogram features can't fully tell `bus` and `large_vehicle` apart) than with pure gradient starvation. It's also in tension with the starvation story that `bus`'s recall (71%) is *higher* than `two_wheeler`'s (52%), despite `bus` being the far rarer class (1.9% vs 7.0% of train). If rarity/starvation were the dominant driver, that ordering would likely be reversed. This is what motivated the taxonomy experiment below.

## 4. Class taxonomy experiment (bus merged vs `bus_separate` vs `truck_separate`, 50 epochs each)

Macro F1: `bus_separate` (original 6 classes) 0.611, bus merged (5 classes, now `baseline`) 0.686, `truck_separate` (7 classes) 0.541.

Bus merged: folding `bus` into `large_vehicle` raises the combined class to F1=0.756, above either `bus` (0.543) or `large_vehicle` (0.492) alone in `bus_separate`. Supports merging `bus` in too, extending `Design_Decisions.md` decision 1's logic. This variant is now `mlp_variants.py`'s `baseline`, and its results live at `results/mlp/` directly rather than under the experiment subfolder.

`truck_separate`: splitting `truck` back out leaves pure `large_vehicle` at only 1,292 train instances, and it becomes nearly unusable (F1=0.098). Confirms decision 1's `truck` merge was correct.

Scripts: `scripts/class_taxonomy_experiment.py`, `scripts/mlp_variants.py`. Results: `results/mlp/` (baseline), `results/mlp/bus_separate/`, `results/mlp/class_taxonomy_experiment/truck_separate/`.

This directly informed `Design_Decisions.md` decision 1's final resolution: merge `bus` into `large_vehicle` too.

## 5. Feature swap: range_sc in place of doppler_spread, and a val split caution

Same architecture/training config as `baseline`, but `range_sc` (per point sensor range) binned into the same N_BINS=16 columns used for `rcs`/`vr_compensated`/`x_rel`/`y_rel`, instead of `doppler_spread` appended unbinned. `scripts/mlp_variants.py`'s `range_sc` variant, results at `results/mlp/range_sc/`.

Macro F1: `baseline` (doppler_spread) 0.686, `range_sc` 0.704 (+0.018), improves `two_wheeler`/`pedestrian`/`pedestrian_group`, slightly worse on `car`/`large_vehicle`. This is the gap that motivated the noise floor check above (Model config): it doesn't survive that check, 0.018 sits well inside the 0.651-0.734 range from split choice alone, not a real gain.

Cross-checked against a `range_sc` run tried independently before this project's fixed split existed: same features/model/bins/weighting, 20 epochs instead of 50, a materially different result (`large_vehicle`/`pedestrian_group` were the weakest classes there, close to the opposite ranking). Val class counts don't match either (`large_vehicle`: 4,456 old vs 6,247 now), confirming it used a different val set entirely, not the same fixed split with different randomness. Two runs are only comparable on the exact same split, the whole reason `sequence_split.py`'s fixed split exists.

**Open question, not yet resolved:** does `range_sc`'s F1 gain reflect a real radar signal, or a range/point-sparsity shortcut (sparse points at long range predicted as `large_vehicle`/`car`, a failure mode observed independently in the older run)? Not yet tested. Candidate checks: per-class `range_sc` percentiles, a permutation test on the `range_sc` columns in val, or a counterfactual swap (overwrite `range_sc` on real pedestrian instances with typical `large_vehicle` values, check if the prediction flips). `range_sc` should not be treated as the better feature until one of these is actually run.

## 6. Architecture capacity: hidden_dim ablation

Is the 16-unit hidden layer a bottleneck, or does the histogram feature representation cap performance regardless of model size? Same architecture shape (2 hidden layers), same features/taxonomy/batch size/LR/epochs as `baseline`, only `hidden_dim` varied: 8, 32, 64, against the standing 16. `scripts/mlp_variants.py`'s `hidden8`/`hidden32`/`hidden64`, results at `results/mlp/hidden8/`, `results/mlp/hidden32/`, `results/mlp/hidden64/`.

Macro F1: 0.688 (`hidden8`), 0.686 (`baseline`, 16), 0.688 (`hidden32`), 0.688 (`hidden64`). All within 0.002 of each other, far inside the noise floor. Going from 8 to 64 units, an 8x range, produces no measurable change, per-class too: every class's hidden_dim spread (0.003-0.026) is smaller than that same class's split-choice spread, `two_wheeler` starkest at 0.005 vs 0.386.

**Decision:** retain `HIDDEN_DIM=16`. The model isn't capacity-limited, the histogram feature representation is the ceiling, not the network's expressive power, consistent with the feature-separability-ceiling read of the `bus`/`large_vehicle` confusion in section 3.

## 7. Architecture depth: deep10 ablation (10 hidden layers)

Same features/taxonomy/N_BINS=16 as `baseline`, only depth varied: `n_hidden_layers=10` instead of 2, same `hidden_dim=16`. `scripts/mlp_variants.py`'s `deep10`, results at `results/mlp/deep10/`.

First attempt (`dropout=0.3`, no `batch_norm`, `lr=1e-5`, 180 epochs) collapsed: predicted only the majority class (`car`) for every point, macro F1 0.122, a vanishing-gradient failure from stacking 10 unnormalized layers, not overfitting. Fixed by adding `batch_norm=True` after each hidden layer, easing dropout to 0.1, and raising `lr` to 1e-3, changed together, so the fix isn't isolated to one lever.

Retrained result: macro F1 0.709 vs `baseline`'s 0.686 (+0.023). This appears to be a gain, but every per-class delta sits inside the noise floor, and `two_wheeler`/`pedestrian`, the classes actually driving the macro F1 ceiling, change negligibly (+0.002, +0.005).

**Decision:** retain the 2-hidden-layer baseline. Depth doesn't help once the network is properly trainable, same conclusion as section 6: the histogram representation is the ceiling, not model capacity in either width or depth.

**Side finding, a real caching bug:** `evaluate_val_metrics` cached its numeric metrics keyed only by file existence, not by whether the underlying model had changed. Retraining `deep10` in place with the fixed config left a stale `mlp_val_metrics.json` from the collapsed first attempt, silently reused, so the first "fixed" result reported was actually the failed run's numbers. Fixed by invalidating the cache whenever it's older than `mlp_model.pt`.

**Scale mismatch, tested: doesn't matter.** The 64 histogram-bin features are all bounded [0,1], but `doppler_spread` is unnormalized (train range 0-59.2, mean 0.233), a scale mismatch at the input layer. Tested directly with `standardized` (z-score standardizes `doppler_spread` on train-fit mean/std, applied unchanged to val/test): macro F1 0.688 vs `baseline`'s 0.686, every per-class delta within 0.004, inside the noise floor. `scripts/mlp_variants.py`'s `standardized`, results at `results/mlp/standardized/`.

**Finding:** the scale mismatch is real but harmless here, consistent with the standing hypothesis that Adam's per-parameter adaptive step size already blunts it.

## 8. Feature design: point count (n_points scalar vs raw-count histogram)

Histogram features are fraction-of-points-per-bin, which discards instance point count entirely (a 1-point and an N-point instance landing in the same bin are indistinguishable). Two ways of putting it back: `n_points` (baseline plus one extra scalar, raw instance point count appended unbinned) and `raw_counts` (baseline with `normalize=false`, unnormalized per-bin counts, so point count is recoverable by summing a feature's bins). `scripts/mlp_variants.py`'s `n_points`/`raw_counts`, results at `results/mlp/n_points/`, `results/mlp/raw_counts/`.

Macro F1: `baseline` 0.686, `n_points` 0.692, `raw_counts` 0.690, both deltas and every per-class delta inside the noise floor. Both nudge the `pedestrian`/`pedestrian_group` precision/recall balance in the theoretically expected direction, but neither shift clears noise.

**Finding:** not a real improvement, by either route. Two independent mechanisms for handing the network point count converge on the same null result.

## 9. Feature design: orientation and angular features (spatial_extent, azimuth_extent)

`x_rel`/`y_rel` are per-point and orientation-dependent (a car seen broadside vs head-on spreads very differently across the two axes for the same physical object). Two tests: `spatial_extent` drops them in favor of one orientation-robust scalar per instance (bounding-box diagonal, `sqrt(x_extent^2+y_extent^2)`), appended unbinned; `azimuth_extent` adds a new scalar on top of `baseline` instead (angular spread as seen by the sensor, max-min of `azimuth_sc`). `scripts/mlp_variants.py`'s `spatial_extent`/`azimuth_extent`, results at `results/mlp/spatial_extent/`, `results/mlp/azimuth_extent/`.

Macro F1: `baseline` 0.686, `spatial_extent` 0.690, `azimuth_extent` 0.696, every per-class delta inside the noise floor; `azimuth_extent`'s `two_wheeler` delta (+0.023) is the largest of any feature-design test so far, still a fraction of that class's 0.386 spread.

Also checked whether `spatial_extent`'s own noise floor is narrower than baseline's, since it was specifically built to remove the orientation-sensitivity mechanism thought to drive part of that noise (Summary item 3). It isn't, meaningfully: macro F1 across the same 6 splits ranged 0.655-0.745 (std 0.030) vs baseline's 0.651-0.734 (std 0.027), essentially unchanged. `large_vehicle`/`two_wheeler` spreads, the classes an orientation fix would most plausibly help, shrank slightly (0.197->0.167, 0.386->0.356) but not by a meaningful amount. Script: `scripts/split_sensitivity.py`'s `features`/`extra_features`/`output_subdir` params, results at `results/mlp/split_search_spatial_extent/fold_<n>/`.

**Finding:** sixth independent axis landing inside the noise floor, both on the fixed baseline split and, for `spatial_extent`, across the noise floor itself. Reinforces that the ceiling isn't from any one specific feature-engineering choice, it's more fundamental to what this per-instance summary vector can express at ~2.9 points/instance average.

## 10. Point count diagnostic: does sparsity explain the ceiling directly?

Sections 6-9 tested whether handing the network more information helps (six null results). This asks a more direct question: sliced the *existing* `baseline` model's val predictions by instance point count, no retraining, to check whether error is actually concentrated in sparse instances. `mlp_classifier.evaluate_by_point_count`, reproducible (loads the cached model, never retrains).

Delta columns are relative to the sparsest bucket (`n_points=1`), via `mlp_classifier.format_point_count_deltas`:

| points | n | accuracy | macro F1 | car | large_vehicle | two_wheeler | pedestrian | pedestrian_group |
|---|---|---|---|---|---|---|---|---|
| 1 | 24,004 | 0.619 | 0.381 | 0.839 | 0.037 | 0.389 | 0.641 | 0.000 |
| 2 | 17,353 | 0.741 (+0.122) | 0.606 (+0.224) | 0.856 (+0.017) | 0.237 (+0.200) | 0.516 (+0.127) | 0.747 (+0.105) | 0.673 (+0.673) |
| 3 | 11,711 | 0.800 (+0.181) | 0.702 (+0.321) | 0.881 (+0.042) | 0.471 (+0.433) | 0.597 (+0.208) | 0.770 (+0.129) | 0.791 (+0.791) |
| 4 | 6,567 | 0.821 (+0.202) | 0.762 (+0.380) | 0.878 (+0.039) | 0.687 (+0.650) | 0.652 (+0.263) | 0.768 (+0.127) | 0.823 (+0.823) |
| 5 | 3,507 | 0.815 (+0.196) | 0.764 (+0.383) | 0.847 (+0.008) | 0.833 (+0.796) | 0.658 (+0.270) | 0.652 (+0.010) | 0.830 (+0.830) |
| 6-10 | 6,348 | 0.854 (+0.236) | 0.715 (+0.334) | 0.798 (-0.041) | 0.963 (+0.926) | 0.525 (+0.136) | 0.476 (-0.165) | 0.813 (+0.813) |
| 11+ | 1,624 | 0.929 (+0.310) | 0.538 (+0.156) | 0.822 (-0.017) | 0.995 (+0.958) | 0.211 (-0.178) | 0.000 (-0.641) | 0.660 (+0.660) |

![Val performance by instance point count](results/mlp/mlp_point_count_curve.png)

A quarter of val (24,004/~71k) is single-point instances, where the model is barely better than chance in macro-F1 terms. Macro F1 roughly doubles from 1 to 5 points (0.381 -> 0.764), same trained model throughout.

`large_vehicle` is the clearest case: f1 climbs almost monotonically, 0.037 (1 point) to 0.995 (11+ points). Being `large_vehicle` is a question about size, structurally unknowable from a single point, and nearly perfect once there's enough support to gauge extent (median 7 points/instance, 62% of val `large_vehicle` instances have 6+ points). `car` is the opposite pattern, flat and high throughout (0.80-0.88), a strong enough single-point signal (RCS/Doppler) that extra points barely matter, consistent with only 6% of `car` instances ever reaching 6+ points.

`two_wheeler`, `pedestrian`, `pedestrian_group` rise then fall, peaking around 3-5 points before declining at 6-10 and 11+. `evaluate_by_point_count`'s per-class `support_<class>` column confirms the drop is a support/class-mix artifact, not a real reversal: `two_wheeler` support falls from 1,391 (bucket 1) to 242 (6-10) to 4 (11+); `pedestrian` falls from 6,318 to 49 to 0. `f1_pedestrian=0.000` at 11+ isn't the model failing, there are zero real pedestrian instances there to evaluate; `two_wheeler` at 11+ (4 instances) is statistically meaningless, not a reliable measurement. Those tail buckets shift heavily toward `large_vehicle` instead (42% and 75% of the bucket by instance count).

Reconciling with section 8's null result: that tested whether *telling* the network an instance's point count helps, it doesn't. This tests whether instances that structurally *have* more real points get classified better, and they clearly do. A count scalar doesn't add information the network can act on; more real points populating the histogram does.

**Finding:** sparsity is a real, large, and now directly demonstrated driver of the ceiling for classes with enough range of point counts to show it, `large_vehicle` above all. It doesn't act uniformly: `car` never needed the points (RCS/Doppler alone separates it at n=1), and `two_wheeler`/`pedestrian` simply don't have enough high-point instances in RadarScenes to say how they'd behave with more, the apparent drop at 6-10/11+ is near-zero support, not degraded classification. First evidence that a change to what data goes in, rather than how it's summarized, could plausibly move the needle (track-aggregation direction, Summary item 1).

## 11. Feature design: explicit per-instance statistics instead of histograms

Replaces the 65-dim histogram encoding entirely with 8 explicit per-instance statistics: `rcs` mean/median/std, `vr_compensated` mean/median (std dropped, `doppler_spread` covers that), `radial` (new: `sqrt(x_rel^2+y_rel^2)`, rotation-invariant distance from centroid) std, `azimuth_sc` std, `doppler_spread` as-is. `scripts/mlp_variants.py`'s `stat_descriptors`, results at `results/mlp/stat_descriptors/`.

Macro F1: `baseline` 0.686, `stat_descriptors` 0.658, a drop, not the usual within-noise flat line. Per-class all inside the noise floor except `pedestrian` (-0.075 vs a spread of 0.073), the first delta in this program to land just outside its noise floor rather than inside.

**Finding:** seventh axis landing at or inside the noise floor overall, but `pedestrian`'s borderline drop warrants further examination when the confusion matrix is studied directly.

## 12. Feature design: bin edge placement (mean/std range, quantile bins)

Two tests, both otherwise identical to `baseline`, only how the 16 bin edges are placed changes. `gaussian_range`: still equal-width bins, but the outer range is `[mean-2std, mean+2std]` instead of decision 2's `[1st, 99th]` percentile. `quantile_bins`: unequal-width bins, edges placed so each bin holds roughly equal training mass instead of equal range. `scripts/mlp_variants.py`'s `gaussian_range`/`quantile_bins`, results at `results/mlp/gaussian_range/` and `results/mlp/quantile_bins/`.

Macro F1: `baseline` 0.686, `gaussian_range` 0.686 (identical, tightest null in this program), `quantile_bins` 0.683, both inside noise per-class. `quantile_bins` also surfaced a real edge case: `x_rel`/`y_rel` each got one duplicate bin edge at exactly `0.0` (true by construction for every single-point instance), collapsing one of 16 bins per feature to permanently empty, 63 informative columns instead of 65. Produced no observable negative effect.

**Finding:** bin edge placement doesn't matter here, even though it plausibly could have (placement changes where a single raw value maps to, not just how points aggregate, unlike most tests in this program). Section 11's null result reinforces this from the other direction: removing binning's discretization step entirely didn't help either, so quantization loss isn't the dominant bottleneck. Consistent with section 10: the real ceiling is upstream of every encoding choice, how many real points an instance has to begin with.

## 13. Confusion matrix study: pedestrian/two_wheeler separability under sparsity

Direct diagnostic on the pedestrian/two_wheeler pair named as a remaining v1.0 item: is their confusion driven by the same sparsity ceiling section 10 found for `large_vehicle`, or something else? `separability_probe.run_probe` (class-weighted logistic regression + random forest, sequence-grouped held-out split, `[rcs, vr_compensated, x_extent, y_extent, doppler_spread]`), run separately on sparse (`n_points<=2`) and dense (`n_points>=5`) subsets. No MLP retraining, a separate sklearn probe on a hand-built feature set. `scripts/pedestrian_separability.py`'s `run_sparse_regime_probe`, results at `results/pedestrian_two_wheeler/`.

| regime | n (pedestrian / two_wheeler) | LR pairwise AUC | LR macro F1 | RF macro F1 |
|---|---|---|---|---|
| sparse (n_points <= 2) | 13,030 / 3,850 | 0.806 | 0.77 | 0.88 |
| dense (n_points >= 5) | 347 / 893 | 0.945 | 0.84 | 0.93 |

Same direction as section 10 (sparse worse than dense), but a much smaller gap than `large_vehicle`/`car` showed there: AUC drops from 0.945 to 0.806, still comfortably separable, not the near-chance collapse `large_vehicle` hit at n=1. Likely reason: `x_extent`/`y_extent` are trivially 0 at `n_points=1`, so whatever separates this pair at low point counts comes almost entirely from `rcs`/`vr_compensated`, both meaningful from a single point, unlike the size/extent signal `large_vehicle` needs several points to expose. Caveat: only 2.3% of pedestrian instances have `n_points>=5`, a small and likely non-representative subset (probably closer-range/favorable geometry).

**Finding:** pedestrian/two_wheeler confusion is not primarily a sparsity problem the way `large_vehicle`/`car`'s is, the physical signal separating them (RCS, radial velocity) survives at n=1-2. This diagnostic used a compact hand-built feature set, not the MLP's own histogram encoding, so it speaks to whether separating signal exists in the raw data, not directly to why the trained MLP itself confuses these two classes; that's still open.

## 14. Feature design: combined_features (everything, additive), the first validated positive result

Every feature-design test so far (sections 8-13) tried one representation change at a time, always inside noise. This tests something different: keep everything `baseline` already has and stack on top of it every other feature tried in this program (nothing removed), to check whether several individually-null signals compound into something detectable together. Histograms (16 bins each): `rcs`, `vr_compensated`, `x_rel`, `y_rel` (baseline's 4) plus `range_sc`, `azimuth_sc` (new, 96 dims total). Unbinned scalars: `doppler_spread` (baseline's 1) plus `n_points`, `azimuth_extent`, `spatial_extent` (already built, unused until now) and `rcs_extent`, `range_extent` (new, same max-min-per-instance recipe as `azimuth_extent`). 102 dims total vs baseline's 65. `scripts/mlp_variants.py`'s `combined_features`, results at `results/mlp/combined_features/`.

Single-split macro F1: `baseline` 0.686, `combined_features` 0.713 (+0.027), the biggest single-split gain in this program, but by itself no more trustworthy than `range_sc`'s (+0.018) or `deep10`'s (+0.023), both of which the split-sensitivity check flattened to noise. Ran the same check here (`split_sensitivity.run_split_sensitivity`, `output_subdir="split_search_combined_features"`, the same 6 splits as every other split-sensitivity check in this project, since split selection depends only on class/sequence composition, not feature choice, so this is a genuine paired fold-by-fold comparison):

| fold | baseline macro F1 | combined_features macro F1 | delta |
|---|---|---|---|
| 0 | 0.679 | 0.708 | +0.029 |
| 1 | 0.734 | 0.775 | +0.041 |
| 2 | 0.701 | 0.738 | +0.037 |
| 3 | 0.651 | 0.701 | +0.050 |
| 4 | 0.695 | 0.734 | +0.039 |
| 5 | 0.688 | 0.706 | +0.018 |

`combined_features` wins every one of the 6 folds, +0.018 to +0.050, mean +0.036. Unlike `range_sc`/`deep10`, this survives the exact check that debunked those. First and only positive, cross-validated result in the whole program.

Why this is statistically credible, not attributable to chance: if the two were truly equivalent, winning all 6 independent folds by chance has probability (1/2)^6 ≈ 1.6% (a simple sign test), a materially stronger claim than a single positive split, and the reason this meets the threshold `range_sc`/`deep10` did not (neither ever showed a 6/6 pattern, just an overlapping range once checked). Caveat: folds share the same sequence pool and fixed test split, so this is strong evidence, not a certified p-value.

Single-split per-class deltas: `car` +0.010, `large_vehicle` -0.020, `two_wheeler` +0.070, `pedestrian` +0.004, `pedestrian_group` +0.067, all individually inside their noise floors, but `two_wheeler` and `pedestrian_group` account for essentially the whole gain. Confirmed properly across the 6-fold split-sensitivity pairing for `two_wheeler` specifically (the class with by far the largest split-choice spread, 0.386): it improves in every single fold, +0.020 to +0.131 (mean +0.067), the largest and most consistent per-class gain of any class, and its own fold-to-fold spread narrows too, 0.386 (baseline) to 0.275 (`combined_features`), ~29% tighter. `large_vehicle`, the class section 10 identified as the clearest, most direct sparsity casualty, did not benefit, if anything moved slightly negative, evidence this gain is not rescuing sparse instances specifically.

Magnitude check against the sparsity ceiling (section 10): `combined_features`'s validated gain (+0.036 macro F1) is roughly a tenth the size of the sparsity effect itself (macro F1 roughly doubling, 0.381 to 0.764, from 1-point to 5-point instances, same trained model). Combined with `large_vehicle` not improving while `two_wheeler`/`pedestrian_group` drove the whole gain, this reads as real, previously-unexploited signal in the richer feature combination, not a dent in the sparsity ceiling itself.

**Finding:** the individual null results in sections 8-13 are not wrong, each correctly answered "does this one feature, alone, help" (no). This answers a different question, "do several of them together help" (yes, validated). Several small, individually-undetectable signals can sum to a detectable combined one, and unlike the earlier replacement-style tests (`spatial_extent` swapped in for `x_rel`/`y_rel`, `range_sc` swapped in for `doppler_spread`), this is purely additive, so nothing already pulling its weight had to compete for a slot. Doesn't touch the fundamental sparsity ceiling, but is a real, keepable gain, concentrated in the two classes this program's earlier tests never specifically fixed.

### 14a. Which of the seven additions is actually doing the work?

Zero-out feature importance on the already-trained `combined_features` model (single val split, no retraining): zero one feature group's columns at a time, measure the macro F1 drop from the full 102-dim model (0.7125). Larger drop implies the model leans on that group more. Fast diagnostic, not split-sensitivity-grade, single-split noise applies to the exact numbers, not the broad ranking.

| feature group | dims | macro F1 after zeroing | drop |
|---|---|---|---|
| `vr_compensated` (hist, baseline) | 16 | 0.381 | -0.331 |
| `spatial_extent` (scalar, added) | 1 | 0.632 | -0.080 |
| `rcs_extent` (scalar, added) | 1 | 0.667 | -0.045 |
| `x_rel` (hist, baseline) | 16 | 0.672 | -0.040 |
| `rcs` (hist, baseline) | 16 | 0.679 | -0.034 |
| `range_sc` (hist, added) | 16 | 0.682 | -0.030 |
| `azimuth_sc` (hist, added) | 16 | 0.685 | -0.027 |
| `y_rel` (hist, baseline) | 16 | 0.695 | -0.018 |
| `n_points` (scalar, added) | 1 | 0.696 | -0.016 |
| `range_extent` (scalar, added) | 1 | 0.706 | -0.007 |
| `doppler_spread` (scalar, baseline) | 1 | 0.706 | -0.006 |
| `azimuth_extent` (scalar, added) | 1 | 0.712 | -0.001 |

`vr_compensated`'s contribution substantially exceeds every other feature group's (raw velocity information nothing else substitutes for). Among the seven additions specifically, importance is concentrated in two, both single scalars: `spatial_extent` (-0.080) and `rcs_extent` (-0.045), each outperforming several 16-dim histogram blocks. The rest (`range_sc`, `azimuth_sc`, `n_points`, `range_extent`, `azimuth_extent`) are minor to negligible individually.

This resolves the addition-vs-replacement puzzle from section 9 with direct evidence: `spatial_extent` tested as a *replacement* for `x_rel`/`y_rel` came out null, not because it's useless, but because it's roughly as useful as what it replaced, so swapping nets out to nothing. `rcs_extent` is the more novel finding: never tested standalone before this variant, first evidence RCS's own within-instance spread carries real signal independent of its central tendency.

### 14b. Do the two most load-bearing additions reproduce the gain alone?

Direct follow-up: isolate the two standouts from 14a and test each as its own minimal addition to `baseline` (nothing else from `combined_features` included). `rcs_extent_only`: baseline's 4 histograms unchanged, `rcs_extent` added alongside `doppler_spread`. `spatial_extent_added`: baseline's `x_rel`/`y_rel` kept (unlike section 9's replacement-style test), `spatial_extent` added alongside `doppler_spread`. `scripts/mlp_variants.py`'s `rcs_extent_only`/`spatial_extent_added`, results at `results/mlp/rcs_extent_only/` and `results/mlp/spatial_extent_added/`.

Single-split macro F1: `baseline` 0.686, `rcs_extent_only` 0.690 (+0.004), `spatial_extent_added` 0.698 (+0.012). Neither approaches `combined_features`'s single-split gain (+0.027) or its validated 6-fold mean (+0.036), even summed (+0.016) the two standout individual contributors account for less than half of the full gain.

**Finding:** despite showing the largest zero-out importance inside the full 102-dim model, neither feature reproduces a meaningful share of the gain in isolation. Zero-out importance measures how much the model currently relies on a feature *given everything else it already has*, not that feature's standalone marginal value, this is the direct demonstration of the difference. The gap between +0.016 (summed individual) and +0.036 (validated combined) points toward a real interaction, several of the seven additions likely need to be present together (`range_sc`/`azimuth_sc`'s own moderate zero-out importance are plausible other contributors) rather than the gain being attributable to any one or two features alone. Neither single-addition result clears the noise floor on a single split, so not run through split-sensitivity, not worth the additional compute given how far both results are from being worth validating.

## 15. two_wheeler fold instability: pairwise KS diagnostic and mechanism

Section 10 established sparsity as the ceiling; separately, `two_wheeler`'s fold-to-fold F1 spread (0.386, the widest of any class, Model config) has stood unexplained since Summary item 3. Direct diagnostic: per-instance (not per-point pooled, which would let busy instances dominate) pairwise Kolmogorov-Smirnov statistic between every pair of the 6 split-sensitivity folds, on `rcs`, `vr_compensated`, `x_rel`, `y_rel`, `spatial_extent`, `doppler_spread`, for `two_wheeler` and `large_vehicle` (comparable sample size, much smaller F1 spread at 0.197, ruling out "just less data" as the explanation). `scripts/fold_stability.py`, results at `results/fold_stability/`.

| feature | two_wheeler mean KS | large_vehicle mean KS |
|---|---|---|
| `vr_compensated` | 0.370 | 0.240 |
| `spatial_extent` | 0.090 | 0.205 |
| `rcs` | 0.086 | 0.108 |
| `doppler_spread` | 0.078 | 0.106 |
| `y_rel` | 0.049 | 0.065 |
| `x_rel` | 0.045 | 0.047 |

`vr_compensated` is the largest value in the table and the only one cleanly asymmetric toward `two_wheeler`. `spatial_extent` is the mirror case, large and asymmetric toward `large_vehicle` instead, which rules out "any large per-feature KS instability explains F1 variance" as a general rule, `large_vehicle`'s F1 spread stays small despite it. The other four features are small and comparable between classes.

**Mechanism, two factors compounding:**

1. `two_wheeler` instances are overwhelmingly sparse (57.7% at `n_points<=2`, vs. 18% for `large_vehicle`, section 10). A per-instance mean is a weighted average over its own points; at n=1-2 that average is close to raw single-point measurement noise, at n=6+ (most of `large_vehicle`) it's already averaged down. Which sparse instances happen to land in which fold swings that fold's whole `vr_compensated` distribution, hence the large cross-fold KS.
2. `vr_compensated` is the single most load-bearing feature in the model, by a wide margin (zero-out drop -0.331, section 14a, next nearest -0.080). Instability in the feature the network relies on hardest translates directly into F1 instability; `large_vehicle`'s analogous unstable feature (`spatial_extent`) has an order of magnitude less zero-out importance, so its instability barely moves F1 regardless of source.

**Finding:** `two_wheeler`'s outsized F1 variance isn't a separate phenomenon from the sparsity ceiling (section 10), it's the same root cause (sparsity) acting through a second path, noisier per-instance estimates of the feature the model depends on most, not just fewer points to classify from. `large_vehicle`'s own unstable feature (`spatial_extent`) wasn't traced to a cause, not needed to explain the F1 asymmetry since section 14a already shows the model barely relies on it either way.

## 16. no_vr_compensated: causal confirmation of section 15's mechanism

Direct test of section 15's causal claim: retrain `baseline`'s exact config with `vr_compensated` removed entirely (features: `rcs`, `x_rel`, `y_rel`; `doppler_spread` unchanged), across the same 6 split-sensitivity folds. `MLP_CONFIG.json`'s `no_vr_compensated` variant, results at `results/mlp/no_vr_compensated/` and `results/mlp/split_search_no_vr_compensated/fold_<n>/`.

`two_wheeler` F1 by fold: 0.324, 0.458, 0.438, 0.422, 0.353, 0.336.

| variant | two_wheeler spread | two_wheeler mean F1 | relative spread |
|---|---|---|---|
| `baseline` | 0.386 | 0.571 | 0.676 |
| `combined_features` | 0.275 | 0.638 | 0.431 |
| `no_vr_compensated` | 0.133 | 0.388 | 0.343 |

Absolute `two_wheeler` F1 collapsed as expected, `vr_compensated` really does carry real signal (section 14a's zero-out drop already showed this), but the spread narrowed too, both raw (0.386 -> 0.133) and relative to its own lower mean (0.676 -> 0.343), so this isn't just a floor-effect artifact of the lower baseline. `large_vehicle`'s spread in this same run (0.175, mean 0.726, relative 0.242) stayed essentially at its own baseline level (spread 0.197, mean 0.756, relative 0.26), consistent with it never having depended on `vr_compensated` much.

**Finding:** removing the one feature identified as both unstable-for-`two_wheeler` (section 15's KS table) and the model's single most load-bearing feature overall (section 14a) genuinely narrows `two_wheeler`'s fold-to-fold F1 spread. A real causal result, not just the correlational KS table alone. `large_vehicle`, whose instability traces to a different, low-leverage feature (`spatial_extent`), is unaffected, as expected.

## 17. two_wheeler bimodality: bicycle vs motorized_two_wheeler

Sections 15-16 established `vr_compensated` as `two_wheeler`'s single most load-bearing and most cross-fold-unstable feature. `two_wheeler` merges two physically different velocity regimes (`bicycle`, `motorized_two_wheeler`, `dataloader.CLASS_GROUPS`), a slow bicycle and a much faster moped don't share one velocity signature, so that reliance is worth checking directly rather than assuming it's on a coherent signal. `scripts/two_wheeler_bimodality.py`, results at `results/two_wheeler_bimodality/`.

**Bimodality check:** 1 vs 2-component Gaussian mixture on per-instance (median) `vr_compensated`, pooled across both sub-labels (34,498 instances total, full dataset). 2 components wins on BIC (199,072 vs 201,321 for 1 component), cluster means 1.19 and 3.75 m/s.

The histogram complicates a clean "bicycle vs motorized_two_wheeler" read, though. `motorized_two_wheeler` is only 4.7% of `two_wheeler` instances (1,617 of 34,498), far too small a share to be the dominant driver of the pooled distribution's shape, which stays dominated by `bicycle`'s own right-skewed shape (peak around 6-7 m/s). Whatever structure the GMM is picking up in the pooled data is more likely within `bicycle` itself (a smaller bump near -2.5 m/s alongside the main peak) than a bicycle/moped split.

`motorized_two_wheeler`'s own distribution, in isolation, is strikingly multimodal on its own though: a sharp spike right at 0 m/s (idling/stopped, the tallest bar of any panel in the figure), a separate broad hump around 3-7 m/s (moving), and a smaller tail near 10 m/s. Structurally very different from `bicycle`'s shape, even though it's a small share of instances.

**Separability probe:** LR + RF (`PROBE_FEATURES`, same machinery as the bus/large_vehicle/truck taxonomy decision), `bicycle` vs `motorized_two_wheeler`. Pairwise AUC: 0.665 (LR), 0.698 (RF), well short of the strong separability that would say these are two cleanly distinguishable populations on this feature set. Caveat: severe class imbalance in the held-out split (298 `motorized_two_wheeler` vs 6,494 `bicycle`, ~22:1), a small and likely noisy sample to trust a precise AUC from.

**Finding:** mixed, not a clean confirmation. The pooled distribution technically prefers 2 GMM components, but that's not well explained by a simple bicycle/moped split, `motorized_two_wheeler` is too rare to dominate the pooled shape, and the full-feature separability probe found only modest separability between the two sub-labels, not the strong separation a "two incompatible velocity regimes" story would predict. `motorized_two_wheeler`'s own distribution is genuinely multimodal in isolation though, so a fold that happens to draw a different mix of it could still shift `vr_compensated`'s cross-fold KS somewhat, but this diagnostic doesn't support that being the primary driver of section 15's instability the way sparsity/averaging noise does. One correction to how this was framed in discussion: `vr_compensated` correctly capturing that a bicycle and a moped move at different speeds isn't a data problem, the raw measurements are accurate. What's questionable is the taxonomy choice to merge two velocity-heterogeneous physical classes under one label, that's a modeling decision, not a defect in the data.

**Why the AUC lands where it does, not higher:** aggregate shape difference and per-instance separability are different questions, the histogram answers the first, the probe the second. `bicycle` peaks around 6-7 m/s; `motorized_two_wheeler`'s "moving" hump sits at 3-7 m/s, directly overlapping it. What makes `motorized_two_wheeler` look visually distinct is the 0 m/s idling spike and the ~10 m/s tail, both real but both minority mass, most instances of both sub-labels sit in that shared 3-7 m/s middle, and a classifier has to get every instance right, not just the distinctive tails. An AUC of 0.665-0.698 fits that: clearly above chance from the tails, well short of the ~0.9+ a small-overlap case would produce. The RF number is also a ceiling, not a floor, it had every `PROBE_FEATURES` column available at once, not `vr_compensated` alone, so `vr_compensated` in isolation contributes no more separability than that. And don't over-read precision into 0.698 itself, 298 `motorized_two_wheeler` against 6,494 `bicycle` in the held-out split is a small, noisy sample for a confident point estimate.

This does not confirm the merge is safe. `two_wheeler` is already merged, that's the existing taxonomy, this section tested whether that merge explains section 15's instability, not whether to merge in the first place. The result was mixed, not a clean rejection of the mechanism, so modest separability here isn't the green light the bus/large_vehicle precedent's poor separability was, especially with the same small/imbalanced sample making the AUC itself uncertain.

**Candidate next test, not yet run:** a full retrain with `bicycle`/`motorized_two_wheeler` split into separate classes would be confounded before it started, splitting out a class that's 4.7% of `two_wheeler` creates a class an order of magnitude sparser than anything currently in the taxonomy, in a fold structure that already struggles to distribute `two_wheeler` evenly (finding 3). Any F1 swings would likely reflect that new sparsity problem, not the original bimodality question. A more surgical, unconfounded version reuses the existing 6 split-sensitivity folds directly: compute each fold's `bicycle`:`motorized_two_wheeler` mix among `two_wheeler` val instances, and check whether folds with a more `motorized_two_wheeler`-heavy mix are the ones with outlier `vr_compensated` KS or F1, a direct test of the composition-shift mechanism this section flagged but didn't check.

## 18. Fold composition check: does motorized_two_wheeler mix explain the KS instability

Ran section 17's proposed surgical test: per fold (the same 6 split-sensitivity folds), `motorized_two_wheeler` share of `two_wheeler` val instances, against that fold's `vr_compensated` KS-to-other-folds and baseline `two_wheeler` F1. `scripts/two_wheeler_bimodality.py`'s `fold_composition_vs_instability`, results at `results/two_wheeler_bimodality/fold_composition_vs_instability.csv`.

Spearman (n=6): `motorized_share` vs `mean_ks_to_other_folds` = -0.486, `motorized_share` vs `two_wheeler_f1` = -0.543, `mean_ks_to_other_folds` vs `two_wheeler_f1` = -0.086. The one fold with essentially no `motorized_two_wheeler` (fold 1, 0.2% share) is both the biggest KS outlier and the best F1, the opposite of the predicted direction. Dropping that single outlier fold collapses `motorized_share` vs `two_wheeler_f1` from -0.543 to -0.2, and the F1 sequence across the other 5 folds shows no ordering by `motorized_share` at all, so the correlation is mostly one fold, not a trend.

**Finding:** does not corroborate the composition-shift mechanism, leans against it. `mean_ks_to_other_folds` vs `two_wheeler_f1`, the relationship the mechanism actually needs, is flat. Combined with section 17's mixed bimodality/separability results, three independent checks now give no evidence-based reason to split `motorized_two_wheeler` out of `two_wheeler` or drop it; sparsity acting through `vr_compensated` (sections 15-16) remains the better-supported explanation for the instability. Current taxonomy (merged) stays as is.

## 19. pedestrian/two_wheeler on the MLP's actual histogram encoding

Section 13 found strong separability (LR pairwise AUC 0.806 sparse, 0.945 dense) between `pedestrian` and `two_wheeler` on a hand-built 5-feature set (`PROBE_FEATURES`: `rcs`, `vr_compensated`, `x_extent`, `y_extent`, `doppler_spread`), but explicitly flagged that this speaks to whether separating signal exists in the raw data, not to why the trained MLP itself confuses this pair, since the MLP never sees that feature set, it sees the 65-dim per-instance histogram encoding (`histogram_separability.build_histogram_features`, `N_BINS=16` per `rcs`/`vr_compensated`/`x_rel`/`y_rel` plus unbinned `doppler_spread`). This section closes that gap: same sparse (`n_points<=2`)/dense (`n_points>=5`) regimes, same `separability_probe.run_probe` machinery, restricted to `pedestrian`/`two_wheeler` only, but on the real 65-dim encoding instead of the hand-built set. `scripts/pedestrian_separability.py`'s `run_encoded_regime_probe`, results at `results/pedestrian_two_wheeler/`. Bin edges fit per regime from that regime's own two-class pool (no separate train split here, same discipline as `histogram_separability.run_bin_sweep`'s own CV probes).

| regime | model | hand-built AUC (section 13) | encoded (65-dim) AUC |
|---|---|---|---|
| sparse (n=65,237 pedestrian / 19,188 two_wheeler) | LR | 0.806 | 0.919 |
| sparse | RF | 0.921 | 0.919 |
| dense (n=1,788 pedestrian / 4,412 two_wheeler) | LR | 0.945 | 0.971 |
| dense | RF | 0.981 | 0.985 |

The hand-built LR numbers were rerun alongside the encoded ones and reproduce section 13 exactly (0.806/0.945), confirming this is a clean apples-to-apples rerun, not a different sample. The encoded representation is at least as separable as the hand-built proxy, higher via LR at both regimes, essentially tied via RF, so binning/normalizing into the 65-dim encoding does not destroy the separating signal section 13 found. That specific concern (the encoding itself being the bottleneck) is ruled out.

- Regime-level recall gap, read directly off the confusion matrices: `two_wheeler` recall was 0.76 (LR and RF) in the sparse regime, ~24% confused as `pedestrian`, vs. 0.89 (LR) to 0.97 (RF) in dense, ~3-11%. Two different explanations would produce this same pattern and haven't been distinguished yet: `x_rel`/`y_rel` becoming informative once there are enough points to expose real spatial extent (near-degenerate at `n_points=1-2`, per section 13's own caveat), or `rcs`/`vr_compensated` simply getting less noisy at higher point counts, the mechanism sections 15-16 already established for `two_wheeler`. Recall alone can't tell these apart, per-feature (grouped by bin) permutation importance on this probe, split by regime, would.

- Grouped permutation importance (`run_encoded_permutation_importance`, all `N_BINS=16` bins of one point-level feature shuffled together, same held-out split as the table above, so `baseline_auc` reproduces it exactly) answers this. Sparse (RF): `vr_compensated` mean AUC drop 0.320, `x_rel` 0.023, `y_rel` 0.017, `rcs` 0.016, `doppler_spread` 0.009, `vr_compensated` alone accounts for essentially all of the ~0.42 gap above chance. Dense (RF): `y_rel` 0.191 (now the top feature), `vr_compensated` 0.112, `x_rel` 0.065, `rcs`/`doppler_spread` ~0. So `x_rel`/`y_rel` were not the answer in the sparse regime, `vr_compensated` alone carries it there; `y_rel` switches on as an additional strong signal in dense, on top of `vr_compensated`, not in place of it. `rcs` contributes almost nothing in either regime, correcting section 13's original assumption that sparse separation came from "rcs/vr_compensated" jointly.

- `rcs`'s near-zero importance: genuine overlap, not redundancy with a correlated feature. `feature_distribution_ks` runs a model-free two-sample KS test per point-level feature, pooling all points per regime (matching exactly what `build_histogram_features` bins), `results/pedestrian_two_wheeler/feature_ecdfs.png`. KS statistic: `rcs` 0.070 (sparse) / 0.072 (dense), by far the smallest of the four, versus `vr_compensated` 0.593 / 0.346, `x_rel` 0.190 / 0.178, `y_rel` 0.083 / 0.126. (p-values are all 0.0 and not informative here, n is in the tens of thousands per class, so the KS null collapses regardless of effect size, the statistic itself is what matters, not significance.) `rcs_position_correlation` checked the alternative explanation directly, point-level Pearson/Spearman correlation between `rcs` and `x_rel`/`y_rel`/`vr_compensated`/`doppler_spread`/`azimuth_sc`, per class, nothing exceeds |r|=0.23 (`azimuth_sc`, pedestrian only), most under 0.15. No candidate feature is correlated with `rcs` strongly enough to be silently carrying its information. `rcs` just doesn't separate this pair, independent of the model.

- Why ~23-24% of `two_wheeler` is still confused as `pedestrian` despite `vr_compensated` being the dominant, genuinely informative feature: two things, and one of them corrects the framing two bullets up. First, `vr_compensated`'s own KS (0.593 sparse) is the largest of any feature but far short of 1.0, real, substantial overlap remains, an idling or tangentially-moving two_wheeler reads near-zero radial velocity, indistinguishable from a pedestrian on this feature alone; the best available feature isn't a perfect one. Second, the "cheap LR/RF does far better than the trained MLP" comparison drawn from the AUC table above conflated AUC with recall, they aren't the same thing. The sparse-regime probe's own two_wheeler recall, at its actual class-weighted operating threshold, was 0.76 for both LR and RF, ~24% misclassified as `pedestrian`, despite an AUC of 0.919. That's within the same range as the real MLP's own conditional accuracy when restricted to "predicted `two_wheeler` or `pedestrian`" only (0.54/(0.54+0.20) = 0.73, from the standing val confusion matrix). A high AUC doesn't promise a clean decision boundary, so there isn't the gap between probe and trained model this section previously implied, once recall is compared to recall instead of to AUC.

- Why the confusion is one-directional (`two_wheeler`->`pedestrian` 20-24%, but `pedestrian`->`two_wheeler` only 1-2%, `pedestrian` recall 0.94): the `vr_compensated` ECDF (`feature_ecdfs.png`) shows containment, not a symmetric overlap. `pedestrian`'s curve is a single tight step, almost all its mass in a narrow near-zero band; `two_wheeler`'s is wider, real mass near zero too (idling/stopped, or moving tangentially so the radial component reads near zero, matching section 17's bimodality check: `motorized_two_wheeler` spikes at 0 m/s idling, `bicycle` has a smaller bump near -2.5 m/s), but keeps climbing to much higher values before reaching 1.0. `pedestrian` has essentially no mass in `two_wheeler`'s characteristic higher-speed range, so there's nothing on the `pedestrian` side to get pulled the other way. `vr_local_density_ratio` makes this precise, not just qualitative: raw point counts inside `abs(vr_compensated) < threshold`, sparse regime, `pedestrian` outnumbers `two_wheeler` ~12.4-12.7:1 in that band (vs. 3.05:1 overall), a much sharper local skew than the global class imbalance, one a single class weight can't correct since it's local to one feature region, not the whole distribution. That's the actual mechanism: at `vr≈0`, "pedestrian" is simply the more likely explanation by raw local count, so a true `two_wheeler` sitting there gets outvoted, while a true `pedestrian` there is exactly where its own class is most concentrated, so nothing pulls it the other way.

- Dense regime tells a different, instructive story: same band, local ratio flips to ~1:1 (0.72-1.07:1), not because the skew reversed but because the base rate itself flips (`two_wheeler` outnumbers `pedestrian` 2.67:1 in dense overall, pedestrians rarely produce 5+ points to begin with) while `pedestrian` still holds a residual, weaker concentration advantage on top of that reversed base rate. `vr_compensated` alone is close to uninformative for a dense near-zero instance either way. That both explains and is explained by the permutation importance ranking (section 19, three bullets up): `y_rel` overtakes `vr_compensated` as the top feature in dense (0.191 vs. 0.112) precisely because dense instances have enough points for `y_rel` to carry real shape signal, giving the model a fallback exactly where velocity stops discriminating. Not yet checked: whether the ~3% of dense `two_wheeler` RF still gets wrong are themselves concentrated at `vr≈0`, the residual case even `y_rel` can't rescue.

## 20. Ceiling vs. fixable: does an independent model actually recover the MLP's errors

Section 19's AUC-vs-recall correction still left an open question: is `two_wheeler`'s error a genuine data ceiling (any reasonable classifier would also get these instances wrong) or a real MLP-specific gap (an independent model gets them right)? Tested directly, pooled across the same 6 split-sensitivity folds (not the single standing split, `two_wheeler` has this project's largest fold-to-fold instability). `scripts/pedestrian_separability.py`'s `mlp_probe_agreement` and `mlp_vs_rf_multiclass`.

**Test A, binary probe agreement:** per fold, load that fold's already-cached baseline MLP (`results/mlp/split_search/fold_<n>`, no retraining), take its actual val predictions for true `two_wheeler` instances, and score those same instances with an independently-refit LR/RF probe (2-class, trained on that fold's train sequences only, no leakage). Pooled: 24,032 `two_wheeler` val instances the MLP called either `two_wheeler` (20,294) or `pedestrian` (3,738).

On the MLP's correct calls, the probe agrees 99.0-99.2% (mean `P(pedestrian)` 0.03-0.04), a clean sanity check. On the MLP's actual errors, the probe *also* says `pedestrian` 71.5% (RF) to 78.8% (LR) of the time (mean `P(pedestrian)` 0.66-0.69), most of the MLP's mistakes are ones an independent model on the same features makes too, real evidence for a genuine ceiling on the majority of the error.

The remaining 21.2-28.5% disagreement looked "fixable" at first, but only checking one direction (does the probe read `pedestrian` at the same rate) isn't enough to claim the probe is actually better overall, a model could disagree on different instances without being more accurate. Full paired comparison (McNemar, same instances, both directions): RF gets 1,064 of the MLP's errors right against only 195 new mistakes on the MLP's correct calls (net accuracy on this population: MLP 84.45%, RF 88.06%, p≈4e-145); LR similarly (791 vs. 168, MLP 84.45% vs. LR 87.04%, p≈3e-97). Not a wash, a real, one-sided, statistically overwhelming improvement, on this specific binary decision.

**Test B, the fair version:** Test A's probe only ever chooses between `two_wheeler` and `pedestrian`, an easier task than the MLP's real 5-way job. Fits a 5-class RandomForest on the *exact* same 65-dim histogram encoding the MLP trains on (same per-fold edges, same features, same fold's train sequences), so the only thing that differs is model family, not task difficulty. First time this project has compared a different model family on the same features (sections 6-7 only ever varied the MLP's own capacity/depth).

Result reverses Test A's headline: `two_wheeler` recall is *worse* for RF (0.662) than the MLP (0.686), and macro recall is comparable-to-slightly-worse (RF 0.718 vs. MLP 0.728), both well inside this project's own split-choice noise floor (0.651-0.734 macro F1). RF does replicate the specific `two_wheeler`->`pedestrian` improvement from Test A (0.126 -> 0.109), but it costs more elsewhere: `two_wheeler`->`car` worsens (0.054 -> 0.075), `two_wheeler`->`pedestrian_group` worsens (0.133 -> 0.151), and `pedestrian`'s own recall drops sharply (0.886 -> 0.779).

**Finding:** the two tests aren't contradictory, they're answering different questions at different scopes. Test A shows real, statistically solid evidence that *some* of the MLP's errors on this one specific binary decision are recoverable by a model with different inductive biases, not purely a data ceiling. Test B shows that recovering them this way isn't free, RF's different biases trade one confusion for others, netting out to no real overall improvement. "Swap the model family" doesn't fix `two_wheeler`, whatever the MLP's decision boundary costs it on this one confusion, it's plausibly buying back elsewhere (most visibly `pedestrian`'s recall). Consistent with, and now a second, distinct line of evidence for, Summary item 1: the ceiling here is the data/features, not the model, capacity changes (sections 6-7) and now model family both land on the same conclusion.

## 21. Confusion matrix study: car/large_vehicle separability and why the real model actually errs

Same confusion-matrix-driven investigation as sections 13/19/20, applied to the other vehicle-cluster pair (`car` 73% correct, 10% confused as `large_vehicle`, Summary section 2). `scripts/car_large_vehicle_separability.py`, results at `results/car_large_vehicle/`. No MLP-vs-RF fair comparison for this pair (out of scope, unlike section 20).

- **Separability probe** (`run_sparse_regime_probe`/`run_encoded_regime_probe`, sparse n<=2 / dense n>=5): sparse is near chance (macro F1 0.43-0.49, AUC 0.61), dense is near perfect (macro F1 0.94, AUC 0.978-0.980), a far bigger sparse/dense gap than pedestrian/two_wheeler ever showed.
- **Permutation importance** (`run_encoded_permutation_importance`, encoded probe): dense is driven almost entirely by `x_rel` (mean_drop 0.30-0.33), `rcs` a distant second (0.045); sparse is led by `rcs` (0.054-0.068) with `x_rel` close behind (0.043-0.047). `vr_compensated`/`doppler_spread` near zero everywhere.
- **KS test** (`feature_distribution_ks`): dense's dominant feature is the more separated one (`x_rel` KS=0.277 vs sparse's best, `rcs` KS=0.125), consistent with the AUC gap above.
- **But the real, jointly-trained MLP's own error rate runs the other way** (`car_misclassification_rcs_profile`, no retraining, real val predictions): car->large_vehicle rate is 5.3% sparse, 7.9% mid, 9.2% dense, worse in the regime the probe says is nearly perfectly separable. The real model's own permutation importance, restricted to true car/large_vehicle val instances (`real_model_pairwise_permutation_importance`), replicates the probe's feature ranking exactly (`rcs` leads sparse 0.101, `x_rel` takes over by mid 0.120 and dominates dense 0.368), so the mismatch isn't about which features the real model uses, it's that probe AUC doesn't translate into real error rate once one set of bin edges and one 16-unit hidden layer have to serve all 5 classes and both regimes at once, a probe gets a dedicated model, the real MLP never does.
- **Raw feature profile on the real errors** (`real_model_feature_profile`): `rcs` shows the classic overlap pattern in mid/dense (misclassified sits between car_correct and large_vehicle_true) but an inverted, "overshoots both" pattern in sparse (misclassified median 9.06, above even car_correct's own p75). `x_extent` (length) of misclassified cars sits close to car_correct's own value in every regime, not large_vehicle's, length overlap does not explain the confusion despite `x_rel` topping the importance ranking. `y_extent` (width) and `spatial_extent` (orientation-robust diagonal) tell the real story: misclassified cars are anomalously wide/large relative to their own class, overshooting even large_vehicle's typical value in sparse/mid, and modestly elevated but still far short of large_vehicle's real size in dense.
- **Confidence margin** (`real_model_confidence_margin`): misclassified cars carry roughly half the softmax margin of car_correct/large_vehicle_true at every regime (median 0.15-0.30 vs 0.47-1.0), genuine low-confidence, borderline calls, not a confident shortcut or a bug signature.

**Finding:** no single clean feature-overlap story explains this confusion, and the isolated probe's separability ranking across regimes is not a reliable predictor of the real model's error-rate ranking (dense/sparse point the opposite direction). The mechanism that does hold up is an anomalously wide car footprint tipping a genuinely uncertain, low-margin decision, concentrated proportionally in the dense regime precisely because that is where the shared model leans hardest on `x_rel`, and a probe-vs-real-model capacity/edge-sharing gap, not raw feature overlap, is the best-supported explanation for why dense fares worse than sparse despite being far more separable in isolation.

See Summary at the top for the ranked takeaways, this log ends here.
