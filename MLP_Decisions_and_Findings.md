## Model config

**Architecture:** 3 layer MLP (2 hidden layers of 16 units each, ReLU), input 65 features, output 5 classes (`MLP_CLASSES`: `car`, `large_vehicle`, `two_wheeler`, `pedestrian`, `pedestrian_group`; bus merged into large_vehicle per decision 1's final resolution).

**Feature design:** per instance histogram encoding. `rcs`, `vr_compensated`, `x_rel`, `y_rel` each binned into N_BINS=16 fraction of points per bin columns (64 dims total), plus `doppler_spread` appended unbinned (1 dim), for 65 dims total. Bin edges use the 1st to 99th percentile range, fit on train only and reused unchanged for val/test.

**Training:** Adam optimizer, class weighted cross entropy loss (weight = max class count / class count). LEARNING_RATE=4e-5, EPOCHS=50, BATCH_SIZE=128, RANDOM_STATE=0.

**Data:** fixed sequence grouped train/val/test split (`Design_Decisions.md` decision 5).

## 1. Batch size and learning rate

Batch size affects whether rare classes contribute to each gradient update. bus makes up only 1.9% of the 358,210 train instances (vs. 42.8% for car); at that frequency, bus is absent from 54% of batches at bs=32 and 30% at bs=64.

**Decision:** use batch_size=128, the smallest candidate keeping every class's miss probability below 10%.

Scale the learning rate accordingly:

```
lr = 1e-5 × (128 / 32) = 4e-5
```

This follows the linear scaling rule and avoids slowing training through both larger batches and a reduced learning rate.

See `scripts/batch_size_selection.py`.

## 2. Epoch count: no gain past roughly epoch 200

Increasing training from 50 to 1000 epochs produced only a small macro F1 improvement (0.611 to 0.626 in the original 6 class taxonomy) and mostly added training cost and validation noise, validation accuracy plateaued after roughly 150 to 200 epochs. bus specifically got slightly worse with more training while most other classes improved, a gradient starvation signature (bus remained poorly represented in individual batches) that is now moot, since bus was merged into large_vehicle (see `Design_Decisions.md` decision 1).

**Decision:** retain EPOCHS=50.

## 3. Confusion matrix findings (epochs=50, original 6 class taxonomy, `results/mlp/bus_separate/mlp_confusion_matrix.png`)

Row-normalized, val set:

- `car`: 73% correct - the most accurate label, and the majority class. 10% confused as `large_vehicle`.
- `large_vehicle`: confused as `car` and `bus`.
- `bus`: confused as `large_vehicle`.
- `two_wheeler`: confused as `pedestrian` and `pedestrian_group`.
- `pedestrian`: recall high.
- `pedestrian_group`: confused a lot into `pedestrian` as expected, given point sparsity (an isolated pedestrian and a sparse group can look similar with few points to work with).

**Cross-check against section 2 above:** the confusion isn't scattered, it clusters within car/large_vehicle/bus (all "vehicle" classes) and separately within pedestrian/pedestrian_group, rather than spreading across unrelated classes. That pattern reads more like a feature-separability ceiling (the histogram features can't fully tell `bus` and `large_vehicle` apart) than pure gradient starvation. It's also in tension with the starvation story that `bus`'s recall (71%) is *higher* than `two_wheeler`'s (52%), despite `bus` being the far rarer class (1.9% vs 7.0% of train). If rarity/starvation were the dominant driver, that ordering would likely be reversed. This is what motivated the taxonomy experiment below.

## 4. Class taxonomy experiment (bus merged vs `bus_separate` vs `truck_separate`, 50 epochs each)

Macro F1: `bus_separate` (original 6 classes) 0.611, bus merged (5 classes, now `baseline`) 0.686, `truck_separate` (7 classes) 0.541.

Bus merged: folding `bus` into `large_vehicle` raises the combined class to F1=0.756, above either `bus` (0.543) or `large_vehicle` (0.492) alone in `bus_separate`. Supports merging `bus` in too, extending `Design_Decisions.md` decision 1's logic. This variant is now `mlp_variants.py`'s `baseline`, and its results live at `results/mlp/` directly rather than under the experiment subfolder.

`truck_separate`: splitting `truck` back out leaves pure `large_vehicle` at only 1,292 train instances, and it becomes nearly unusable (F1=0.098). Confirms decision 1's `truck` merge was correct.

Scripts: `scripts/class_taxonomy_experiment.py`, `scripts/mlp_variants.py`. Results: `results/mlp/` (baseline), `results/mlp/bus_separate/`, `results/mlp/class_taxonomy_experiment/truck_separate/`.

This directly informed `Design_Decisions.md` decision 1's final resolution: merge `bus` into `large_vehicle` too.

## 5. Feature swap: range_sc in place of doppler_spread, and a val split caution

Same architecture/training config as `baseline`, but `range_sc` (per point sensor range) binned into the same N_BINS=16 columns used for `rcs`/`vr_compensated`/`x_rel`/`y_rel`, instead of `doppler_spread` appended unbinned. `scripts/mlp_variants.py`'s `range_sc` variant, results at `results/mlp/range_sc/`.

Macro F1: `baseline` (doppler_spread) 0.686, `range_sc` 0.704. `range_sc` wins on `two_wheeler`, `pedestrian`, `pedestrian_group`, loses a little on `car` and `large_vehicle`.

This was tried once before independently (same features, same model, same bins, same weighting, 20 epochs instead of 50), with a different result: `two_wheeler` and `pedestrian` were the strongest classes (91.0%, 89.4% recall) and `large_vehicle`/`pedestrian_group` the weakest (55.3%, 54.3%), close to the opposite ranking of the current run. Comparing the two runs' confusion matrices directly:

| confusion | old run | current baseline | current range_sc |
|---|---|---|---|
| pedestrian_group -> pedestrian | 40.5% (8726/21562) | 40.1% (7165/17858) | 29.3% (5232/17858) |
| large_vehicle -> car | 41.0% (1825/4456) | 14.0% (876/6247) | 14.4% (897/6247) |

`pedestrian_group -> pedestrian` replicates closely (40.5% vs 40.1%), and `range_sc` measurably reduces it (29.3%), consistent with its `pedestrian_group` F1 gain. `large_vehicle -> car` does not replicate at all: 41.0% in the old run vs 14.0-14.4% now, a 3x gap present in both current variants, so not explained by range_sc itself.

The val class counts don't match either (`pedestrian_group`: 21,562 old vs 17,858 now; `large_vehicle`: 4,456 vs 6,247 now), meaning the old run used a different val set entirely, not the same fixed split with different randomness. This lines up with the already-established finding that `large_vehicle` has noisy cross-fold metrics driven by sequence coverage, not just instance count (`Design_Decisions.md` decision 1, `taxonomy_separability.py`): different sequences landing in val are enough to swing one class's recall by 20+ points and its dominant confusion by 3x. Two runs are only comparable if they share the exact same split, which is the whole reason `sequence_split.py`'s fixed split exists.

**Open question, not yet resolved:** does `range_sc`'s F1 gain reflect a real radar signal, or a range/point-sparsity shortcut (sparse points at long range predicted as `large_vehicle`/`car`, a failure mode observed independently in the older run)? Not yet tested. Candidate checks: per-class `range_sc` percentiles (is the confound even present in the data), a permutation test on the `range_sc` columns in val, or a counterfactual swap (overwrite `range_sc` on real pedestrian instances with typical `large_vehicle` values, check if the prediction flips). `range_sc` should not be treated as the better feature until one of these is actually run.

## 6. Architecture capacity: hidden_dim ablation

Is the 16-unit hidden layer a bottleneck, or does the histogram feature representation cap performance regardless of model size? Same architecture shape (2 hidden layers), same features/taxonomy/batch size/LR/epochs as `baseline`, only `hidden_dim` varied: 8, 32, 64, against the standing 16. `scripts/mlp_variants.py`'s `hidden8`/`hidden32`/`hidden64`, results at `results/mlp/hidden8/`, `results/mlp/hidden32/`, `results/mlp/hidden64/`.

Macro F1: 0.688 (`hidden8`), 0.686 (`baseline`, 16), 0.688 (`hidden32`), 0.688 (`hidden64`). All within 0.002 of each other, far inside the measured noise floor (std 0.027, section 5). Going from 8 to 64 units, an 8x range, moves nothing.

**Decision:** retain `HIDDEN_DIM=16`. The model isn't capacity-limited, the histogram feature representation is the ceiling, not the network's expressive power, consistent with the feature-separability-ceiling read of the `bus`/`large_vehicle` confusion in section 3.

Checked per-class too, in case the macro average was hiding a real trade-off. It wasn't: every class's hidden_dim spread (0.003-0.026) is smaller than that same class's spread from split choice alone (0.056-0.386, measured in the split sensitivity check). `pedestrian_group` has the largest capacity spread (0.026) but it's still well inside its own split-choice spread (0.117); `two_wheeler` is the starkest, 0.005 from capacity against 0.386 from split choice. No class benefits from more or less capacity.

## 7. Architecture depth: deep10 ablation (10 hidden layers)

Same features/taxonomy/N_BINS=16 as `baseline`, only depth varied: `n_hidden_layers=10` instead of 2, same `hidden_dim=16`. `scripts/mlp_variants.py`'s `deep10`, results at `results/mlp/deep10/`.

First attempt (`dropout=0.3`, no `batch_norm`, `lr=1e-5`, 180 epochs) collapsed: predicted only the majority class (`car`) for every point, macro F1 0.122. Train and val failed together, train loss froze near ln(5) and train accuracy also stayed low, the signature of gradient signal never propagating through 10 unnormalized layers rather than overfitting. Compounded by dropout applied at every one of the 10 layers (effective survival probability ~0.7^10 ≈ 3%) and a learning rate too small to make progress regardless.

Fixed by adding `batch_norm=True` after each hidden layer, easing dropout to 0.1, and raising `lr` to 1e-3, changed together, so the fix isn't isolated to one lever. BatchNorm adds real per-epoch cost (25.3s vs 13.2s unnormalized), capping epochs at 100 to stay under a 45 minute budget; val accuracy plateaus by epoch 20-25 in a timing probe, so 100 leaves real margin.

Retrained result: macro F1 0.709 vs `baseline`'s 0.686 (delta +0.023). Looks like a gain, but every per-class delta (`car` +0.023, `large_vehicle` +0.068, `two_wheeler` +0.002, `pedestrian` +0.005, `pedestrian_group` +0.014) sits inside that class's own split-choice noise spread (section 5), the same standard that ruled out `range_sc` and the `hidden_dim` sweep. Not a real improvement, and `two_wheeler`/`pedestrian`, the classes actually driving the macro F1 ceiling, are essentially untouched.

**Decision:** retain the 2-hidden-layer baseline. Depth doesn't help once it's actually trainable, consistent with section 6: the bottleneck is the histogram feature representation, not model capacity in either width or depth.

**Side finding, a real caching bug:** `evaluate_val_metrics` cached its numeric metrics keyed only by file existence, not by whether the underlying model had changed. Retraining `deep10` in place with the fixed config left a stale `mlp_val_metrics.json` from the collapsed first attempt, which got silently reused, so the first "fixed" result reported was actually the failed run's numbers. Fixed by invalidating the cache whenever it's older than `mlp_model.pt`.

**Open, not acted on:** the 64 histogram-bin features are all bounded [0,1], but `doppler_spread` is unnormalized (train range 0-59.2, mean 0.233), a scale mismatch at the input layer. Adam's per-parameter step size, and for `deep10`, BatchNorm sitting right after the first layer, both blunt this, which is likely why it hasn't visibly broken anything, but it hasn't been tested directly (e.g. z-score standardizing `doppler_spread` on train stats).

## 8. Feature design: point count (n_points scalar vs raw-count histogram)

Histogram features are fraction-of-points-per-bin, which discards instance point count entirely (a 1-point and an N-point instance landing in the same bin are indistinguishable). Two ways of putting it back: `n_points` (baseline plus one extra scalar, raw instance point count appended unbinned) and `raw_counts` (baseline with `normalize=false`, unnormalized per-bin counts, so point count is recoverable by summing a feature's bins). `scripts/mlp_variants.py`'s `n_points`/`raw_counts`, results at `results/mlp/n_points/`, `results/mlp/raw_counts/`.

Macro F1: `baseline` 0.686, `n_points` 0.692, `raw_counts` 0.690. Both deltas, and every per-class delta, sit inside the split-choice noise floor (section 5). Both nudge the `pedestrian`/`pedestrian_group` precision/recall balance in the theoretically expected direction (`pedestrian` recall down from 0.944, `pedestrian_group` recall up from 0.544), but neither shift clears noise either.

**Finding:** not a real improvement, by either route. Two independent mechanisms for handing the network point count converge on the same null result, stronger evidence than either alone that point count isn't the missing piece, or that the true effect is smaller than this dataset's split noise can resolve.

## 9. Feature design: orientation and angular features (spatial_extent, azimuth_extent)

`x_rel`/`y_rel` are per-point and orientation-dependent (a car seen broadside vs head-on spreads very differently across the two axes for the same physical object). Two tests: `spatial_extent` drops them in favor of one orientation-robust scalar per instance (bounding-box diagonal, `sqrt(x_extent^2+y_extent^2)`), appended unbinned; `azimuth_extent` adds a new scalar on top of `baseline` instead (angular spread as seen by the sensor, max-min of `azimuth_sc`, a point-level column nothing had used until now). `scripts/mlp_variants.py`'s `spatial_extent`/`azimuth_extent`, results at `results/mlp/spatial_extent/`, `results/mlp/azimuth_extent/`.

Macro F1: `baseline` 0.686, `spatial_extent` 0.690, `azimuth_extent` 0.696. Every per-class delta sits inside the split-choice noise floor (section 5); `azimuth_extent`'s `two_wheeler` delta (+0.023) is the largest of any feature-design test so far, still a fraction of that class's 0.386 noise spread.

Also checked whether `spatial_extent`'s own split-choice noise floor is narrower than baseline's, since it was specifically built to remove the orientation-sensitivity mechanism thought to drive part of that noise (see Key takeaways). It isn't, meaningfully: macro F1 across the same 6 splits ranged 0.655-0.745 (std 0.030) vs baseline's 0.651-0.734 (std 0.027), essentially unchanged. Per-class, `large_vehicle` and `two_wheeler`, the classes an orientation fix would most plausibly help, shrank a little (0.197->0.167, 0.386->0.356) but not enough to matter; `car`/`pedestrian`/`pedestrian_group` were flat. Script: `scripts/split_sensitivity.py`'s `features`/`extra_features`/`output_subdir` params (generalized for this check), results at `results/mlp/split_search_spatial_extent/fold_<n>/`.

**Finding:** sixth independent axis (capacity, depth, `n_points`, `raw_counts`, `spatial_extent`, `azimuth_extent`) landing inside the noise floor, both on the fixed baseline split and, for `spatial_extent`, across the noise floor itself. Reinforces that the ceiling isn't from any one specific feature-engineering choice, it's more fundamental to what this per-instance summary vector can express at ~2.9 points/instance average.

## Key takeaways

**Never compare metrics across two different splits, even both stratified by class proportion.** Section 5's `range_sc` cross check is the concrete example: comparing this project's fixed split against an independently built one swung `large_vehicle`'s recall by 20+ points and its dominant confusion (`large_vehicle -> car`) by 3x, with nothing about the model or features changing. A number sourced from outside this project's fixed split (`sequence_split.py`) isn't a caveat away from being comparable, it's a different measurement.

**Why proportional counts don't guarantee representative content.** `StratifiedGroupKFold` only targets matching instance-count ratios per split, and it can't split a sequence, so hitting that ratio means moving whole sequences around. If a class's supply is concentrated in a couple of sequences, that becomes a coarse bin-packing problem, not a representative sample: whichever few sequences happen to land in val define that split's entire picture of the class, regardless of how correct the overall ratio comes out.

This matters because a class's radar signature isn't one fixed thing, it varies by aspect angle and motion state. A bicycle stopped at a red light, facing the sensor, reads zero Doppler (stationary) and a narrow RCS cross-section (front-on). A bicycle crossing broadside in front of the sensor also reads near-zero Doppler, since Doppler only sees the radial velocity component and almost all of that motion is perpendicular to the sensor, but its RCS is much larger (broadside cross-section). Same label, different feature distribution. If the sequences carrying the "stopped, front-on" flavor land in train and the "crossing, broadside" ones land in val, the model is trained on one physical presentation of `bicycle` and tested on another, not because bicycles are inherently hard, but because train and val are sampling different physical presentations of the same label. This is a real part of `two_wheeler`'s instability across runs here (top1 sequence share 18.8%, top3 33.8%, see the check below).

**Fold count doesn't fix this, it only reveals it.** Every fold, at any fold count, is still built from whole sequences, so no individual fold's val set gets less lumpy just because there are more of them. What more folds buys you is watching the metric swing across several different lumpy allocations, that variance is diagnostic, it tells you the class is unstable, it doesn't correct any single fold. The only real fix is more independent sequences carrying the class.

**Split selection test, inconclusive.** Tried picking a "better" val set by generating the 6 distinct valid splits at the fixed 70/15/15 proportions (6 isn't a chosen number, it's just how many equal-sized chunks a k-fold split needs so each one lands close to val's 15% share) and scoring each by how closely train and val's per-class feature distributions matched (two-sample Kolmogorov-Smirnov statistic). Trained `baseline` on all 6: macro F1 ranged 0.651 to 0.734 purely from split choice, and the best-matching split scored near the worst while the worst-matching split scored best, so distribution match didn't predict model performance. Inconclusive, no candidate is a demonstrated improvement. **Decision:** keep the standing split as is, not worth the full retrain of every variant for an unproven gain. Side effect: this range (0.083) also confirms the `range_sc` vs `baseline` gap (0.018) from section 5 is noise, not signal, closing that open question. Script: `scripts/split_sensitivity.py`, results at `results/mlp/split_search/fold_<n>/`.

**Root cause is data collection, not splitting.** This is the long tail problem: naturalistic driving passively records whatever crosses the sensor's path, so rare classes end up with thin, incidental coverage of their possible presentations, while common classes (`car`) get broad coverage for free just by being everywhere. RadarScenes is fixed and already collected, so the practical response isn't a better split, it's knowing which classes have this kind of thin coverage before trusting a single number for them. Check per class: instance count, number of distinct sequences containing it, and the share coming from its single busiest sequence (`taxonomy_separability.plot_taxonomy_class_balance`), a groupby, not a training run.
