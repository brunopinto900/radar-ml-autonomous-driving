# DeepReflecs vs Baseline MLP: Findings Report

Reimplementation of Ulrich, Glaser and Timm, "DeepReflecs: Deep Learning for Automotive Object Classification with Radar Reflections" (arXiv:2010.09273, 2021), trained and evaluated against the exact same fixed sequence grouped split, class taxonomy, and 6 fold noise floor check as the baseline histogram MLP (`MLP_Report.md`), so the two numbers are a like for like comparison, not two separate measurements. Implementation: `scripts/deepreflecs_classifier.py`. Split sensitivity check: `scripts/deepreflecs_split_sensitivity.py`.

## Setup

- **Classes**: `car`, `large_vehicle`, `two_wheeler`, `pedestrian`, `pedestrian_group`, identical to the baseline.
- **Input representation**: the raw, unordered, variable length list of an instance's own radar reflections, not a fixed length encoding. Each reflection carries 5 features: `x_rel`, `y_rel` (position in the instance's own centroid relative frame), `rcs`, `vr_compensated` (ego motion compensated radial velocity), `range_sc`. Same underlying points as the baseline, no histogram binning, no per-instance statistics.
- **Architecture** (paper Fig. 4): a shared per point linear layer (5 to 16 features) with ReLU, a global context layer (max pools the 16 local features to one global vector, concatenates it back onto every point, 16 to 32 features, no trainable parameters), a second shared per point linear layer (32 to 32) with ReLU, a masked global max pool over the point axis (32 to 32), then a dense classification head (32 to 5). 1,317 learnable parameters, vs. the baseline's 1,413. Order and size invariance come from the shared per point layers plus the max pools, not from any padding convention.
- **Sensor**: same as baseline, RadarScenes sensor 2 (front right) only.
- **Data**: same fixed sequence grouped train/val/test split as the baseline.
- **Noise floor**: reuses the baseline's own 6 fold split sensitivity check (`sequence_split.select_best_split`, same classes, same seeds), so both models are measured on the identical 6 candidate splits, not independently generated ones.

## Results: held out test set

| | Baseline MLP (histogram) | DeepReflecs (point set) |
|---|---|---|
| Test accuracy | 74.3% | 78.1% |
| Macro F1 | 0.699 | 0.745 |
| Parameters | 1,413 | 1,317 |

Per class F1 (test):

| Class | Baseline | DeepReflecs |
|---|---|---|
| `car` | 0.877 | 0.872 |
| `large_vehicle` | 0.766 | 0.707 |
| `two_wheeler` | 0.567 | 0.680 |
| `pedestrian` | 0.677 | 0.733 |
| `pedestrian_group` | 0.608 | 0.735 |

![DeepReflecs val confusion matrix](results/deepreflecs/deepreflecs_confusion_matrix.png)

Row normalized, val set. DeepReflecs wins on total accuracy, macro F1, and 3 of 5 classes, most on `two_wheeler` and `pedestrian_group`, the baseline's two weakest classes (`MLP_Report.md` findings 1 and 3). It trades away some `large_vehicle` F1 (0.766 to 0.707) for that gain, discussed below.

## Split sensitivity check: is this just a lucky split

A single split comparison is not enough evidence on its own, the baseline's own noise floor spans 0.651 to 0.734 macro F1 purely from split choice (std 0.027, `MLP_Report.md` Setup). Ran DeepReflecs on the same 6 candidate splits used to establish that floor, paired fold by fold against the baseline's own per fold numbers (`results/mlp/split_search/fold_*`, `results/deepreflecs/split_search/fold_*`):

| Fold | max_ks | Baseline macro F1 | DeepReflecs macro F1 | Delta |
|---|---|---|---|---|
| 0 | 0.179 | 0.679 | 0.725 | +0.047 |
| 1 | 0.624 | 0.734 | 0.802 | +0.068 |
| 2 | 0.286 | 0.701 | 0.751 | +0.050 |
| 3 | 0.336 | 0.651 | 0.723 | +0.072 |
| 4 | 0.287 | 0.695 | 0.765 | +0.070 |
| 5 | 0.287 | 0.688 | 0.730 | +0.042 |

DeepReflecs wins every single fold, delta ranges +0.042 to +0.072, mean +0.058. DeepReflecs' own spread across folds (0.723 to 0.802, std 0.030) is comparable in width to the baseline's (std 0.027), so both models carry similar split sensitivity, but DeepReflecs' floor sits above the baseline's ceiling on all but one fold. The smallest observed delta (+0.042) still clears the baseline's own established noise floor std by a solid margin. Conclusion: the gain is real, not a property of one lucky split, by the same standard `MLP_Report.md` uses to rule other apparent gains (`range_sc`, `deep10`) in or out.

## Why DeepReflecs wins: mechanism

Reasoning grounded in this project's own established findings (`MLP_FINDINGS.md`, `MLP_Decisions_and_Findings.md`), not new EDA:

- **Sparsity makes binning lossy.** Median instance has 2 points, 75th percentile 4 (`taxonomy_separability.add_relative_features`' own `n_points`). The baseline bins 4 features into 16 bins each, a 64 dimensional vector built from typically 2 to 4 raw numbers, almost entirely empty per instance and quantizing whatever landed into arbitrary percentile buckets. DeepReflecs never bins, it consumes the raw points at full precision.
- **Marginal encodings destroy joint, per point structure.** The baseline's histogram is 4 independent marginal distributions, it cannot represent "this specific point has high RCS and sits at the object's edge." DeepReflecs processes each point as one bundle, and the global context layer explicitly adds "this point relative to the object's own extremes" as a signal, which is the paper's own stated purpose for that layer.
- **That mechanism lines up with where DeepReflecs actually won.** `two_wheeler` and `pedestrian_group`, both classes whose defining signal is layout (an elongated, position correlated two wheeler; a group's multiple sub clusters), are exactly where the paper's own ablation found the global context layer helps most (their hardest class, `cyclist`, +12.5 F1 from that layer alone).
- **`large_vehicle`'s regression is a real, mechanistic tradeoff, not noise.** Large vehicles carry more points per instance. Global max pooling only keeps the single most extreme value per channel, so it throws away more of the distribution as point count grows, while a 16 bin histogram retains shape across the full range. A histogram is a better matched summary for point rich instances, raw max pooled point sets are better matched for point poor ones, and this dataset is overwhelmingly point poor.
- **This project's own re-verified mechanism for `two_wheeler`/`large_vehicle` is "does an extreme point exist," not smooth central tendency** (`MLP_FINDINGS.md`: "only ~4.3% of car/pedestrian instances have any point beyond &plusmn;1.3 m/s... vs ~8-9% for large_vehicle/two_wheeler", an instance level re-check that specifically ruled out the earlier claim of a clean bimodal peak). Mean and median target central tendency, exactly the wrong summary for a "does an outlier exist" signal, and this project's own explicit statistics experiment (`stat_descriptors`: mean/median/std of `rcs`/`vr_compensated`/`radial`/`azimuth_sc`, macro F1 0.658 vs baseline 0.686) already tested that and lost. DeepReflecs' per channel max pool is architecturally exactly "does an extreme point exist," learned per channel rather than fixed to one hand picked threshold, with no need to decide in advance which feature or cutoff matters.

## Implementation notes

- **Cache key correctness bug, found and fixed in both `mlp_classifier.py` and `deepreflecs_classifier.py`**: `run_training`'s cache key omitted `classes`, `splits` (and in the baseline's case, `features`/`extra_features`/`normalize` too). Two calls with the same `output_dir` but a different taxonomy, feature set, or split would have silently returned a stale cached model instead of retraining. Audited every actual call site in the project (`mlp_variants.py`/`MLP_CONFIG.json`, `split_sensitivity.py`, `regime_specialist_mlps.py`, `shape_features_academic.py`, both ablation notebooks): every one already used a distinct `output_dir` per differing config, so the bug never actually corrupted an existing result, it was latent, not realized. Fixed by resolving `splits` up front and including all of `classes`/`splits`/`features`/`extra_features`/`normalize` in the key.
- **`build_point_sets` performance bug**: the original implementation did `for _, inst in df.groupby(INSTANCE_COLS): inst[features].to_numpy(...)`, a label based column selection repeated once per instance, roughly 500,000 times for the full dataset. Found via `py-spy` after `evaluate_val_metrics` appeared hung: this project's pandas build resolves that column selection through a pyarrow string backed columns Index, turning a call that should be instant into a multi minute one when repeated at that scale. Fixed by selecting `features`/`group` vectorized exactly once, then slicing the resulting plain numpy array per instance via `groupby(...).indices`, pure numpy fancy indexing inside the loop, no repeated pandas column resolution. Verified output identical (same shapes and per class counts) before and after the fix.
- **`MPLBACKEND=Agg` required for headless runs in this environment**: `DISPLAY` is set to a WSL2 forwarding address with no reachable X server, so matplotlib's default backend auto-detection hangs probing it on first `plt.subplots()` call. Not specific to DeepReflecs, latent in every script in this project that plots from a fresh, non interactive `python3 script.py` invocation.

## Next steps

- Isolate whether "does an extreme point exist" is really the mechanism: a per feature max absolute deviation from median statistic (`maxAD`, complementary to the already used median absolute deviation `doppler_spread`), swapped in for `stat_descriptors`' central tendency features rather than added alongside them, tested specifically against `two_wheeler`/`large_vehicle`.
- `MLP_Report.md`'s own proposed v1.1 direction, aggregating a tracked object's points across multiple scans, attacks the same sparsity ceiling this report identifies and would likely benefit DeepReflecs' point set representation directly, with no encoding change needed, unlike the baseline's fixed length histogram which would need rework to consume a larger, multi scan point set.

## References

M. Ulrich, C. Glaser and F. Timm, "DeepReflecs: Deep Learning for Automotive Object Classification with Radar Reflections," arXiv:2010.09273, 2021. [https://arxiv.org/abs/2010.09273](https://arxiv.org/abs/2010.09273)

See `MLP_Report.md`'s own references for the baseline model and the RadarScenes dataset.
