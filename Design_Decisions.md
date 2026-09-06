## 1. Class taxonomy: merge large_vehicle/truck/train/bus, merge bicycle/motorized_two_wheeler into two_wheeler, drop animal/other_dynamic

RadarScenes defines 12 raw classes. Several are too rare to train or evaluate reliably, per a 5-sequence, single-sensor sample:

| Class | Instances (158 seq) | Pct |
| --- | --- | --- |
| car | 214,395 | 42.6% |
| pedestrian_group | 126,517 | 25.1% |
| pedestrian | 79,372 | 15.8% |
| bicycle | 32,881 | 6.5% |
| truck | 31,395 | 6.2% |
| bus | 9,483 | 1.9% |
| other_dynamic | 4,558 | 0.9% |
| large_vehicle | 3,330 | 0.7% |
| motorized_two_wheeler | 1,617 | 0.3% |
| animal | 154 | 0.0% |
| train | 57 | 0.0% |

`train` is additionally confined to a single sequence, so no split can separate train from validation without data leakage.

**Decision:** reduce to the official `radar_scenes` package's `ClassificationLabel` grouping (`radar_scenes/labels.py`), the dataset authors' own recommended scheme for ML tasks:

```
car                    -> car
large_vehicle          -> large_vehicle
truck                  -> large_vehicle
train                  -> large_vehicle
bus                    -> bus
bicycle                -> two_wheeler
motorized_two_wheeler  -> two_wheeler
pedestrian             -> pedestrian
pedestrian_group       -> pedestrian_group
static                 -> (dropped)
animal                 -> (dropped)
other_dynamic          -> (dropped)
```

`animal` and `other_dynamic` are dropped rather than merged: both map to `None` in the official scheme, low-signal catch-all classes rather than coherent categories.

Evidence for merging large_vehicle/truck/train: a logistic regression and random forest probe (`scripts/taxonomy_separability.py`, features: median RCS, median compensated Doppler, x/y extent, Doppler spread) found large_vehicle/truck pairwise AUC of 0.632, near chance, and 74% RF confusion. Confirmed at 5-fold CV on the fixed split (decision 5): large_vehicle AUC reaches 0.78 to 0.79 but F1 only 0.23 to 0.24, unusable as a standalone class given only 23 of 130 train and val sequences contain an instance. train's 57 instances are too few to evaluate independently regardless.

`bus` was initially kept separate. Pairwise evidence was mixed: well separated from large_vehicle (AUC 0.888), poorly separated from truck (AUC 0.657), and the probe's 5 hand-aggregated scalars are not the production feature representation. Once the real MLP classifier existed, its confusion matrix showed heavy bus/large_vehicle confusion, and a direct ablation confirmed the merge improves rather than masks it: combined F1 0.756 versus 0.543 and 0.492 separately (`MLP_Report.md`).

**Final taxonomy:** 5 classes, car, large_vehicle, two_wheeler, pedestrian, pedestrian_group, the `baseline` variant in `scripts/mlp_variants.py`. The 6-class taxonomy with bus separate remains available as the `bus_separate` variant.

## 2. Histogram bin range: percentile clip, not mean plus/minus kσ

Per-feature histogram range is set to [p1, p99] (`feature_distributions.py`, `histogram_separability.py`) rather than mean ± kσ. Mean and standard deviation are themselves outlier-sensitive, particularly for skewed, heavy-tailed features such as doppler_spread: for car, p1=0.0, p50≈0.01, p99≈14.77, a near-zero mass with a long positive tail. The tail that mean ± kσ intends to exclude also inflates σ, widening the resulting range instead. Percentiles are order statistics and make no distributional assumption, directly bounding the central 98% of observations.

## 3. Histogram bin count: 16

Selected via the RF separability probe, not visual inspection. Macro AUC improves substantially from 8 to 16 bins and plateaus after. Per-class F1 does not plateau uniformly: large_vehicle peaks at 16 (0.531 single-fold, 0.611 at 5-fold CV) and drops at 32; bus peaks at 4 and degrades with more bins, added resolution increases sparsity rather than signal for the smallest class; two_wheeler keeps improving through 32 and is the only class pulling the macro average past 16. Confirmed at 5-fold CV on the fixed split (decision 5): RF macro F1 is 0.647 at 16 versus 0.650 at 32, effectively flat. 16 bins is the selected value: a reasonable compromise across classes, and specifically optimal for large_vehicle.

## 4. Trusting RF/LR probe results despite using simple models

RF and LR are simpler than a neural network, but their separability results remain valid. A 300-tree RF is a flexible nonlinear ensemble well suited to tabular histogram features, so a low AUC indicates limited signal in the encoding for that class pair, not necessarily insufficient model capacity. A raw point-based architecture could still extract more from the same points, which this probe does not assess. RF/LR agreement further indicates the observed separability is a property of the features, not a model-specific artifact.

The one case where RF and LR diverge is consistent with RF's own limitation: bootstrap resampling leaves a rare class thin or absent in many trees. On truck (~31k instances, 59 of 130 sequences), RF outperforms LR (F1 0.784 versus 0.544). On large_vehicle (~3.3k instances, 23 of 130 sequences), both converge to the same poor range (F1 0.244 versus 0.233): too little data for RF's flexibility to help.

## 5. Fixed train/val/test split, by sequence

Decisions 1 through 3 each selected among candidates using the same single held-out fold, adequate for choosing encoding defaults but leaving every selected value mildly inflated by having been chosen against that fold, with no untouched data for an unbiased final number.

**Decision:** split by sequence, not instance or scan, once, into train/val/test (approximately 70/15/15, `scripts/sequence_split.py`), via two chained `StratifiedGroupKFold` calls, grouped by sequence and stratified by instance-level class, taking the first fold of each, cached to `results/data/sequence_split.json`. Class balance held despite grouping by sequence: even bus, the rarest class, stayed within 1.8 to 2.1% across all three splits. val supports ongoing comparisons; test is checked once, at the end.

This is a fixed assignment, not a search. `sequence_split.py` also defines `select_best_split`, which searches candidate val carves by KS-statistic match to train's per-class feature distributions, but never writes to the cache and was never adopted: scoring 6 candidates this way found the best KS match performed near the worst on macro F1 (`MLP_Report.md`).

## 6. Feature representation: histogram encoding, not raw point sets

Open question from decision 4: whether a raw point-based architecture, consuming per-instance point sets directly with no hand-built histogram, could learn a richer representation than the histogram encoding RF/LR were probed on.
