## 1. Class grouping: merge train/truck/large_vehicle, keep bus separate, merge two_wheeler, drop animal/other_dynamic

RadarScenes has 12 raw classes. Trained directly on those, several
are too rare to learn or evaluate reliably. From a 5-sequence, single-sensor sample:


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


`train` in particular is bad enough because all train instances are contained within a single sequence, meaning they cannot be split across training and validation datasets without introducing data leakage.

**Decision:** reduce to 6 classes, matching the official `radar_scenes` package's own
`ClassificationLabel` grouping (`radar_scenes/labels.py`), this isn't just a
convenience merge, it's the dataset authors' own recommended scheme for ML tasks:

```
car               -> car
large_vehicle      -> large_vehicle
truck              -> large_vehicle
train              -> large_vehicle
bus                -> bus
bicycle            -> two_wheeler
motorized_two_wheeler -> two_wheeler
pedestrian         -> pedestrian
pedestrian_group   -> pedestrian_group
static             -> (dropped, not trained on)
animal             -> (dropped, not trained on)
other_dynamic       -> (dropped, not trained on)
```

`animal` and `other_dynamic` are dropped rather than merged because they're low-signal catch-all classes (officially mapped to None) rather than coherent categories for a classifier.

This assumes train/truck/bus share a similar-enough radar
signature to large_vehicle that merging doesn't destroy separability. Planned
EDA (RCS/point-count distributions per class) should confirm with data rather than just
trusting the assumption. Revisit if train looks radically different. However, the train label count is too low and is essentially noise.

Merge truck, bus, and large_vehicle into a single large_vehicle class if the logistic-regression probe, trained on the full feature vector (RCS, compensated Doppler, x_rel, y_rel, doppler_spread), shows poor held-out separability between them. Support the decision with pairwise Jensen–Shannon divergence and a shuffled-label baseline, rather than relying on 1D histogram overlap.

### After running scripts/taxonomy_separability.py
PROBE_FEATURES in scripts/taxonomy_separability.py:

rcs — per-instance median RCS (median of all that instance's points)
vr_compensated — per-instance median compensated radial velocity
x_extent — max(x_rel) − min(x_rel), where x_rel is each point's x_cc minus its instance's mean x_cc
y_extent — same, for y_cc
doppler_spread — per-instance median absolute deviation of vr_compensated (median-based spread, per your earlier spec)

large_vehicle / truck: Strong case for merging. Their pairwise AUC is the weakest of the three (0.632, only slightly above chance), and the RF confusion matrix shows large_vehicle being classified as truck 74% of the time. Both metrics point to the same conclusion, providing consistent evidence rather than noise.
bus: The evidence is mixed. bus is well separated from large_vehicle (AUC 0.888), but poorly separated from truck (AUC 0.657, with 34% RF confusion). Therefore, the data does not support either treating bus as clearly distinct from both classes or merging it with both. Whether to merge bus is ultimately a design judgment based on the acceptable level of bus/truck ambiguity, or requires additional evidence such as more sequences or richer features.

### Confirmed with 5-fold CV on the fixed train+val split (decision 5)

The numbers above came from a single held-out fold with no separate test set. Re-running on decision 5's fixed split (test excluded entirely) with `separability_probe.run_probe_cv`'s proper 5-fold averaging adds precision/recall/f1, which weren't available before - computed on 68 of the 130 train+val sequences, since the other 62 contain none of these 3 classes at all (only 23 sequences ever contain a large_vehicle instance, 27 for bus, 59 for truck):

| class | LR AUC | LR F1 | RF AUC | RF F1 |
| --- | --- | --- | --- | --- |
| large_vehicle | 0.779 ± 0.041 | 0.233 ± 0.052 | 0.791 ± 0.084 | 0.244 ± 0.069 |
| truck | 0.677 ± 0.065 | 0.544 ± 0.055 | 0.775 ± 0.053 | 0.784 ± 0.070 |
| bus | 0.756 ± 0.070 | 0.513 ± 0.063 | 0.818 ± 0.068 | 0.554 ± 0.105 |

This strengthens the large_vehicle/truck merge case rather than changing it: AUC alone looks passable (0.78-0.79), but precision/recall expose `large_vehicle` as barely usable as its own class (RF precision 0.239, recall 0.287) - most of what a model calls "large_vehicle" is wrong, and most true instances are missed, sharper evidence than but consistent with the original 74%-confused-as-truck confusion matrix. The large std on `large_vehicle` (recall ± 0.142, nearly 50% relative) reflects that only 23 sequences ever contain one - a sequence-coverage scarcity on top of the instance-count imbalance already noted above.

**Resolution:** merge large_vehicle into truck (target class name kept as `large_vehicle`, per the official scheme). Keep `bus` as its own class

bus is kept separate despite the mixed pairwise, `run_separability_probe` trains on 5 hand-aggregated scalars (median RCS, median Doppler, two extents, one spread value) rather than full per-instance distributions. Actual classifier will use paper-faithful per-instance histograms. Revisit this later: if bus is still heavily confused with large_vehicle/truck in the real classifier's confusion matrix once real features are used, merge it in then, don't decide it now with a crude probe.

## 2. Histogram bin range: percentile clip, not mean ± kσ
Two approaches can be used to define a feature’s histogram range while limiting the influence of outliers: [p1, p99], as implemented in feature_distributions.py and histogram_separability.py, or the common alternative of mean ± kσ.

We chose the percentile-based approach because mean and standard deviation are themselves sensitive to outliers. This is particularly problematic for skewed or heavy-tailed features such as doppler_spread. For cars, for example, p1=0.0, p50≈0.01, and p99≈14.77, indicating a large concentration of near-zero values combined with a long positive tail rather than a Gaussian distribution. In this case, the tail can inflate σ—the same tail that mean ± kσ is intended to exclude—causing the resulting effective range to remain unnecessarily wide.

Percentiles are based on order statistics and therefore do not make assumptions about the distribution’s shape: [p1, p99] directly defines the range containing the central 98% of observations.

## 3. Histogram bin count: 16

16 bins were selected from the random-forest separability probe, not from eyeballing the histograms: macro AUC improves substantially from 8→16 bins and plateaus thereafter. That plateau only clearly holds for AUC, though - per-class F1 (RF) splits by class rather than plateauing cleanly. `large_vehicle` peaks exactly at 16 (0.531) and drops at 32 (0.516), which directly supports the choice. `bus`, the smallest and most data-starved class, peaks at 4 bins (0.585) and never recovers (0.508 at 16, 0.511 at 32) - more resolution just means sparser, noisier features for it. `two_wheeler` keeps climbing through 32 (0.533→0.581), and is what pulls the macro-F1 average past 16 on its own. So there's no bin count that's best for every class: 16 is a reasonable compromise, avoids 32's added sparsity and dimensionality, and happens to be specifically optimal for `large_vehicle` - the class at the center of this whole investigation - rather than a universal optimum.

**Revisit at Day 6/7:** this was picked by comparing 4 candidates on one held-out sequence-grouped split, with no separate validation/test set - proportionate for choosing an encoding default, not a validated final answer. Once the real baseline classifier has a proper train/val/test split, re-check whether 16 still holds up on genuinely untouched data, the same way `bus`'s merge decision (decision 1) is flagged for revisiting once the real classifier's confusion matrix exists.

### Confirmed with 5-fold CV on the fixed train+val split (decision 5), now that decision 6 makes it load-bearing

Re-run on decision 5's fixed split (test excluded) with `separability_probe.run_probe_cv`'s 5-fold averaging, since decision 6 committed to histogram encoding for the real classifier:

| n_bins | model | macro AUC | macro F1 |
| --- | --- | --- | --- |
| 4 | RF | 0.874 | 0.577 |
| 8 | RF | 0.911 | 0.618 |
| 16 | RF | 0.925 | 0.647 |
| 32 | RF | 0.928 | 0.650 |

This strengthens the choice rather than changing it, and more cleanly than the original single-fold estimate: RF macro F1 16→32 is now +0.003 (essentially flat), versus +0.010 before. `large_vehicle` F1 plateaus exactly at 16 (0.611) and stays flat through 32, instead of dropping like the noisier single-fold run showed. `bus` still doesn't benefit from more bins - F1 hovers ~0.49-0.51 through 16, then drops to 0.455 at 32. `two_wheeler` is still the one class that keeps climbing (0.472→0.534→0.583→0.616), but its pull on the macro average is smaller now than the earlier estimate suggested. 16 remains the pick.

## 4. Trusting RF/LR probe results despite using "simple" models

Random-Forest and Logistic Regression are simpler than a neural net, so can their separability results be trusted? Yes. A Random Forest with 300 trees is a flexible nonlinear ensemble and is well suited to tabular feature vectors such as the 16-bin histograms + doppler_spread. The main limitation is not model capacity, but the hand-crafted representation: a low AUC indicates that the histogram encoding provides limited separating signal for that class pair, not that a larger model would necessarily extract more from the same 65 features. A raw point-based architecture could still learn a richer representation directly from the point sets, which this probe does not assess. Agreement between RF and LR further suggests that the observed separability is a property of the features rather than a model-specific artifact.

## 5. Fixed train/val/test split, by sequence

Every probe so far (taxonomy merge, bin range, bin count) reused the same single held-out fold across multiple candidates to pick a winner - proportionate for picking encoding defaults (decision 4), but it means each "winning" number is mildly inflated by having been selected among candidates checked against that same fold, and there was no genuinely untouched data to report a final, honest number against.

**Decision:** split sequences (not instances, not scans) once into train/val/test (~70/15/15 - `scripts/sequence_split.py`), via two chained `StratifiedGroupKFold` calls (grouped by sequence_name, stratified by final class at the instance level), cached to `results/sequence_split.json` so it stays fixed rather than getting regenerated per run. Class balance held up well across all three splits despite grouping by sequence - even `bus`, the rarest class, stayed within 1.8-2.1% across train/val/test. `val` is for freely comparing candidates going forward (bin count revisit, model/architecture choices for the real classifier); `test` is set aside and checked exactly once, at the end, for the number that goes in the writeup.

## 6. Feature representation: histogram encoding, not raw point sets

Resolves the open question decision 4 flagged: whether a raw point-based architecture (consuming per-instance point sets directly, no hand-built histogram) could learn a richer representation than the histogram encoding RF/LR were probed on.