## 1. Class grouping: merge train/truck/bus/large_vehicle, two_wheeler, drop animal/other_dynamic

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
bus                -> large_vehicle
train              -> large_vehicle
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
large_vehicle / truck: Strong case for merging. Their pairwise AUC is the weakest of the three (0.632, only slightly above chance), and the RF confusion matrix shows large_vehicle being classified as truck 74% of the time. Both metrics point to the same conclusion, providing consistent evidence rather than noise.
bus: The evidence is mixed. bus is well separated from large_vehicle (AUC 0.888), but poorly separated from truck (AUC 0.657, with 34% RF confusion). Therefore, the data does not support either treating bus as clearly distinct from both classes or merging it with both. Whether to merge bus is ultimately a design judgment based on the acceptable level of bus/truck ambiguity, or requires additional evidence such as more sequences or richer features.