# First Results (Day 6 model, 20-epoch checkpoint)

Findings from actually training and probing the Day 6 baseline (`scripts/train_mlp_full.py`,
`scripts/class_confusion_diagnostics.py`). Model/training design lives in `MLP_DESIGN.md`,
dataset/encoding design in `DESIGN_DECISIONS.md`, the epoch/gradient-norm investigation in
`MLP_FINDINGS.md`. This file is the deep-dive into the largest confusion specifically.

## Summary (one paragraph)

The Day 6 baseline (paper-faithful 3-layer histogram MLP, 20-epoch checkpoint) reaches 74.1%
raw / 74.6% balanced validation accuracy, but that average hides a real split: `car`,
`pedestrian`, and `two_wheeler` are classified well (83-91% recall), while `large_vehicle` and
`pedestrian_group` sit much lower (54-55%). The dominant error, `large_vehicle` misclassified
as `car` (42.5% of true `large_vehicle` instances), is now explained: it's specifically the
sparse, small-footprint `large_vehicle` instances (avg. 2.5 points, 2m bounding-box diagonal)
getting confused with `car`, not a range effect and not a general `large_vehicle`/`car`
inseparability - correctly-classified `large_vehicle` instances average 7.3 points and a 9.2m
diagonal, several times larger. The much smaller reverse confusion (`car` misclassified as
`large_vehicle`, 4.2%) is only partly explained by the same footprint-size effect and not by
point count, which behaves differently for `car` (low point count is normal for the class
generally, not specific to its errors). A secondary `large_vehicle`/`pedestrian_group`
confusion (93 instances, ~2% of true `large_vehicle`) remains unexplained by any tested
feature and was deprioritized given its small size relative to the dominant car confusion.

## Headline accuracy (validation, 20-epoch checkpoint)

- Raw accuracy: **74.1%**
- Balanced accuracy (macro-average recall): **74.6%**

Per-class recall:

| class | recall |
|---|---|
| two_wheeler | 91.0% |
| pedestrian | 89.4% |
| car | 82.9% |
| large_vehicle | 55.3% |
| pedestrian_group | 54.3% |

## Deep dive: the large_vehicle/car confusion

Confusion matrix asymmetry: `large_vehicle→car` = 42.5% (1825/4290), `car→large_vehicle` =
4.2% (797/18848) - a real, large asymmetry, not an artifact of how the percentages were framed.

**Tested and rejected:** raw range (`mean_range`) - no threshold or separation for
`large_vehicle`; correct and wrong predictions spread across the full 0-100m span equally.

**Tested and confirmed - `large_vehicle→car` direction:** point count, mean extent, and bbox
diagonal all show a real, substantial gap between correctly-classified and car-confused
`large_vehicle` instances:

| metric | misclassified as car | correct large_vehicle | ratio |
|---|---|---|---|
| point count (mean) | 2.50 | 7.29 | ~2.9x |
| mean extent_rel (mean) | 0.79m | 2.84m | ~3.6x |
| bbox diagonal (mean) | 2.00m | 9.23m | ~4.6x |

This was invisible in the initial overlaid-scatter version of these plots (overplotting hid
it) and only became visible once scatter was replaced with violin plots + a numeric summary
table (`scripts/class_confusion_diagnostics.py`) - the methodology lesson of this
investigation: a scatter with heavy overlap shows *where any point exists*, not *where most
points are*, and can hide a real difference in central tendency.

**Side thread, deprioritized:** `large_vehicle→pedestrian_group` (93 instances, ~2.1% of true
`large_vehicle`) - point count, mean extent, and bbox diagonal all came back null (no
separation). A drop in the ocean relative to the 41% car confusion; not pursued further.

**Reverse direction (`car→large_vehicle`, 797 instances) - mixed, genuinely different
pattern:**
- Point count does *not* cleanly separate correct from wrong here - correct-car median is
  already 2.0 points, and the wrong-prediction rows don't move consistently in one direction
  (`large_vehicle`-confused mean=3.96 is *higher* than correct car's 2.72, while
  `pedestrian`/`two_wheeler`-confused are *lower*). Car's low point count is a general property
  of the class (33.8% single-point instances, Day 4 EDA), not something distinguishing right
  from wrong.
- Extent-based metrics do show a real, smaller effect in the expected direction:
  `car→large_vehicle` mean extent 1.31m vs. correct car's 0.72m (~1.8x), bbox diagonal 3.48m
  vs. 1.84m (~1.9x). Weaker than the reverse direction's 3.6-4.6x, but present.

**Net picture:** `large_vehicle→car` is well-explained - it's specifically the sparse/small-
footprint `large_vehicle` instances that get mistaken for `car`. `car→large_vehicle` is
smaller in volume and only partially explained - footprint size contributes a bit, point count
doesn't discriminate the way it does for the reverse direction, and nothing tested fully
accounts for it yet.
