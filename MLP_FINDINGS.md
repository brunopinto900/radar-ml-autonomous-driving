# MLP Findings (Day 6)

Results from actually running the design in `MLP_DESIGN.md` on the full data
(`scripts/train_mlp_full.py`, `scripts/class_confusion_diagnostics.py`), not the smoke test
(`scripts/train_mlp.py`). Findings only - design rationale lives in `MLP_DESIGN.md`,
dataset/encoding rationale in `DESIGN_DECISIONS.md`.

Setup: paper-faithful 3-layer MLP (`80 -> 16 -> 16 -> 5`), `lr = 1e-5`, `batch_size = 256`,
full train (354,277 instances) / val (67,663 instances) splits, seed 42. **Canonical checkpoint
is 20 epochs / 16 hidden units** (`results/mlp_full_run/baseline_20epoch_h16/model.pt` - see section 1
for why 20 epochs, section 5 for why 16 units). All numbers below use this checkpoint unless
stated otherwise.

## 1. Headline accuracy and per-class recall

- Raw validation accuracy: **74.1%**
- Balanced validation accuracy (macro-average recall): **74.6%**

| class | recall |
|---|---|
| two_wheeler | 91.0% |
| pedestrian | 89.4% |
| car | 82.9% |
| large_vehicle | 55.3% |
| pedestrian_group | 54.3% |

Two specific, large, structural confusions - not noise (row-total percentages, i.e. of all
true instances of that class, regardless of what they got predicted as):

- **`pedestrian_group` -> `pedestrian`**: 40.5% of true `pedestrian_group` instances
  (8,726/21,562). Separated mainly by point count/density, and `pedestrian` is >54%
  single-point instances (`EDA.md`) - a sparse group can look identical to a single walker to a
  model with no explicit point-count feature.
- **`large_vehicle` -> `car`**: 41.0% of true `large_vehicle` instances (1,825/4,456).
  Deep-dive in section 4 - this one is now well-explained, not just observed.

(Section 4 also quotes a "42.5%" figure for this same confusion - that's a *different*,
narrower denominator restricted to instances predicted as either `car` or `large_vehicle`
only, excluding the ~4% predicted as one of the other 3 classes. Close to the 41.0% row-total
number here but not the same calculation - noted so the two don't look like a contradiction.)

Both confusions are consequences of the class-grouping and feature-set decisions (decision 1,
and the paper-faithful feature set not including anything point-count-aware), not evidence of
a training bug - confirmed directly in sections 2-3 below.

## 2. 1000 epochs vs. 20 epochs: the paper's epoch count is excessive here

First run: full 1000 epochs, ~23.2 min (`results/mlp_full_run/epoch_ablation_1000epoch/`).
Second run: same everything, cut to 20 epochs, ~0.4 min
(`results/mlp_full_run/baseline_20epoch_h16/`) - deterministic (same seed), so this is literally the
first 20 epochs of the same trajectory, not a separate run.

| | 20 epochs | 1000 epochs |
|---|---|---|
| train_acc | 73.7% | 77.2% |
| val_acc (raw) | 74.1% | 75.6% |
| val_acc (balanced, macro recall) | 74.6% | 75.1% |
| wall-clock | 0.4 min | 23.2 min |

Val accuracy/loss are essentially flat past ~epoch 20
(`results/mlp_full_run/epoch_ablation_1000epoch/{cost,accuracy}.png`) - the remaining 980 epochs
buy on the order of 1-1.5pp raw accuracy
for ~60x the training time. Not zero benefit (the 1000-epoch confusion matrix does show real,
if small, gains concentrated in `car` and `pedestrian_group` specifically), but nowhere close
to proportional to the extra epochs. **Decision: use the 20-epoch checkpoint going forward** -
cheap to retrain if a later change (features, architecture) shifts this.

**Not overfitting at either length.** Train/val accuracy track each other closely the whole way
(train leads val by ~1.6pp at epoch 1000, both still rising together, no divergence). If
overfitting were the concern, more epochs would be the wrong direction to worry about; it
isn't the concern here - the plateau is a ceiling, not a train/val split.

**Is the plateau a local minimum (i.e., would a different `lr` escape it)?** Zooming into the
first 20 epochs shows something worth noting: train/val *loss* decreases smoothly and
monotonically from epoch 1, but train/val *accuracy* actually **dips** between epochs 2 and 5
(val_acc 0.543 -> 0.504) before climbing back past its epoch-2 value around epoch 9 and
continuing up. This is not a sign of a local-minimum trap - loss (continuous, what the
optimizer actually descends) and accuracy (a non-differentiable argmax readout) are not
required to move together. A very early, lucky near-random-guess accuracy from initialization
can be higher than the accuracy of a model that has since started genuinely reshaping its
decision boundaries but hasn't finished - the loss curve, not the accuracy curve, is the honest
signal of optimizer progress here, and it never stalls or reverses at any point across all 1000
epochs. **Conclusion: don't decrease `lr`** - a smaller step size would only slow arrival at
the same plateau, not help escape it; nothing in the shape of either curve looks like classic
local-minimum stagnation (an extended stall followed by a jump). The plateau's likely cause is
capacity/feature-separability, not optimization - confirmed in sections 3 and 5.

## 3. Gradient norms: no vanishing or exploding gradients

Checked directly rather than inferred from curve shape - section 2's local-minimum discussion
argued from the loss curve's shape alone, so this closes the loop with a harder measurement.
Tracked the mean per-batch L2 gradient norm of each Linear layer's weight matrix
(`results/mlp_full_run/baseline_20epoch_h16/grad_norms.png`), input-side to output-side.

All three layers stay within the same order of magnitude throughout training (roughly
0.05-0.35). The input-side layer (closest to the 80-dim feature vector - where vanishing
gradients would bite first) is the **largest** by epoch 20, not the smallest - the opposite of
the vanishing-gradient signature, where gradients shrink by orders of magnitude the further
back they propagate. The middle layer dips slightly relative to the other two but never
collapses toward zero. Nothing exceeds ~0.35 at any point - no explosion either.

**Conclusion: not a vanishing/exploding gradient problem.** Makes sense in hindsight - this is
a 3-layer, ~1.6k-parameter ReLU network, well below the depth where this typically bites
(mostly a 10+ layer or RNN problem).

## 4. Deep dive: the large_vehicle/car confusion

Confusion matrix asymmetry: `large_vehicle->car` = 42.5% (1825/4290), `car->large_vehicle` =
4.2% (797/18848) - both restricted to instances predicted as either `car` or `large_vehicle`
(see the note in section 1 about this vs. the row-total framing) - a real, large asymmetry,
not an artifact of how the percentages were framed.

**Tested and rejected:** raw range (`mean_range`) - no threshold or separation for
`large_vehicle`; correct and wrong predictions spread across the full 0-100m span equally.

**Tested and confirmed - `large_vehicle->car` direction:** point count, mean extent, and bbox
diagonal all show a real, substantial gap between correctly-classified and car-confused
`large_vehicle` instances:

| metric | misclassified as car | correct large_vehicle | ratio |
|---|---|---|---|
| point count (mean) | 2.50 | 7.29 | ~2.9x |
| mean extent_rel (mean) | 0.79m | 2.84m | ~3.6x |
| bbox diagonal (mean) | 2.00m | 9.23m | ~4.6x |

This was invisible in the initial overlaid-scatter version of these plots (overplotting hid
it) and only became visible once scatter was replaced with violin plots + a numeric summary
table (`scripts/class_confusion_diagnostics.py`) - a scatter with heavy overlap shows *where
any point exists*, not *where most points are*, and can hide a real difference in central
tendency.

**Side thread, deprioritized:** `large_vehicle->pedestrian_group` (93 instances, ~2.1% of true
`large_vehicle`) - point count, mean extent, and bbox diagonal all came back null (no
separation). A drop in the ocean relative to the 41% car confusion; not pursued further.

**Reverse direction (`car->large_vehicle`, 797 instances) - mixed, genuinely different
pattern:**
- Point count does *not* cleanly separate correct from wrong here - correct-car median is
  already 2.0 points, and the wrong-prediction rows don't move consistently in one direction
  (`large_vehicle`-confused mean=3.96 is *higher* than correct car's 2.72, while
  `pedestrian`/`two_wheeler`-confused are *lower*). Car's low point count is a general property
  of the class (33.8% single-point instances, Day 4 EDA), not something distinguishing right
  from wrong.
- Extent-based metrics do show a real, smaller effect in the expected direction:
  `car->large_vehicle` mean extent 1.31m vs. correct car's 0.72m (~1.8x), bbox diagonal 3.48m
  vs. 1.84m (~1.9x). Weaker than the reverse direction's 3.6-4.6x, but present.

**Net picture:** `large_vehicle->car` is well-explained - it's specifically the sparse/small-
footprint `large_vehicle` instances that get mistaken for `car`. `car->large_vehicle` is
smaller in volume and only partially explained.

## 5. Capacity ablation: a bigger model doesn't fix it

Tests whether the baseline's capacity (~1.6k params) is the bottleneck, or whether it's a
features/information-content ceiling instead. Same everything else - only `HIDDEN_DIM` changed:
`16 -> 64` (`results/mlp_full_run/capacity_ablation_h64/`, separate from the canonical
`baseline_20epoch_h16/` checkpoint).

| | 16 hidden units (canonical) | 64 hidden units |
|---|---|---|
| train accuracy | 73.7% | 75.5% |
| val accuracy (raw) | 74.1% | 73.9% |
| train - val gap | -0.4pp | +1.6pp |

`large_vehicle` true-class confusion (val):

| | 16 hidden units | 64 hidden units |
|---|---|---|
| large_vehicle -> car (wrong) | 1825 | 2152 |
| large_vehicle -> large_vehicle (correct) | 2465 | 2186 |

**Conclusion: making the model bigger is not the right fix.** Validation accuracy did not
improve with 4x the hidden units - flat, if anything slightly down - while training accuracy
climbed, opening a train-val gap where none existed before (the standard overfitting
signature). The specific confusion this was meant to fix got *worse*. Mechanism: a 1-2 point
instance's 80-dim input vector is nearly all zeros, carrying very little real information
regardless of model size - extra capacity doesn't give the model a richer input to work with,
it just gives it more room to fit idiosyncrasies of the training set in that same
low-information regime. Overfitting, but for the wrong reason: not "the signal is too complex
for this model," but "there's too little signal here for any model to exploit, and a bigger
one just memorizes noise instead."

**Caveat:** one run per size, no repeated seeds, and the train-val gap is small (1.6pp) - real
and in the right direction, but a mild/early signal, not a dramatic one.

## 6. Dropping sparse training instances: tested, makes things worse (closed)

Hypothesis: sparse (≤3-point) training instances are noise, not weak-but-real signal, so
dropping them from training should help. Tested by filtering the **training set only** to
≥4-point instances (`MIN_TRAIN_POINTS = 4`, val stays full/unfiltered) - aggressive: 354,277 ->
91,269 train instances (-74.2%, most classes are majority-sparse to begin with, `EDA.md`).
**Rejected, sharply, in every variant tested:**

| | canonical (full train) | min4pts (train filtered) |
|---|---|---|
| val accuracy, full val set | 74.1% | 48.6% |
| val accuracy, identical dense (≥4pt) val subset only - the fair, apples-to-apples test | 83.2% | 52.7% |

The second row (`results/mlp_full_run/dense_subset_diagnostics/`) rules out "it's just being
scored on cases it never learned" - same 14,322 test instances both times, only the training
data differs, and min4pts still loses by 30.5pp. `car` recall specifically: 81.5% -> 3.3%.
`large_vehicle` (81.4% -> 96.5%) and `pedestrian_group` (84.3% -> 86.9%) improve, but the losses
in `car`, `pedestrian` (64.6% -> 45.5%), and `two_wheeler` (96.7% -> 59.3%) dominate.

On the sparse (≤3pt) val subset (`results/mlp_full_run/sparse_subset_diagnostics/`), the
collapse isn't a uniform "everything becomes `large_vehicle`" - `pedestrian_group` absorbs the
most mispredictions overall (40.4%), `car` second (29.4%), `large_vehicle` third (17.6%). It's
concentrated on `car` specifically: 50.4% recall with 42.5% going to `large_vehicle`, ~10x the
canonical model's 4.2% (pair-restricted, section 4).

**Why `car` specifically - three explanations tested, two wrong:**

1. *Data starvation?* No - `car` still has the most raw training examples of any class after
   filtering (35,659, more than `large_vehicle`'s 21,660; full counts and per-class recall over
   training in `results/mlp_full_run/*/class_imbalance.png`).
2. *Class-weight reweighting favoring `large_vehicle`?* No - `car`'s loss weight barely moves
   (0.44 -> 0.51); `large_vehicle`'s actually drops (2.17 -> 0.84) once it's no longer rare in
   the filtered set.
3. *Structural feature overlap - the filter keeps only `car`'s atypical, `large_vehicle`-sized
   tail (only 22.2% of `car` has ≥4 points vs. `large_vehicle`'s 66.4% self-retention)?*
   Plausible from the point-count/extent proxies, but checked directly on the raw 80-dim feature
   vectors (`scripts/car_large_vehicle_feature_overlap.py`) and it **doesn't hold**: a logistic
   regression still separates the dense-`car` tail from `large_vehicle` at 98.5% held-out AUC,
   barely down from full-`car`'s 99.2%. The signal to tell them apart is still there.

**The actual driver: a hidden ~4x fewer-gradient-steps confound.** `EPOCHS=20` was held
constant across every ablation, but steps/epoch = dataset_size / batch_size, so the same "20
epochs" meant ~27,700 total gradient steps for the full 354,277-instance set and only ~7,100 for
the 91,269-instance filtered set. Per-class accuracy over those 20 epochs
(`scripts/min4pts_per_class_accuracy.py`) shows `car` peaking at epoch 6 (60.8%) then collapsing
continuously to 3.2% by epoch 20 - the mirror image of `large_vehicle`'s smooth monotonic climb
(2.0% -> 95.2%, never plateauing). Not a low-representation-class effect (`car` is the *largest*
class in the filtered set, 39.1%) - every extra gradient step keeps buying `large_vehicle`
confidence at `car`'s specific expense, because the loss saved by confidently getting
`large_vehicle` right outweighs the loss cost of flipping already-marginal `car` instances.

Retraining step-matched (80 epochs, ~28,600 steps, `results/mlp_full_run/min_train_points_epoch_matched/`)
confirms it: `car` recall recovers to 75.7% (vs. 82.9% baseline) and overall val accuracy to
67.9% (vs. 74.1% baseline) - most of the gap closes, but not all of it, and `large_vehicle`
recall drops to 48.5% as a direct trade-off. A new train/val gap also appears (84.9%/67.9%, the
standard overfitting signature) that isn't present anywhere else in this document. Adding
regularization (dropout/weight-decay) to train longer without that overfitting was considered
but not run - even a well-tuned version of this approach is trained on 4x less data than the
baseline and was never going to beat it outright, so it wasn't worth pursuing further.

**Conclusion: closed, in every variant tried** (train-only filter, train+val filter, dense-only
fair comparison, step-matched retrain). Sparse instances carry weak-but-real signal, not noise -
removing them costs both data volume and, less obviously, a large chunk of effective training
(fewer gradient steps at the same nominal epoch count) at the same time. Keeping them
outperforms every tested alternative to removing them.

Files: `results/mlp_full_run/{min_train_points_ablation,min_train_val_points_ablation,
min_train_points_epoch_matched}/`, `results/mlp_full_run/{sparse,dense}_subset_diagnostics/`,
`results/mlp_full_run/large_vehicle_car_confusion/car_large_vehicle_pca.png`,
`results/mlp_full_run/*/class_imbalance.png`.

## Next steps (not yet decided, not ranked)

- **Explicit point-count feature.** Point count is technically already recoverable from the
  raw-count histograms (sum across bins), but the model has to learn that summation itself;
  making it an explicit input feature removes that burden. Directly targeted at the
  `large_vehicle/car` confusion (section 4) and the `pedestrian`/`pedestrian_group` one
  (section 1).
- **Physics-only vs. full-feature-set ablation** (`TODO.md` Day 7, part of Day 6's own original
  scope, not yet run): `rcs`+`vr_compensated` alone vs. the full 5-feature set. Tests whether
  `range_sc`/`x_rel`/`y_rel` carry real shape signal or are a scene-layout shortcut - a
  different question from anything above.
- **Revisit the `large_vehicle` merge** (`DESIGN_DECISIONS.md` decision 1) - now has a real
  number attached (55% recall) instead of just a "not yet checked" caveat, but the most
  disruptive option (redo the split, retrain from scratch).
- **Sensor fusion** (`TODO.md` v1.2, "all sensors, not just sensor #2") - now more directly
  motivated: sparsity (not capacity, section 5) is the confirmed bottleneck, and the current
  pipeline only uses 1 of 4 available sensors. Not a one-line fix, though - RadarScenes scans
  each sensor independently/asynchronously (`dataloader.py`: "a scene is always one sensor's
  measurement"), so genuinely denser per-instance point clouds would need real spatial/temporal
  fusion across sensors, not just dropping the `sensor_id == 2` filter (which would mostly add
  more separate, still-sparse instances rather than enriching existing ones).
- **Capacity is closed** (section 5) - not a candidate next step, ruled out.
- **Dropping sparse training instances is closed** (section 6) - not a candidate next step,
  ruled out; made accuracy sharply worse and inflated the `car`/`large_vehicle` confusion.
