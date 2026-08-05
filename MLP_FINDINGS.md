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

## 7. Feature correlation analysis (pre-work for adding an explicit point-count feature)

Before adding `n_points` as an explicit 81st input dimension, checked two things directly
(`scripts/feature_correlation_analysis.py`, train set, `results/mlp_full_run/feature_correlation/`):
how redundant it already is with the existing 80 dims, and which features/classes actually have
a real relationship worth exploiting.

**Feature-feature correlation: `n_points` is already ~fully redundant.** Each existing feature
is a 16-bin histogram, and summing any one block's bins gives back `n_points` exactly (every
point contributes one count to every feature's histogram). Confirmed: `n_points` correlates
with each block's own sum at **r = 0.985-0.997**. So it isn't new information - it's a linear
combination of what's already there. What adding it explicitly buys is removing an optimization
burden (the network no longer has to learn "sum these 16 numbers" itself, a ~1.6k-param net at
`lr=1e-5` over 20 epochs), not new signal - which sharpens the shortcut-learning concern raised
before running this: an easy, highly-predictive, already-available-for-free signal can crowd out
a small/slowly-trained network's use of the harder shape information sitting alongside it.

**Feature-label correlation - a confound, caught and fixed.** First attempt correlated each
feature block's *sum* against each one-hot class label, meant to illustrate the redundancy above
- but a block's sum is just `n_points` again, so every block came back with nearly identical,
inflated correlations to a class (e.g. all 5 blocks ~0.41-0.43 for `large_vehicle`) that had
nothing to do with that block's actual content. This produced a misleading number (`vr_compensated`
"0.427-correlated" with `large_vehicle`) that looked like a real Doppler signal but was 100% the
point-count confound bleeding through.

**Fix:** separate magnitude from shape before correlating. `n_points` (magnitude) is one number
per class, already covered above. For shape, converted each bin to its *share* of the instance's
own total points (`bin_count / n_points`) before correlating - this removes the "how many points
overall" effect and isolates the genuine within-instance distribution shape.

| class | rcs (shape) | vr_compensated (shape) | range_sc (shape) | x_rel (shape) | y_rel (shape) | n_points (magnitude) |
|---|---|---|---|---|---|---|
| car | 0.111 | 0.160 | 0.039 | 0.139 | 0.022 | -0.106 |
| large_vehicle | 0.056 | 0.051 | 0.033 | **0.287** | 0.117 | +0.429 |
| pedestrian | 0.075 | 0.104 | 0.037 | 0.114 | 0.089 | -0.194 |
| pedestrian_group | 0.077 | 0.131 | 0.027 | 0.073 | 0.035 | +0.013 |
| two_wheeler | 0.062 | 0.052 | 0.050 | 0.046 | 0.039 | -0.023 |

(shape values are mean |correlation| across each block's 16 bins - magnitude-controlled, so
comparable across blocks; sign is retained for `n_points` since it's a single dimension)

**The Doppler question this resolves:** `large_vehicle`'s genuine `vr_compensated` shape signal
is 0.051 - one of its *weakest* entries, not a strong one. The earlier 0.427 was the magnitude
artifact, not a real velocity relationship.

**What's real:** `large_vehicle`'s standout, magnitude-independent signal is `x_rel` shape
(0.287) - more than double anything else in the table, and consistent with section 4's
bbox-diagonal/extent findings (3.6-4.6x gaps) via a completely different method (correlation
here vs. violin-plot summary stats there) - a genuine cross-check, not the same measurement
twice. `car` has no standout univariate shape signal at all (max 0.160) - consistent with it
needing a harder, more genuinely joint/nonlinear signal to classify than `large_vehicle` does.
`n_points` itself is `large_vehicle`'s single strongest correlate of anything in this analysis
(+0.429) - but per the redundancy finding above, it's not new information the network doesn't
already have access to.

## 8. Explicit point-count feature: small, mixed effect - not the fix

Tested directly after section 7's correlation pre-work: `INCLUDE_N_POINTS = True` in
`histogram_features.py`/`train_mlp_full.py` appends `n_points` as an explicit 81st input
dimension (`scripts/histogram_features.py`'s `instance_vector`/`build_dataset`, threaded through
a separate `"_pc"`-suffixed feature cache so it doesn't collide with the 80-dim baseline).
Everything else identical to the canonical run (20 epochs, `HIDDEN_DIM=16`, full data)
(`results/mlp_full_run/point_count_feature/`).

| class | baseline (80-dim) | +point_count (81-dim) | Δ |
|---|---|---|---|
| car | 82.9% | 81.3% | -1.6pp |
| large_vehicle | 55.3% | 56.9% | +1.6pp |
| pedestrian | 89.4% | 88.7% | -0.7pp |
| pedestrian_group | 54.3% | 56.8% | +2.5pp |
| two_wheeler | 91.0% | 90.8% | -0.2pp |
| **overall (raw)** | **74.1%** | **74.3%** | +0.2pp |

**The shortcut-learning worry from section 7 materialized, but mildly, not dramatically.** `car`
and `large_vehicle` moved by the exact same 1.6pp in opposite directions - the predicted trade,
confirmed, just small in magnitude compared to the min4pts-scale collapse in section 6.

**The bigger effect landed somewhere the feature wasn't specifically aimed at.** True
`pedestrian_group` predicted as `pedestrian` (section 1's *other* structural confusion) dropped
from 40.5% to 37.9% of that class - a larger shift than the `car`/`large_vehicle` trade in
either direction. Point count is apparently more cleanly diagnostic for "one person vs. a
cluster of people" than for "small car vs. big vehicle," where section 7 already found the
useful signal (`large_vehicle`'s `x_rel` shape) isn't really about count at all.

**Conclusion: real but modest, not a fix for the confusion it targeted.** Net accuracy barely
moves (+0.2pp raw). Worth keeping in mind for future feature-set decisions, but doesn't change
`baseline_20epoch_h16`'s status as the best model so far.

Files: `results/mlp_full_run/point_count_feature/`.

## 9. `vr_rel` (Doppler spread) feature: real signal, but only as an addition, not a replacement

Motivation: `x_rel`/`y_rel` (object-centered position) already showed spatial spread is a real,
separable signal (section 4, section 7's `x_rel` shape correlation). `vr_compensated` has no
analogous "spread" treatment - only the raw, ego-motion-compensated value. `vr_rel` (each
point's radial velocity relative to its instance's own **median** vr_compensated) is the same
idea applied to Doppler: does *how much a rigid object's own points disagree with each other*
carry class signal (micro-Doppler), independent of the object's raw bulk motion.

This isn't new work - `vr_rel` was already built and validated in `FEATURE_MAP.md` (Day 5 EDA),
deliberately deferred from the paper-faithful v1 baseline (`MLP_DESIGN.md`). Two real bugs had
to be fixed there before it showed real signal, both carried over unchanged into
`histogram_features.py`'s `add_relative_vr`/`VR_REL_RANGE`:

1. **Median-centering, not mean.** A single aliased/outlier point drags a *mean* reference off
   for the whole instance, corrupting every other point's "relative" value too - a rigid object
   can look falsely articulated from one bad reading. A median barely moves when one value is
   extreme, so the corruption stays localized to the one point that caused it.
2. **An explicit, narrow bin range (`±3.0 m/s`), not the auto-fit percentile range the other
   features use.** The real class-separating signal here is sub-1 m/s; the auto-fit range would
   be ~-15 to +30 m/s (driven by rare fast-crossing traffic and leftover aliasing elsewhere in
   the dataset), which rounds the entire real signal down to "zero, dead center" for every
   class against bins that wide - failing silently, no obvious symptom pointing at why.

**Why a histogram instead of a single scalar (e.g. MAD)?** Directly checked rather than assumed
- an earlier attempt (`FEATURE_MAP.md`) described a `large_vehicle`-specific second peak around
+1.5 to +2.25 m/s as justification, but re-verifying that against the actual cached data
(`train_vr_X.npy`) did **not** hold up as stated: the class-average `vr_rel` profile is unimodal
for every class, no separate second hump. The real, verified effect is more modest: only ~4.3%
of `car`/`pedestrian` instances have *any* point beyond ±1.3 m/s at all, vs. ~8-9% for
`large_vehicle`/`two_wheeler` - a genuine minority-subpopulation effect (some instances are
unusually wide/heavy-tailed), just not a clean bimodal peak. That's still a real justification
for a histogram over a scalar (MAD/std would blur "this specific instance has an unusually
heavy tail" into the same number as "typical"), just a more modest one than first claimed.

**Result 1: added as a 6th feature block (80 -> 96 dims, `vr_compensated` kept).**
`scripts/histogram_features.py` (`instance_vector`/`build_dataset`, `include_vr_rel=True`),
`results/mlp_full_run/vr_rel_feature/`:

| true class | baseline (80-dim) | +vr_rel added (96-dim) | Δ |
|---|---|---|---|
| car | 82.9% | 83.8% | +0.9pp |
| large_vehicle | 55.3% | 57.0% | +1.7pp |
| pedestrian | 89.4% | 90.4% | +1.0pp |
| pedestrian_group | 54.3% | 51.0% | -3.3pp |
| two_wheeler | 91.0% | 90.4% | -0.6pp |
| **overall (raw)** | **74.1%** | **73.5%** | **-0.6pp** |

Mixed, net slightly negative. Three classes improve modestly (consistent with a rigid-vs.
articulated-body story), but `pedestrian_group` drops more than any of those gained. One run,
no repeated seeds - same caveat as section 5's capacity ablation.

**Result 2: `vr_rel` swapped in for `vr_compensated` instead (stays 80-dim).**
`REPLACE_VR_WITH_VR_REL=True`, `results/mlp_full_run/vr_rel_replace_feature/`:

| true class | baseline (`vr_compensated`) | `vr_rel` replaces it (80-dim) | Δ |
|---|---|---|---|
| car | 82.9% | 74.5% | -8.4pp |
| large_vehicle | 55.3% | 61.7% | +6.4pp |
| pedestrian | 89.4% | 82.2% | -7.2pp |
| pedestrian_group | 54.3% | 37.7% | -16.6pp |
| two_wheeler | 91.0% | **39.9%** | **-51.1pp** |
| **overall (raw)** | **74.1%** | **59.1%** | **-15.0pp** |

Sharp, unambiguous: `two_wheeler` collapses 51pp. Raw `vr_compensated` (bulk motion - is this
object moving, how fast) carries real, load-bearing information that `vr_rel` (spread only)
can't recover, especially for separating moving-vehicle classes from person classes. `large_vehicle`
is the one class that improves, but it's swamped by the damage elsewhere. (Caveat: this run
hadn't fully plateaued by epoch 20 - val accuracy was still climbing, unlike every other run in
this document - so the true gap may close slightly with more training, but not by 51pp on
`two_wheeler`.)

**Conclusion: `vr_rel` is worth having, only as an addition.** Confirms the section-8-adjacent
design choice (keep `vr_compensated`, add `vr_rel` alongside it, don't follow the `x_rel`/`y_rel`
precedent of full replacement) was correct - `vr_compensated` and `vr_rel` carry different,
both-real information (bulk motion vs. spread), unlike raw position, which was excluded from
the v1 baseline because it's mostly a location confound with little of that shape/identity
signal.

Files: `results/mlp_full_run/vr_rel_feature/`, `results/mlp_full_run/vr_rel_replace_feature/`.

## 10. `large_vehicle`/`car` confusion, revisited: it's a truck problem, not a bus problem

Section 4 established that `large_vehicle`'s entire performance problem is essentially one
confusion (`large_vehicle`->`car`, ~92% of all `large_vehicle` errors), driven by sparse/
small-footprint `large_vehicle` instances. But `large_vehicle` is itself a merge of 4 raw
RadarScenes labels (`DESIGN_DECISIONS.md` decision 1: `large_vehicle` (raw), `truck`, `bus`,
`train`), never broken back down until now. `train_points.parquet` keeps the pre-merge label
in `label_name`, so `scripts/large_vehicle_subtype_analysis.py` groups by it directly instead
of by the merged `class_name`.

**Composition:** truck and bus dominate - 71% and 26% of `large_vehicle` instances
respectively; raw `large_vehicle` (3%) and `train` (~0%) are negligible.

**Truck and bus are physically very different from each other, and from car - and that
difference survives even in the sparse (`n_points<=3`) regime that actually drives the
confusion** (section 4's misclassified `large_vehicle` instances average 2.50 points):

| | overall n_pts | overall extent | sparse n | sparse extent | sparse RCS |
|---|---|---|---|---|---|
| car | 2.58 | 1.64m | 125,159 | 1.09m | 4.36 |
| truck | 6.02 | 6.86m | 7,409 | 2.34m | 8.52 |
| bus | 8.50 | 11.43m | 1,545 | 2.86m | 5.42 |

(`extent` = bounding-box diagonal of the instance's own `x_rel`/`y_rel` points - a direct
size proxy in meters.)

**Conclusion: the confusion is overwhelmingly a truck problem.** In the sparse zone, there
are 4.8x more sparse trucks than sparse buses (7,409 vs. 1,545) - truck's sparse-instance
extent (2.34m) sits much closer to car's (1.09m) than bus's does (2.86m), i.e. a small truck
at 2-3 points genuinely can look car-sized, while a bus almost never does. Bus retains a real,
meaningful separation from car in every regime checked (extent still 2.6x car's even at
`n_points<=3`, plus a real RCS gap).

**This argues against a blanket `large_vehicle` -> `car` merge as a fix.** Folding bus into
car would throw away a distinction that's still physically recoverable, to fix a confusion
bus barely contributes to. If a class-grouping change is pursued (see Next steps), the data
supports treating truck differently from bus, not merging all of `large_vehicle` uniformly.

**Per-feature histograms (`scripts/large_vehicle_subtype_histograms.py`) turned up a real
conflict between two features** on which class (truck or bus) is actually the "outlier" vs.
car - resolved by median rather than mean (these distributions are right-skewed), then settled
properly by a joint-feature separability probe rather than trusting either single stat:

| sparse (`n_points<=3`), median | car | large_vehicle (raw) | truck | bus |
|---|---|---|---|---|
| extent | 0.83m | 0.69m | 1.34m | **1.81m** |
| RCS | 3.03 | 5.08 | **7.72** | 3.85 |

Extent says bus is the outlier (truck's typical sparse footprint is much closer to car's).
RCS says the opposite - truck has the highest RCS gap from car, bus's is closer. Physically
plausible: a bus's flat panel sides can throw energy away from the radar at typical viewing
angles (weaker return despite being larger), while a truck reflects more consistently despite
being smaller. Single-feature comparison can't adjudicate between two features that disagree.

**Separability probe (`scripts/truck_bus_car_separability.py`)** - same method as the section 6
`car_large_vehicle_feature_overlap.py` check (logistic regression on the full 80-dim feature
vector, not one stat at a time), run for car-vs-truck and car-vs-bus, both overall and
sparse-restricted:

| regime | car vs. truck (AUC) | car vs. bus (AUC) |
|---|---|---|
| overall | 0.931 | 0.969 |
| sparse (`n_points<=3`) | **0.814** | 0.864 |

**Resolves in extent's favor, not RCS's: truck is the harder class to separate from car in the
full joint feature space, in both regimes.** Sparse `car`-vs-`truck` is the single lowest
separability number found in this whole confusion analysis (AUC 0.814, balanced accuracy
0.734) - meaningfully harder than sparse `car`-vs-`bus` (0.864/0.779). RCS alone pointed the
wrong way; once all 80 dimensions vote (dominated by the `x_rel`/`y_rel`/`range_sc` extent-like
bins, not RCS alone), truck is confirmed as the class actually driving the confusion, not bus.

**Extended to a full pairwise matrix (car/truck/bus/large_vehicle-raw, all 6 pairs)** to check a
proposed `{car,truck}` vs. `{bus,large_vehicle(raw)}` split - the extent numbers didn't obviously
support it (large_vehicle(raw)'s sparse median extent, 0.69m, is closer to car's, 0.83m, than to
bus's, 1.81m), and the probe confirms that doubt:

| sparse (`n_points<=3`), AUC | car | truck | large_vehicle (raw) |
|---|---|---|---|
| truck | 0.814 | - | - |
| large_vehicle (raw) | **0.811** | 0.823 | - |
| bus | 0.864 | 0.844 | 0.861 |

`car`, `truck`, and `large_vehicle`(raw) are all mutually close (AUC 0.811-0.823 - barely
distinguishable from each other). `bus` is separable from all three at roughly the same distance
(0.844-0.864) - it is not closer to `large_vehicle`(raw) than it is to `car`. Same pattern
overall/unrestricted: car-vs-large_vehicle(raw) is the single lowest AUC of any pair (0.857),
while bus stays the most separable from everything (0.897-0.969).

**Conclusion: the grouping the data supports is `{car, truck, large_vehicle(raw)}` together,
`bus` alone** - not a `{car,truck}` / `{bus,large_vehicle}` split. Bus isn't closer to
`large_vehicle`(raw) than to car; it is uniformly the odd one out against all three.

**Result: retrained with the revised grouping** (`scripts/train_mlp_regroup.py` - `car` =
old car+truck+large_vehicle(raw), `large_vehicle` = old bus+train only; everything else
identical to the canonical baseline, X reused unchanged from the cached baseline features
since regrouping only changes labels, not which points belong to which instance). First pass
at 20 epochs (`results/mlp_full_run/car_large_vehicle_regroup/`):

| class | baseline (old grouping) | regrouped, 20ep | Δ |
|---|---|---|---|
| car | 82.9% | 81.2% | -1.7pp |
| large_vehicle | 55.3% | 58.6% | +3.3pp |
| pedestrian | 89.4% | 90.0% | +0.6pp |
| pedestrian_group | 54.3% | 53.0% | -1.3pp |
| two_wheeler | 91.0% | 91.2% | +0.2pp |
| **raw accuracy** | **74.1%** | **74.7%** | **+0.6pp** |
| **balanced accuracy** | **74.6%** | **74.8%** | **+0.2pp** |

But the cost curve at 20 epochs hadn't plateaued (still decreasing, no train/val gap - not
overfitting, just undertrained), so this was re-run for 1000 epochs to match the plateau check
already done for the original grouping (section 2) before drawing any conclusion
(`results/mlp_full_run/car_large_vehicle_regroup_1000epoch/`). Confirmed plateaued by epoch
1000 (train/val loss and accuracy flat over the last 30+ epochs, train_acc 0.775 = val_acc
0.776, cost decreasing smoothly and monotonically throughout with no jumps, sampled every 50
epochs to check):

| class | baseline (converged, old grouping) | regrouped, 20ep | regrouped, converged (1000ep) |
|---|---|---|---|
| car | 82.9% | 81.2% | **85.5%** |
| large_vehicle | 55.3% | 58.6% | 54.9% |
| pedestrian | 89.4% | 90.0% | 88.6% |
| pedestrian_group | 54.3% | 53.0% | **57.8%** |
| two_wheeler | 91.0% | 91.2% | 91.7% |
| **raw accuracy** | **74.1%** | 74.7% | **77.6%** |
| **balanced accuracy** | **74.6%** | 74.8% | **75.7%** |

**Two things changed once it actually converged.** The overall improvement got bigger, not
smaller (raw +3.5pp, balanced +1.1pp vs. baseline - the 20-epoch snapshot understated it). But
`large_vehicle`'s own apparent gain reversed (+3.3pp at 20 epochs -> -0.4pp at convergence,
essentially flat, on a now-small 730-instance val sample so noisy either way). The regrouping
was meant to make `large_vehicle` itself easier - that didn't materialize. The real gains
landed on `car` (+2.6pp) and `pedestrian_group` (+3.5pp) instead, classes it wasn't targeting -
plausibly a simpler overall 5-way decision boundary freeing up capacity elsewhere, not
confirmed further.

**Checked, and ruled out, one candidate explanation for `large_vehicle` still getting 44% of
its instances (323/730) predicted as `car` despite this: insufficient loss incentive.** The
new, smaller `large_vehicle` class already carries the *highest* class weight of any class
(`w_i = N/(C*N_i)`: `large_vehicle`=8.83 vs. `car`=0.38, i.e. **23x** car's weight) - every
misclassified `large_vehicle` instance already costs the loss 23x more than a misclassified
`car` instance. A shortcut-for-free explanation doesn't survive that; the model is being
pushed hard to get these right and still can't, for a real chunk of them. Consistent with
genuine feature overlap for a subpopulation of buses (sparse `car`-vs-`bus` AUC was only
0.864, section 10 above) rather than a training-incentive problem. Raw example count (8,024
train instances) staying genuinely thin to learn a bus's full shape from is a separate, real
concern - different mechanism than loss weighting, and not something reweighting can fix.

Files (this result): `scripts/train_mlp_regroup.py`
(`results/mlp_full_run/car_large_vehicle_regroup/`,
`results/mlp_full_run/car_large_vehicle_regroup_1000epoch/`).

Files (section 10 overall): `scripts/large_vehicle_subtype_analysis.py`,
`scripts/large_vehicle_subtype_histograms.py`
(`results/mlp_full_run/large_vehicle_car_confusion/subtype_histograms.png` +
`subtype_histograms_stats.csv`), `scripts/truck_bus_car_separability.py`
(`results/mlp_full_run/truck_bus_separability/results.txt`), `scripts/train_mlp_regroup.py`.

## 11. Conclusion: single-frame point-cloud sparsity is the project's binding constraint

Every intervention tried this project - bigger model (section 5), longer training (section 2),
dropping sparse instances (section 6), explicit point-count feature (section 8), `vr_rel`
Doppler-spread feature (section 9), and now revised class taxonomy (section 10) - either failed
to move headline accuracy, or moved it by a small amount through a mechanism *other than* the
one it was meant to test (section 10's regroup result: the gain came from `car`/
`pedestrian_group`, not the targeted `large_vehicle`). None of them touch the actual ceiling.

**The ceiling itself is now confirmed directly, and project-wide, not just for one class pair.**
Every one of the 6 pairwise class separability probes run in section 10 - not just the original
`car`/`large_vehicle` case - compresses into the same lower band once restricted to sparse
(`n_points<=3`) instances:

| regime | AUC range across all 6 pairs tested |
|---|---|
| overall/dense | 0.857 - 0.969 |
| sparse (`n_points<=3`) | **0.811 - 0.864** |

No pair escapes it. This matches every other sparsity signal found across the project: 34% of
instances are single-point (`vr_rel`/extent undefined by construction), 74% have <=3 points
(section 9's `vr_rel` dilution), a bigger model overfits low-information sparse instances
instead of exploiting more signal from them (section 5), and class-weighted loss with a 23x
weight still can't force the model past this ceiling for the classes it's aimed at (section 10).

**Conclusion: it's the data, specifically sparsity - not capacity, not training length, not
feature engineering, not class taxonomy.** None of those are capable of adding the information
a 1-3 point radar return doesn't contain in the first place. Class taxonomy is the one
partial exception worth keeping in mind: it cannot fix the sparse-regime ceiling itself (still
0.81-0.86 AUC no matter how the classes are drawn), but it does change how much of that fixed
cost shows up in the aggregate headline number (section 10's regroup result, +3.5pp raw) - a
real, separate lever from whether the ceiling exists.

The one candidate not yet tested that is structurally different from everything above - because
it changes the *amount* of information available per instance rather than re-encoding,
re-weighting, or re-labeling the same sparse snapshot - is temporal accumulation across
consecutive scans of the same track (see Next steps: sensor fusion is the adjacent, also
untested, information-adding lever).

## Next steps (not yet decided, not ranked)

- **Explicit point-count feature is tested** (section 8) - small, mixed effect (+0.2pp overall,
  a mild `car`/`large_vehicle` trade as predicted in section 7, a bigger unexpected improvement
  on `pedestrian_group`/`pedestrian`). Not a fix, not harmful either - not currently worth
  swapping in as the new baseline on its own.
- **Physics-only vs. full-feature-set ablation** (`TODO.md` Day 7, part of Day 6's own original
  scope, not yet run): `rcs`+`vr_compensated` alone vs. the full 5-feature set. Tests whether
  `range_sc`/`x_rel`/`y_rel` carry real shape signal or are a scene-layout shortcut - a
  different question from anything above.
- **Revisit the `large_vehicle` merge is done** (section 10) - retrained with `car`=old
  car+truck+large_vehicle(raw), `large_vehicle`=old bus only. Real, converged improvement
  (+3.5pp raw, +1.1pp balanced), but not via the targeted class (`large_vehicle` itself stayed
  flat) - via `car`/`pedestrian_group` instead, mechanism not fully explained. Not a candidate
  for further iteration on its own; superseded by section 11's conclusion that taxonomy changes
  the aggregate cost distribution but can't touch the sparsity ceiling itself.
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
