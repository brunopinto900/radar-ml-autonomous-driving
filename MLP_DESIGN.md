# MLP Design (Day 6)

Consolidates the model design decided across `DESIGN_DECISIONS.md`, `FEATURE_MAP.md`, and
`TODO.md`'s Day 6 line into one reference. Covers input/output shape, architecture, training
setup, and what's still open. Not implemented yet - this is the plan the pipeline gets built
against.

## 0. Pipeline smoke test - before the real run

Standard practice: most bugs in a pipeline like this are
plumbing (misaligned labels, wrong tensor shapes, a bin-edge computation subtly off, a loss
that isn't actually decreasing), not architecture or hyperparameters. Validate the workflow
cheaply before spending time on the real run.

- **Overfit a tiny slice first, before anything bigger.** A couple hundred instances, 1 hidden
  layer, train until it gets close to 100% *training* accuracy on that tiny set. Sharper
  diagnostic than jumping straight to a larger reduced run: with a real bug (misaligned
  labels, broken loss, wrong shapes), the model typically *can't* overfit even a trivial
  amount of data - an unambiguous failure signal, unlike "a reduced run gave 45% accuracy,"
  which could mean either "bug" or "not enough data/capacity" and doesn't tell you which.
- **Then step up to a reduced run** (e.g. 25% of train, 1 hidden layer) with a small val slice
  too - not just training data, so the eval/confusion-matrix code path gets exercised as well,
  not only the training loop.
- **Sample randomly, not positionally.** `train_points.parquet` is built by concatenating
  sequences one after another (`build_points_table.py`'s `pd.concat`) - taking the first N%
  of *rows* risks a subset dominated by whichever sequences happen to come first, possibly
  missing some classes entirely. Take a random sample of *instances* (fixed seed, same
  discipline as everything else in this project), not a positional slice.
- **Reuse the real pipeline code**, not a simplified stand-in written just for the small test.
  The point is validating the code that will run for real - if the small test passes using
  throwaway logic instead of the actual histogram/bin-edge functions, nothing about the real
  pipeline has actually been proven.
- **Don't read anything into the accuracy number from this phase.** It exists to answer "does
  the code work," not "is this approach good" - a tiny model on a quarter of the data,
  correctly implemented, is still expected to perform badly. That question waits for the real
  run (full architecture, full data, per the sections below).

## Input representation

Per-instance histogram encoding, matching Tatarchenko & Rambach's method (see `TODO.md` Day
8) - not point-wise (tried and rejected, see `FEATURE_MAP.md`: without per-object grouping,
position features can only encode environment, not shape, and no spread/shape signal is
computable from a single point).

One instance = one `(sequence_name, timestamp, track_id)` group - one tracked object, one
radar scan. Built via `track_id` grouping at training time; doesn't solve the real-world
object-association gap (`TODO.md` v2).

**Features** (`M = 5`, matches the paper's own feature set - see `DESIGN_DECISIONS.md`
decision 3):

| feature | source | notes |
|---|---|---|
| `rcs` | raw | RCS [dBsm] |
| `vr_compensated` | raw | ego-motion-compensated radial velocity [m/s] |
| `range_sc` | raw | matches paper - only x/y/z are object-centered in their method, not range |
| `x_rel` | relative to instance's own **mean** x | object-centered, per paper's reasoning (shape independent of range) |
| `y_rel` | relative to instance's own **mean** y | same |

Not included in the v1 baseline: `azimuth_sc` (redundant with `x`/`y`), and the `_rel`/spread
extensions beyond the paper (`vr_rel`, `extent_rel`, `rcs_rel`, `azimuth_rel`) - real signal
found in `FEATURE_MAP.md`, deliberately deferred to `TODO.md` v1.1 until the paper-faithful
baseline works.

**Bins**: `K = 16` per feature, uniform across all five (`DESIGN_DECISIONS.md` decision 3,
from a population-level sweep in `scripts/bin_sweep.py`). Bin edges are global (dataset-wide,
computed from the **train split only** - no val/test leakage), percentile-based (0.5-99.5%)
except `vr_rel`-style explicit overrides where needed (not used in this raw-feature v1 set).

**Counts, not density.** Raw per-bin point counts, not normalized - matches the paper, lets
the histogram implicitly encode instance size (point count) alongside shape.

**Shape**: each instance's `M` histograms are flattened and concatenated into one vector of
length `M * K` = `5 * 16` = **80**. Full training set: `N_instances * 80`.

## Output

Softmax over the 5 final classes: `car`, `large_vehicle`, `pedestrian`, `pedestrian_group`,
`two_wheeler` (`DESIGN_DECISIONS.md` decision 1). Shape: `N_instances * 5`. `argmax` over the
class dimension gives the single predicted label per instance for the confusion matrix.

## Labels and loss

Labels are **integer class indices** (0-4, `car` -> 0, `large_vehicle` -> 1, `pedestrian` ->
2, `pedestrian_group` -> 3, `two_wheeler` -> 4 - `class_to_idx` in
`scripts/histogram_features.py`), not one-hot vectors. `nn.CrossEntropyLoss` takes integer
targets directly.

**Class-weighted cross-entropy**: `w_i = N_samples / (N_classes * N_i)` (paper's formula) -
real imbalance is significant (`car` 160,818 vs. `two_wheeler` 21,280, ~7.6:1 in the train
split). Implemented in `scripts/train_mlp.py`'s `class_weights`.

## Architecture (paper-faithful baseline)

Starting from the paper's own numbers, for the same reason the feature set does - keeps the
Day 8 comparison an actual apples-to-apples check on design choices, not a different problem
framing entirely:

- 3 fully-connected layers, 16 hidden units each: `80 -> 16 -> 16 -> 5`
- Activation: not specified in the paper's method section - default to ReLU unless there's a
  reason to deviate (standard choice, no evidence yet that it matters here)
- Loss: class-weighted cross-entropy (see Labels and loss above)
- Optimizer: Adam, `lr = 1e-5` (paper's number; batch size overridden to 256, see
  `DESIGN_DECISIONS.md` decision 4)
- Epochs: paper trains 1000 - **confirmed excessive** on the real run. Val accuracy/loss
  plateau by ~epoch 20 with no overfitting (train/val track together the whole way); the
  remaining 980 epochs bought ~1-1.5pp raw val accuracy for ~60x the training time. Using the
  20-epoch checkpoint going forward. Full writeup, including why this isn't a local-minimum/lr
  problem, in `MLP_FINDINGS.md`.

## Data pipeline (not yet built)

Nothing computes the real per-instance feature vectors yet - everything so far
(`scripts/instance_histograms.py`, `scripts/bin_sweep.py`) is diagnostic (small example
samples or pooled population plots), not a function that maps an instance's points to its
80-dim vector for the full ~354k train instances. Needed:

1. Shared bin-edge computation (reuse `compute_bin_edges`-style logic), fit on train only.
2. A function: instance's points -> concatenated `M * K` histogram vector.
3. Applied across train/val/test splits, written out in a form the training loop can load.

## Open decisions - not yet resolved, don't let them become silent defaults

- **Low-point-count instances.** `pedestrian` is 54.4% single-point instances (not
  pedestrian-specific: `car` is 33.8% too - see `TODO.md` Day 6). A 1-point instance's 16-bin
  histogram is nearly empty. Interim plan: include everything in v1, check honestly (Day 7)
  whether accuracy is actually bad specifically on sparse instances before filtering/weighting
  them specially.
- **Evaluation metric.** Given the ~7.6:1 real imbalance, raw accuracy can look good while the
  model is actually bad at minority classes. Report balanced accuracy / per-class recall from
  the confusion matrix as the headline number, not plain accuracy.
- **Bin count is provisional.** `N_BINS = 16` came from a visual sweep. `TODO.md`'s "Histogram
  bin strategy" section lists 4 more rigorous options (formula-based, separability
  quantification, model-driven ablation, soft/Gaussian binning) - worth revisiting once the
  pipeline exists and a bin-count ablation is cheap to run.

## Planned ablations (Day 6/7)

1. **Physics-only vs. full feature set**: `rcs` + `vr_compensated` histograms alone vs. the
   full 5-feature set. Tests whether `range_sc`/`x_rel`/`y_rel` are contributing real shape
   signal or a scene-layout shortcut (`TODO.md` Day 6).
2. **Bin-count ablation**: `N_BINS` in `{8, 16, 32}`, compare validation accuracy directly.

## Known limitations carried into this design

- Requires `track_id`-based grouping at training time; the real-world clustering/object-
  association gap is not solved (`TODO.md` v2).
- Histograms have no cross-feature correspondence - can't represent true point-to-point
  geometric shape, only per-feature marginal statistics (`FEATURE_MAP.md`). A deliberate trade
  the paper also makes; not guaranteed to cost nothing on this dataset/task.
- `range_sc` stays raw (matches the paper) - may reintroduce the scene-position confound
  discussed throughout `FEATURE_MAP.md`. The physics-vs-full-feature ablation above is the
  direct check on how much this matters.
