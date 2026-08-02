# Design Decisions

Decisions that shape the dataset, made deliberately rather than defaulted into.
Standing rule: LLM writes the boilerplate, these calls are mine (Bruno's).

## 1. Class grouping: merge train/truck/bus/large_vehicle, two_wheeler, drop animal/other_dynamic

RadarScenes ships 12 raw classes (`label_id` 0-11). Trained directly on those, several
are too rare to learn or evaluate reliably. From a 5-sequence, single-sensor sample:

| class      | instances |
|------------|-----------|
| car        | 5663      |
| pedestrian | 1406      |
| truck      | 1317      |
| pedestrian_group | 1058 |
| bus        | 196       |
| other_dynamic | 187    |
| large_vehicle | 120    |
| train      | 57        |
| bicycle    | 45        |

`train` in particular is bad enough that it forces bad tradeoffs elsewhere: it appears
in so few sequences that a sequence-level train/val/test split (see decision 2) can't
land anywhere near a target ratio for it without dragging along whatever else those
same sequences contain.

**Decision:** reduce to 6 classes, matching the official `radar_scenes` package's own
`ClassificationLabel` grouping (`radar_scenes/labels.py`) - this isn't just a
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
static             -> static
animal             -> (dropped, not trained on)
other_dynamic       -> (dropped, not trained on)
```

`animal` and `other_dynamic` are dropped rather than merged anywhere - they're
catch-all/low-signal classes in the original dataset design too (the official scheme
maps them to `None`), not a coherent category to teach a classifier.

**Caveat, not yet checked:** this assumes train/truck/bus share a similar-enough radar
signature to large_vehicle that merging doesn't destroy separability. Day 4's planned
EDA (RCS/point-count distributions per class) should confirm this rather than just
trusting the precedent - if train looks radically different, revisit.

Implemented in `scripts/dataloader.py` (`CLASS_GROUPS`, `GROUP_COLORS`), applied in
`scripts/build_points_table.py` (adds `class_name` column, drops dropped classes).

## 2. Train/val/test split: sequence-level, official validation = test

RadarScenes ships a 2-way split (`sequences.json`: 130 `train` / 28 `validation`
sequences), not the standard 3-way train/val/test. Standard practice wants three:
train (fit), val (tune/iterate on, checked repeatedly), test (final number, touched once).

**Decision:**
- The unit of assignment is the whole **sequence**, never an individual scene or
  instance. Reasons: (a) a sequence's objects are correlated (same physical car
  tracked across thousands of scenes, same road/background) - splitting within a
  sequence leaks information between splits; (b) it keeps each sequence usable as a
  contiguous time series if/when temporal modeling happens later (not now, per
  TODO.md's single-frame scope for v0.1).
- The 28 official `validation` sequences become the **test** set - final number,
  not touched during iteration.
- The 130 official `train` sequences are further split 85/15 (fixed
  `random_state=42`) into internal **train** (110 sequences) / **val** (20
  sequences), giving ~70/13/17 overall - close to a standard 70/15/15 without
  disturbing the fixed test size.
- This is a **plain random sequence-level split**, not a stratified one. Multi-label
  stratified group splitting (balancing several classes' proportions simultaneously
  while keeping whole sequences intact) doesn't reduce to plain `sklearn`
  stratification, which only balances one label across freely-movable individual
  rows - see the worked example in conversation history. A proper solution needs an
  iterative/greedy stratification algorithm; that's more engineering than this
  project's timeline justifies right now, especially after decision 1 already
  removed the worst-offending rare class.
- **Known limitation, not yet checked:** a plain random split can still leave a rare
  class (e.g. `two_wheeler`, 45 instances in the 5-sequence sample) badly
  under/over-represented in val by chance. Plan is to verify per-class counts per
  split once the points table covers more sequences, and manually swap a sequence or
  two if something's degenerate, rather than building a full stratification
  algorithm.

Implemented in `scripts/make_split.py`, output at `results/sequence_splits.csv`
(sequence_name -> train/val/test).

## 3. Histogram encoding for Day 6: paper-faithful feature space, bin counts from a sweep

Day 5/6 needs a fixed-size input representation per instance for the MLP. Landed on
Tatarchenko & Rambach's histogram-encoding scheme (see `TODO.md` Day 8 for the paper) after
first trying and rejecting a point-wise (no grouping at all) design - point-wise solves the
real-world object-association problem for free, but without any per-object reference frame,
position features can only encode *where* an object was recorded, never its shape, and no
spread/shape signal is computable from a single point in isolation (full reasoning in
`FEATURE_MAP.md`). Per-instance histograms need `track_id` grouping at training time; the
real-world clustering gap this doesn't solve is tracked separately (`TODO.md` v2).

**Feature space: matches the paper exactly, not our own `_rel`/spread extensions (yet).**
Their features are radial distance, ego-motion-compensated Doppler velocity, RCS, and
Cartesian `x`/`y`/`z` computed *relative to the tracked object's own center* (their reasoning:
"the object shape in Cartesian coordinates is independent of the distance between the object
and the radar sensor"). Only `x`/`y`/`z` are object-centered in their method - radial distance
stays raw. Mirrored exactly for the sensor-#2, 2D case: `rcs` (raw), `vr_compensated` (raw),
`range_sc` (raw), `x_rel`/`y_rel` (relative to each instance's own **mean** position - mean,
not median, since "centroid" means center of mass and there's no established
single-outlier-corruption problem for position the way aliasing corrupts `vr_compensated`).
The relative/de-meaned "spread" features found in `FEATURE_MAP.md` (`vr_rel` with
median-centering, `extent_rel`, `rcs_rel`, `range_rel`) go beyond what the paper does and are
deliberately deferred to `TODO.md`'s v1.1 future work - build the paper-faithful baseline
first, add the extensions once it's working, not before.

**Bin count: 16 bins per feature, uniform across all five.** Checked with
`scripts/bin_sweep.py` - a population-level (all points, all instances of a class, pooled;
individual instances are usually far too few points to meaningfully validate bin count on
their own) sweep of [8, 16, 32, 64] bins per feature, by class. Findings:

- `rcs`: 16-32 bins resolves the real two-cluster class separation cleanly; 64 starts showing
  jagged tail noise (bins with too few points to be stable).
- `vr_compensated`: needed finer bins than the others to show its real structure - 8-16 bins
  looks flat, 32 bins reveals genuine multi-modal traffic-speed clusters in `car`/
  `large_vehicle` and a real double-hump in `two_wheeler`; 64 turns visibly noisy.
- `range_sc`: tolerates fine bins better than the rest - a `pedestrian_group` secondary bump
  around 35-45m sharpens cleanly all the way to 64 bins without much added noise.
- `x_rel`/`y_rel`: `pedestrian`'s peak density keeps climbing as bins get finer (not noise -
  its spatial footprint is genuinely near a single point, per `EDA.md` item #4's >54%
  single-point instances, so finer bins just keep resolving the same near-delta-function
  shape). `large_vehicle` stays broad and flat at every resolution checked - the real signal.
  Separation is already clear by 16 bins; finer bins add no new information here.

**Decision:** one uniform `N_BINS = 16` across all five features for the Day 6 baseline -
comfortably inside the "shows real structure, not yet noisy" range for every feature checked,
close to the paper's own uniform `K = 20`, and keeps the encoding simple (matches their
choice of one fixed bin count rather than tuning per feature). `vr_compensated` specifically
would benefit from finer bins (32) to show its multi-modal structure in full - noted as a
first candidate for Day 7's iteration pass if results look off, not built in from the start.

Implemented in `scripts/instance_histograms.py` (per-instance histograms, currently `N_BINS =
8` there - not yet updated to match this decision) and `scripts/bin_sweep.py` (the sweep
itself, population-level, `results/bin_sweep/`).
