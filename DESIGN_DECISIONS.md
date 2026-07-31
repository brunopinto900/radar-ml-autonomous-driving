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
