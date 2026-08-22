# radar-ml-autonomous-driving

Deep learning on automotive radar data for autonomous driving perception, using the
[RadarScenes](https://radar-scenes.com/) dataset (158 real driving sequences, 4 radar
sensors + camera + odometry, point-wise semantic labels).

## Setup

```bash
python3 -m venv .radar_ml
source .radar_ml/bin/activate
pip install -r requirements.txt
```

Data is expected at `data/RadarScenes/RadarScenes/data/sequence_<N>/` (one folder per
sequence, each with `radar_data.h5`, `scenes.json`, and a `camera/` folder).

## Layout

```
scripts/             data loading, table building, plotting, separability probes, MLP classifier
notebooks/           EDA notebooks
results/             generated plots + cached tables (gitignored, recreated by running the scripts)
data/                the RadarScenes dataset (gitignored)
Design_Decisions.md               taxonomy/encoding/split decisions and the evidence behind them
MLP_Decisions_and_Findings.md     MLP architecture, hyperparameters, and findings, and the reasoning behind them
visualize.sh          rad_viewer launcher
```

## `scripts/dataloader.py`

Lightweight loader/plotter for one scene at a time (h5py + numpy + matplotlib, no heavy
dependencies). A **scene** is one measurement from *one* radar sensor at one timestamp —
a sequence interleaves ~4-5k scenes from its 4 sensors chronologically, it does not merge
them. A **track_id** groups the points within a scene that belong to one physical object
(a plain `label_id` only tells you the class, not which points form one instance).

Data layer:
- `load_scene(sequence_name, timestamp=None)` — one scene's detections as a structured array.
- `object_instance(detections, track_id)` — one tracked object's `{track_id, label_id, label_name, points}`. This is the `{points, attributes, label}` unit used for training data.
- `list_track_ids(detections)` / `largest_object(detections)` — enumerate or auto-pick objects in a scene.
- `SENSORS` — sensor_id → (name, x, y, yaw) mounting position in car coordinates (4 radars on the front bumper); `sensor_label(sensor_id)` formats it as e.g. `"sensor 2 (front right)"`. `axis_limits()` and the plot titles use this so a sensor's real mounting point is always in frame, not fixed at the car origin.
- `CLASS_GROUPS` — raw `label_name` → final training class (12 raw RadarScenes classes down to 6; see `Design_Decisions.md` decision 1 for why). `FINAL_CLASS_COLORS` is the matching plot palette.

Visualization:
- `inspect_scene(sequence_name, timestamp, track_id=None)` — the main entry point. Loads a scene, plots it next to the closest camera frame (`results/scenes/scene_plot.png`), auto-picks (or uses a given) tracked object, prints its attribute table, and plots that object colored by RCS and by Doppler (`results/scenes/object_attributes.png`).

```bash
python3 scripts/dataloader.py   # runs inspect_scene("sequence_2", <a busy sensor-2 timestamp>)
```

## `scripts/build_points_table.py`

Builds the `{points, attributes, label}` dataset as a long-format table: one row per radar
point belonging to a tracked (dynamic) object, restricted to sensor #2 (front right), across
all 158 sequences. Instance identity (`sequence_name`, `timestamp`, `track_id`) is repeated on
every point's row — group by those three columns to recover one object instance's point set.
`build_and_save_points_table()` skips rebuilding if `results/data/points_table.parquet` already
covers exactly the requested sequences.

```bash
python3 scripts/build_points_table.py   # writes results/data/points_table.parquet
```

## `scripts/class_imbalance.py`

`plot_class_imbalance(table_path=None)` loads the points table, counts labeled object
instances per class (one count per instance, not per point) with percentages, and plots/saves
a log-scale bar chart colored by class (`results/class_imbalance/class_counts.png`). Returns
`(summary, fig)`.

```bash
python3 scripts/class_imbalance.py
```

## `scripts/taxonomy_separability.py` / `scripts/separability_probe.py`

Answers a specific question from `Design_Decisions.md` decision 1: is merging
`large_vehicle`/`truck`/`bus` justified, or does it destroy separability? `separability_probe.py`
holds the reusable, feature-set-agnostic core: a sequence-grouped held-out split
(`StratifiedGroupKFold` on `sequence_name`, with leakage asserts — an instance-level split would
leak sequence-specific background/weather/vehicle signal), class-count-weighted
`LogisticRegression` + `RandomForestClassifier`, per-class one-vs-rest and (LR) pairwise
ROC-AUC, and a saved confusion matrix. `run_probe` does one held-out fold; `run_probe_cv` reruns
it across `n_splits` folds and reports mean ± std, the version to actually trust (see decision
1's "Confirmed with 5-fold CV" subsection). `taxonomy_separability.py` builds this
investigation's specific per-instance features (`rcs`/`vr_compensated` median, `x`/`y` extent,
`doppler_spread`) and calls into `separability_probe.py` on them; `add_relative_features()`
(adds `x_rel`/`y_rel`/`doppler_spread`) caches the expensive `doppler_spread` computation to
`results/data/doppler_spread_cache.parquet`, since it's a ~2-3 min per-instance MAD at full
scale. `plot_taxonomy_class_balance` and `plot_cv_fold_class_counts` visualize why the rarer of
these classes (`large_vehicle`) has such noisy cross-fold metrics — sequence coverage, not just
instance count.

```bash
python3 scripts/taxonomy_separability.py   # prints the 5-fold CV report, saves confusion matrices to results/taxonomy/
```

## `scripts/sequence_split.py`

Design_Decisions.md decision 5: a fixed train/val/test split (~70/15/15) by *sequence*, not
instance or scan, via two chained `StratifiedGroupKFold` calls (grouped by `sequence_name`,
stratified by final class). Every probe/model past this point reuses the same split — `val` for
freely comparing candidates, `test` checked exactly once at the end — instead of each one
picking its own held-out fold. Cached to `results/data/sequence_split.json`; `load_split()`
reads it back without needing a `df`.

```bash
python3 scripts/sequence_split.py   # writes results/data/sequence_split.json
```

## `scripts/feature_distributions.py` / `scripts/histogram_separability.py`

Deciding the per-instance histogram encoding's bin range/count for `rcs`, `vr_compensated`,
`x_rel`, `y_rel` (plus `doppler_spread`, appended unbinned since it's already one value per
instance). `feature_distributions.py` computes per-class percentiles and a pooled bin-count
sweep plot per feature (`results/feature_distributions/bin_sweep_<feature>.png`), with the
plotted range clipped to `[p1, p99]` so a long tail doesn't blow out the axis. But that plot
pools every point across ~500k instances, which doesn't reflect what a single instance's
histogram (~2.9 points on average) looks like — so `histogram_separability.py` builds the
actual histogram-encoded per-instance features at each candidate bin count and scores them via
`separability_probe.run_probe_cv` (5-fold mean ± std), restricted to the fixed split's
train+val sequences, rather than deciding the bin count by eye or against a single fold.
`fit_bin_edges()` fits percentile-based edges from a given `df` alone — pass a train-only `df`
and reuse the returned edges for val/test, since edges are a fitted preprocessing parameter.
Cached to `results/histogram_separability/bin_sweep_results.parquet`, keyed by the exact bin
counts requested.

```bash
python3 scripts/feature_distributions.py   # writes results/feature_distributions/bin_sweep_<feature>.png
python3 scripts/histogram_separability.py  # writes results/histogram_separability/bin_sweep_results.parquet
```

## `scripts/batch_size_selection.py`

Picks the MLP's batch size (and a matching learning rate) from the train split's actual class
frequencies instead of guessing — a batch of size `bs` misses a class entirely with probability
`(1-p)^bs`, so this solves for the smallest batch size keeping every class's miss probability
under a threshold, then scales the learning rate with it via the linear scaling rule. Prints
its recommendation, doesn't write to `mlp_classifier.py` itself (see
`MLP_Decisions_and_Findings.md` section 1 for the numbers and reasoning it produced).

```bash
python3 scripts/batch_size_selection.py
```

## `scripts/mlp_classifier.py`

The baseline classifier: a 3-layer MLP (65 → 16 → 16 → 6) on the histogram-encoded per-instance
feature vector, trained with Adam and class-count-weighted cross-entropy on the fixed
train/val/test split. Architecture, hyperparameters, the reasoning behind each, and the
confusion matrix and per-class precision/recall/f1 findings are all in
`MLP_Decisions_and_Findings.md`. `run_training()` caches the trained model and history keyed by the exact
config, so a repeat call with the same config skips straight to the cached result rather than
retraining; `evaluate_val_metrics()` similarly only ever loads the cached model, never retrains.
Both take a `classes` list and an `output_dir`, letting a caller train/evaluate a different
class taxonomy or hyperparameter config without touching the standing baseline's cache — see
`scripts/mlp_variants.py`.

```bash
python3 scripts/mlp_classifier.py   # trains (or loads from cache) and writes to results/mlp/
```

## `scripts/mlp_variants.py` / `scripts/class_taxonomy_experiment.py`

`mlp_variants.py` is the single source of truth for every trained MLP variant produced so far
(`baseline`, `bus_separate`, `epochs_1000`, `truck_separate`) — each entry in `MLP_VARIANTS` maps
a name to its `classes`/`class_groups`/`output_dir`/hyperparameter overrides, so running or
loading one is `run_variant(raw_df, "bus_separate")` instead of hand assembling those per call.
`baseline` (bus merged into large_vehicle) is the standing model and mlp_classifier.py's own
default; `bus_separate` is the original, pre-merge taxonomy, kept for reference.
`class_taxonomy_experiment.py` is the ablation that led to that merge decision: does keeping
`bus` separate actually help the downstream MLP, versus merging it in, or splitting `truck`
back out instead? See `MLP_Decisions_and_Findings.md` section 4 for the results.

```bash
python3 scripts/mlp_variants.py <variant>              # defaults to "baseline"
python3 scripts/class_taxonomy_experiment.py <variant> # defaults to running both taxonomy variants
```

## `notebooks/`

`data_analysis.ipynb` builds the points table, plots class imbalance, builds the fixed
train/val/test split, and runs the taxonomy separability probe (single-fold and 5-fold CV),
interleaved with written findings and the decision 1 resolution. `feature_distributions.ipynb`
runs the per-feature percentile table and bin-count sweep (single-fold and 5-fold CV), with the
decision 3 resolution. `mlp_classifier.ipynb` builds features, trains (cached) the baseline MLP,
and reports its validation metrics, confusion matrix, and per-class precision/recall/f1 —
interleaved with the `MLP_Decisions_and_Findings.md` decisions and findings.

## `visualize.sh`

Opens the official [radar_scenes](https://github.com/oleschum/radar_scenes) `rad_viewer`
Qt GUI for a full sequence — nicer/interactive, but heavy (PySide6 + pyqtgraph), so it's
kept as an occasional look-at-the-data tool rather than part of the pipeline.

```bash
chmod +x ./visualize.sh
./visualize.sh [sequence_number]   # defaults to sequence_1
```

On WSL2 without WSLg, this needs a working X server on the Windows host (`DISPLAY` is set
in the script) and the system package `libxcb-cursor0` (`sudo apt-get install -y libxcb-cursor0`).
