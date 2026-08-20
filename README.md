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
scripts/             data loading, table building, plotting, separability probes
notebooks/           EDA notebooks
results/             generated plots + cached parquet tables (gitignored, recreated by running the scripts)
data/                the RadarScenes dataset (gitignored)
Design_Decisions.md  taxonomy/encoding decisions and the evidence behind them
visualize.sh         rad_viewer launcher
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
- `inspect_scene(sequence_name, timestamp, track_id=None)` — the main entry point. Loads a scene, plots it next to the closest camera frame (`results/scene_plot.png`), auto-picks (or uses a given) tracked object, prints its attribute table, and plots that object colored by RCS and by Doppler (`results/object_attributes.png`).

```bash
python3 scripts/dataloader.py   # runs inspect_scene("sequence_2", <a busy sensor-2 timestamp>)
```

## `scripts/build_points_table.py`

Builds the `{points, attributes, label}` dataset as a long-format table: one row per radar
point belonging to a tracked (dynamic) object, restricted to sensor #2 (front right), across
all 158 sequences. Instance identity (`sequence_name`, `timestamp`, `track_id`) is repeated on
every point's row — group by those three columns to recover one object instance's point set.
`build_and_save_points_table()` skips rebuilding if `results/points_table.parquet` already
covers exactly the requested sequences.

```bash
python3 scripts/build_points_table.py   # writes results/points_table.parquet
```

## `scripts/class_imbalance.py`

`plot_class_imbalance(table_path=None)` loads the points table, counts labeled object
instances per class (one count per instance, not per point) with percentages, and plots/saves
a log-scale bar chart colored by class (`results/class_counts.png`). Returns `(summary, fig)`.

```bash
python3 scripts/class_imbalance.py
```

## `scripts/taxonomy_separability.py` / `scripts/separability_probe.py`

Answers a specific question from `Design_Decisions.md` decision 1: is merging
`large_vehicle`/`truck`/`bus` justified, or does it destroy separability? `separability_probe.py`
holds the reusable, feature-set-agnostic core (`run_probe`): a sequence-grouped held-out split
(`StratifiedGroupKFold` on `sequence_name`, with leakage asserts — an instance-level split would
leak sequence-specific background/weather/vehicle signal), class-count-weighted
`LogisticRegression` + `RandomForestClassifier`, per-class one-vs-rest and (LR) pairwise
ROC-AUC, and a saved confusion matrix. `taxonomy_separability.py` builds this investigation's
specific per-instance features (`rcs`/`vr_compensated` median, `x`/`y` extent, `doppler_spread`)
and calls `run_probe` on them; `add_relative_features()` (adds `x_rel`/`y_rel`/`doppler_spread`)
caches the expensive `doppler_spread` computation to `results/doppler_spread_cache.parquet`,
since it's a ~2-3 min per-instance MAD at full scale.

```bash
python3 scripts/taxonomy_separability.py   # prints report + AUCs, saves confusion matrices
```

## `scripts/feature_distributions.py` / `scripts/histogram_separability.py`

Day 5: deciding the per-instance histogram encoding's bin range/count for `rcs`,
`vr_compensated`, `x_rel`, `y_rel` (plus `doppler_spread`, appended unbinned since it's already
one value per instance). `feature_distributions.py` computes per-class percentiles and a
pooled bin-count sweep plot per feature (`results/bin_sweep_<feature>.png`), with the plotted
range clipped to `[p1, p99]` so a long tail doesn't blow out the axis. But that plot pools every
point across ~500k instances, which doesn't reflect what a single instance's histogram (~2.9
points on average) looks like — so `histogram_separability.py` builds the actual histogram-encoded
per-instance features at each candidate bin count and scores them via `separability_probe.run_probe`
(macro-average per-class AUC), rather than deciding the bin count by eye. Cached to
`results/bin_sweep_results.parquet`, keyed by the exact bin counts requested.

```bash
python3 scripts/feature_distributions.py   # writes results/bin_sweep_<feature>.png
python3 scripts/histogram_separability.py  # writes results/bin_sweep_results.parquet
```

## `notebooks/`

`data_analysis.ipynb` builds the points table, plots class imbalance, and runs the taxonomy
separability probe, interleaved with written findings and the decision 1 resolution.
`feature_distributions.ipynb` runs the per-feature percentile table and bin-count sweep plots,
with room for written bin-count notes.

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