# radar-ml-autonomous-driving

Deep learning on automotive radar data for autonomous driving perception, using the
[RadarScenes](https://radar-scenes.com/) dataset (158 real driving sequences, 4 radar
sensors + camera + odometry, point-wise semantic labels).

See [TODO.md](TODO.md) for the project plan and [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md)
for the reasoning behind the class grouping and train/val/test split.

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
scripts/     source (dataloader.py, build_points_table.py, class_imbalance.py,
             split_balance.py, make_split.py, check_gpu.py)
results/     generated tables/plots (gitignored, recreated by running the scripts)
data/        the RadarScenes dataset (gitignored)
visualize.sh   rad_viewer launcher (kept in root - it's an entry point, not a library)
```

## Pipeline

Run in this order (each reads a previous step's output from `results/`):

```bash
python3 scripts/make_split.py          # -> results/sequence_splits.csv
python3 scripts/build_points_table.py  # -> results/{train,val,test}_points.parquet
python3 scripts/class_imbalance.py     # -> results/class_counts.png
python3 scripts/split_balance.py       # -> results/split_balance.png
python3 scripts/dataloader.py          # -> results/scene_plot.png, object_attributes.png
```

- **`make_split.py`** — sequence-level train/val/test split (never scene/instance-level,
  to avoid leaking a tracked object's other appearances across splits). The 28 sequences
  RadarScenes itself marks `validation` become the held-out **test** set; the 130 marked
  `train` are further split 85/15 (fixed seed) into internal **train**/**val**. See
  DESIGN_DECISIONS.md decision 2 for why this is a plain random split, not stratified,
  and its known limitation (`two_wheeler` is skewed - tracked in TODO.md as v1.2 work).

- **`build_points_table.py`** — builds the actual `{points, attributes, label}` dataset:
  one row per radar point (long format), restricted to sensor #2 (front-right) for now,
  reduced from RadarScenes' 12 raw classes to 6 trainable ones (`CLASS_GROUPS` in
  `dataloader.py` - see DESIGN_DECISIONS.md decision 1). Writes one parquet per split.

- **`class_imbalance.py`** — instance-level class counts *within* the train split (log-scale
  bar chart). Counts instances, not raw points, so an object with many detections doesn't
  outweigh one with few.

- **`split_balance.py`** — sanity-checks the split itself: does each class's distribution
  across train/val/test match the expected sequence-count ratio, or did the random split
  skew a rare class into one split disproportionately?

## `scripts/dataloader.py`

Lightweight loader/plotter for one scene at a time (h5py + numpy + matplotlib, no heavy
dependencies). Also doubles as the shared module for the rest of the pipeline - every
other script imports its constants (`LABELS`, `CLASS_GROUPS`, `GROUP_COLORS`,
`OBJECT_ATTRS`, `DATA_ROOT`, `RESULTS_DIR`, `sensor_label`) rather than redefining them.

A **scene** is one measurement from *one* radar sensor at one timestamp — a sequence
interleaves ~4-5k scenes from its 4 sensors chronologically, it does not merge them. A
**track_id** groups the points within a scene that belong to one physical object (a plain
`label_id` only tells you the class, not which points form one instance).

Data layer:
- `load_scene(sequence_name, timestamp=None)` — one scene's detections as a structured array.
- `object_instance(detections, track_id)` — one tracked object's `{track_id, label_id, label_name, points}`. This is the `{points, attributes, label}` unit used for training data.
- `list_track_ids(detections)` / `largest_object(detections)` — enumerate or auto-pick objects in a scene.

Visualization:
- `inspect_scene(sequence_name, timestamp, track_id=None)` — the main entry point. Loads a scene, plots it next to the closest camera frame (`results/scene_plot.png`), auto-picks (or uses a given) tracked object, prints its attribute table, and plots that object colored by RCS and by Doppler (`results/object_attributes.png`).

```bash
python3 scripts/dataloader.py   # runs inspect_scene("sequence_1", <a busy timestamp>)
```

## `visualize.sh`

Opens the official [radar_scenes](https://github.com/oleschum/radar_scenes) `rad_viewer`
Qt GUI for a full sequence — nicer/interactive, but heavy (PySide6 + pyqtgraph), so it's
kept as an occasional look-at-the-data tool rather than part of the pipeline.

```bash
./visualize.sh [sequence_number]   # defaults to sequence_1
```

On WSL2 without WSLg, this needs a working X server on the Windows host (`DISPLAY` is set
in the script) and the system package `libxcb-cursor0` (`sudo apt-get install -y libxcb-cursor0`).
