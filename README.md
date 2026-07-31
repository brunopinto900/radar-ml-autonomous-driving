# radar-ml-autonomous-driving

Deep learning on automotive radar data for autonomous driving perception, using the
[RadarScenes](https://radar-scenes.com/) dataset (158 real driving sequences, 4 radar
sensors + camera + odometry, point-wise semantic labels).

See [TODO.md](TODO.md) for the project plan.

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
scripts/     source (dataloader.py, check_gpu.py)
results/     generated plots (gitignored, recreated by running the scripts)
data/        the RadarScenes dataset (gitignored)
visualize.sh   rad_viewer launcher (kept in root - it's an entry point, not a library)
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
