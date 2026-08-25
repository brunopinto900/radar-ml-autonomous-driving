# radar-ml-autonomous-driving

Per-instance object classifier (MLP, histogram-encoded point cloud, 5 classes) on the
[RadarScenes](https://radar-scenes.com/) dataset (158 sequences, 4 radar sensors + camera +
odometry, point-wise labels). Every design choice is backed by a validated experiment, see
"Where to start reading" below.

## Setup

```bash
python3 -m venv .radar_ml
source .radar_ml/bin/activate
pip install -r requirements.txt
```

- Data expected at `data/RadarScenes/RadarScenes/data/sequence_<N>/` (`radar_data.h5`,
  `scenes.json`, `camera/` per sequence).

## Quick start

```bash
python3 scripts/build_points_table.py   # {points, attributes, label} instance table
python3 scripts/sequence_split.py       # fixed, sequence-grouped train/val/test split
python3 scripts/mlp_classifier.py       # trains (or loads cached) baseline MLP, writes to results/mlp/
```

- Every step is cached, safe to re-run.
- `mlp_classifier.py` always trains/evaluates the standing baseline config. To run a
  different variant (feature set, architecture, taxonomy), use `mlp_variants.py` instead:

```bash
python3 scripts/mlp_variants.py                     # defaults to baseline, same result as mlp_classifier.py
python3 scripts/mlp_variants.py combined_features    # any variant name from MLP_CONFIG.json
python3 scripts/mlp_variants.py deep10
```

- Variant names are the `"variants"` keys in `MLP_CONFIG.json` (e.g. `baseline`,
  `bus_separate`, `range_sc`, `hidden32`, `deep10`, `combined_features`). Each caches its
  training/eval under its own `output_dir`, so running a variant never overwrites the
  baseline's results.

## Where to start reading

1. `Design_Decisions.md`: taxonomy, encoding, and split decisions, with evidence.
2. `notebooks/data_analysis.ipynb`, `notebooks/feature_distributions.ipynb`: the EDA/separability work behind those decisions.
3. `notebooks/mlp_classifier.ipynb`: builds the settled baseline model, reports its metrics.
4. `notebooks/mlp_ablations.ipynb` + `MLP_Decisions_and_Findings.md`: the open-ended ablation program. The `.md` opens with a ranked summary if you just want the headline results.

## Layout

```
scripts/             data loading, table building, plotting, separability probes, MLP classifier
notebooks/           EDA + MLP notebooks
results/             generated plots + cached tables (gitignored)
data/                the RadarScenes dataset (gitignored)
Design_Decisions.md               taxonomy/encoding/split decisions and evidence
MLP_Decisions_and_Findings.md     MLP architecture, hyperparameters, and findings
MLP_CONFIG.json                   every trained MLP variant's config
visualize.sh          rad_viewer launcher
```

## Scripts

- **`dataloader.py`**: loads/plots one scene at a time (h5py, no heavy deps); `inspect_scene()` is the entry point; `CLASS_GROUPS` is the raw-to-training-class taxonomy map. `python3 scripts/dataloader.py`
- **`build_points_table.py`**: builds the `{points, attributes, label}` instance table across all 158 sequences (sensor #2 only). `python3 scripts/build_points_table.py`
- **`class_imbalance.py`**: plots per-class instance counts. `python3 scripts/class_imbalance.py`
- **`taxonomy_separability.py` / `separability_probe.py`**: LR + RF separability probe (sequence-grouped CV) behind the `large_vehicle`/`truck`/`bus` merge decision (`Design_Decisions.md` decision 1). `python3 scripts/taxonomy_separability.py`
- **`sequence_split.py`**: fixed ~70/15/15 sequence-grouped train/val/test split, cached to `results/data/sequence_split.json`. `python3 scripts/sequence_split.py`
- **`feature_distributions.py` / `histogram_separability.py`**: picks the histogram encoding's bin range/count via separability-probe CV, not visual inspection. `python3 scripts/feature_distributions.py`, `python3 scripts/histogram_separability.py`
- **`batch_size_selection.py`**: picks batch size + learning rate from train-split class frequencies (rare-class per-batch miss probability). `python3 scripts/batch_size_selection.py`
- **`mlp_classifier.py`**: the baseline MLP (65→16→16→5), cached training/eval; architecture and findings in `MLP_Decisions_and_Findings.md`. `python3 scripts/mlp_classifier.py`
- **`MLP_CONFIG.json` / `mlp_variants.py` / `class_taxonomy_experiment.py`**: registry of every trained MLP variant (see "Quick start" above for how to run one), and the taxonomy ablation that led to the `bus`/`large_vehicle` merge. `python3 scripts/mlp_variants.py <variant>`
- **`split_sensitivity.py`**: how much macro F1 depends on split choice alone; the noise floor every other ablation is judged against. `python3 scripts/split_sensitivity.py`
- **`pedestrian_separability.py`**: sparse-vs-dense separability probe for the `pedestrian`/`two_wheeler` pair (`MLP_Decisions_and_Findings.md` section 13).
- **`visualize.sh`**: official `rad_viewer` Qt GUI for a full sequence, heavier (PySide6 + pyqtgraph), kept as an occasional inspection tool. `./visualize.sh [sequence_number]` (WSL2 without WSLg needs a Windows-host X server and `libxcb-cursor0`).
