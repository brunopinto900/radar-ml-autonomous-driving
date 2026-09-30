"""Reproduces the multi-scan track-accumulation pipeline end to end: pooled DeepReflecs
baseline -> GRU over frozen per-scan embeddings -> fusion of the two. Three stages,
run in order, each as its own subprocess so memory is released between them (the
windowed point-set arrays at this scale are large enough that keeping all three
stages' data resident in one process OOMs on a 12GB machine).

Only the GRU stage skips retraining when its exact config is already cached; the
pooled-baseline and fusion stages always retrain from scratch (the underlying
functions rebuild their point-set/split arrays and never check a cache before
training). Safe to re-run in the sense that it won't crash and won't corrupt other
configs' results, but re-running it overwrites the pooled-baseline and fusion
results with a fresh, not bit-identical (GPU training isn't deterministic), training
run each time.

Usage:
    python3 scripts/run_multiscan_pipeline.py [N] [sensor_scope]

    N            window size, number of scans pooled/accumulated per decision (default 20)
    sensor_scope "all" (all 4 sensors, cross-sensor handoff) or "sensor2" (single sensor,
                 default "all")

Examples:
    python3 scripts/run_multiscan_pipeline.py            # N=20, all sensors (headline result)
    python3 scripts/run_multiscan_pipeline.py 10 sensor2  # N=10, sensor 2 only
"""
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent

STAGE_POOLED = """
from build_points_table import build_and_save_points_table
from mlp_classifier import apply_mlp_class_groups
from deepreflecs_track_accumulation import (
    POINTS_TABLE_SEQ_PATH, POINTS_TABLE_SEQ_ALLSENSORS_PATH, TRACK_ACC_DIR,
    add_relative_features_seq, run_windowed_training_ragged, evaluate_windowed_test_metrics_ragged,
)

if "{scope}" == "all":
    df = build_and_save_points_table(table_path=POINTS_TABLE_SEQ_ALLSENSORS_PATH, sensor_id=None)
else:
    df = build_and_save_points_table(table_path=POINTS_TABLE_SEQ_PATH)
df = add_relative_features_seq(df)
df = apply_mlp_class_groups(df)

# ragged path: at N={n} the dense whole-split padded array doesn't fit in memory (see
# prepare_windowed_split_point_sets_ragged's docstring), so pooled baseline always uses
# the ragged train/eval functions, never the plain (dense) ones.
pooled_dir = TRACK_ACC_DIR / "N{n}_stride1_{tag}"
model, _, test_sets, y_test, mean, std = run_windowed_training_ragged(df, n={n}, output_dir=pooled_dir)
pooled_metrics = evaluate_windowed_test_metrics_ragged(model, test_sets, y_test, mean, std, n={n}, output_dir=pooled_dir)
print(f"pooled baseline macro F1: {{pooled_metrics['f1'].mean():.4f}}")
"""

STAGE_GRU = """
from build_points_table import build_and_save_points_table
from mlp_classifier import apply_mlp_class_groups
from deepreflecs_track_accumulation import (
    POINTS_TABLE_SEQ_PATH, POINTS_TABLE_SEQ_ALLSENSORS_PATH, add_relative_features_seq,
)
from deepreflecs_rnn_track_accumulation import (
    RNN_DIR, get_or_compute_scan_embeddings, run_gru_training, evaluate_gru_test_metrics,
)

# sensor2 is always needed too: the frozen per-scan encoder feeding the GRU was
# trained on sensor2, so its standardization stats are reused whenever df is all-sensor
sensor2_df = build_and_save_points_table(table_path=POINTS_TABLE_SEQ_PATH)
sensor2_df = add_relative_features_seq(sensor2_df)
sensor2_df = apply_mlp_class_groups(sensor2_df)

if "{scope}" == "all":
    df = build_and_save_points_table(table_path=POINTS_TABLE_SEQ_ALLSENSORS_PATH, sensor_id=None)
    df = add_relative_features_seq(df)
    df = apply_mlp_class_groups(df)
else:
    df = sensor2_df

embeddings_cache = RNN_DIR / "scan_embeddings_{tag}.parquet"
embeddings_df = get_or_compute_scan_embeddings(df, cache_path=embeddings_cache, standardization_df=sensor2_df)
gru_dir = RNN_DIR / "N{n}_stride1_gru_h64_{tag}"
gru_model, _, gru_test_seqs, gru_y_test = run_gru_training(embeddings_df, n={n}, output_dir=gru_dir)
gru_metrics, _, _ = evaluate_gru_test_metrics(gru_model, gru_test_seqs, gru_y_test, output_dir=gru_dir)
print(f"GRU macro F1: {{gru_metrics['f1'].mean():.4f}}")
"""

STAGE_FUSION = """
from build_points_table import build_and_save_points_table
from mlp_classifier import apply_mlp_class_groups
from deepreflecs_track_accumulation import (
    POINTS_TABLE_SEQ_PATH, POINTS_TABLE_SEQ_ALLSENSORS_PATH, TRACK_ACC_DIR, add_relative_features_seq,
)
from deepreflecs_rnn_track_accumulation import (
    RNN_DIR, get_or_compute_scan_embeddings, run_fusion_training, evaluate_fusion_test_metrics,
)

if "{scope}" == "all":
    df = build_and_save_points_table(table_path=POINTS_TABLE_SEQ_ALLSENSORS_PATH, sensor_id=None)
else:
    df = build_and_save_points_table(table_path=POINTS_TABLE_SEQ_PATH)
df = add_relative_features_seq(df)
df = apply_mlp_class_groups(df)

embeddings_cache = RNN_DIR / "scan_embeddings_{tag}.parquet"
embeddings_df = get_or_compute_scan_embeddings(df, cache_path=embeddings_cache)

pooled_dir = TRACK_ACC_DIR / "N{n}_stride1_{tag}"
gru_dir = RNN_DIR / "N{n}_stride1_gru_h64_{tag}"
fusion_dir = RNN_DIR / "N{n}_stride1_fusion_pooled_gru64_{tag}"
fusion_model, _, X_test, y_test = run_fusion_training(
    df, embeddings_df, n={n}, pooled_encoder_dir=pooled_dir, gru_model_dir=gru_dir, output_dir=fusion_dir,
)
fusion_metrics = evaluate_fusion_test_metrics(fusion_model, X_test, y_test, n={n}, output_dir=fusion_dir)
print(f"fusion macro F1: {{fusion_metrics['f1'].mean():.4f}}")
"""


def run_stage(name: str, template: str, n: int, scope: str, tag: str):
    print(f"--- stage: {name} ---")
    code = template.format(n=n, scope=scope, tag=tag)
    subprocess.run([sys.executable, "-u", "-c", code], check=True, cwd=SCRIPTS_DIR)


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    SENSOR_SCOPE = sys.argv[2] if len(sys.argv) > 2 else "all"
    if SENSOR_SCOPE not in ("all", "sensor2"):
        raise ValueError(f"sensor_scope must be 'all' or 'sensor2', got {SENSOR_SCOPE!r}")

    tag = "allsensors" if SENSOR_SCOPE == "all" else "sensor2"
    print(f"Pipeline: N={N}, sensor_scope={SENSOR_SCOPE}")

    run_stage("pooled baseline", STAGE_POOLED, N, SENSOR_SCOPE, tag)
    run_stage("GRU", STAGE_GRU, N, SENSOR_SCOPE, tag)
    run_stage("fusion", STAGE_FUSION, N, SENSOR_SCOPE, tag)
