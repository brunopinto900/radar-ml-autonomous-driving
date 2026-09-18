"""Build the {points, attributes, label} dataset.

One row = one radar point belonging to a tracked (dynamic) object. Instance identity
(sequence_name, timestamp, track_id) is repeated on every point's row; group by those
three columns to recover one instance's point set. Output is a parquet file."""
import h5py
import pandas as pd

from dataloader import DATA_ROOT, LABELS, OBJECT_ATTRS, RESULTS_DIR

SENSOR_ID = 2  # front-right corner radar (sensors.json: x=3.86, y=-0.7), single sensor for now

ALL_SEQUENCES = sorted(
    (p.name for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.startswith("sequence_")),
    key=lambda name: int(name.split("_")[1]),
)

def sequence_points(sequence_name: str, sensor_id: int | None = SENSOR_ID) -> pd.DataFrame:
    """One row per point belonging to a tracked object. sensor_id=None keeps every
    sensor's points (a track's own timestamps are unique across sensors within a
    sequence, verified directly, so no two sensors' detections of the same track ever
    collide in the (sequence_name, timestamp, track_id) instance key), restricted to
    one sensor's points otherwise."""
    with h5py.File(DATA_ROOT / sequence_name / "radar_data.h5", "r") as f:
        radar_data = f["radar_data"][:]

    df = pd.DataFrame({attr: radar_data[attr] for attr in OBJECT_ATTRS})
    df["timestamp"] = radar_data["timestamp"]
    df["track_id"] = radar_data["track_id"]
    df["label_id"] = radar_data["label_id"]
    mask = df["track_id"] != b""
    if sensor_id is not None:
        mask &= df["sensor_id"] == sensor_id
    df = df[mask]
    df["sequence_name"] = sequence_name
    df["label_name"] = df["label_id"].map(lambda label_id: LABELS[label_id][0])
    return df


def build_points_table(sequence_names: list[str], sensor_id: int | None = SENSOR_ID) -> pd.DataFrame:
    return pd.concat([sequence_points(name, sensor_id) for name in sequence_names], ignore_index=True)


def build_and_save_points_table(
    sequence_names: list[str] | None = None, table_path=None, sensor_id: int | None = SENSOR_ID
) -> pd.DataFrame:
    """Build the points table and save it as a parquet file. Returns the table.

    Skips rebuilding if table_path already exists AND covers exactly the requested
    sequence_names, otherwise rebuilds (e.g. a cached full-158-sequence table won't be
    silently returned for a request for just a couple of sequences)."""
    if sequence_names is None:
        sequence_names = ALL_SEQUENCES
    if table_path is None:
        table_path = RESULTS_DIR / "data" / "points_table.parquet"

    if table_path.exists():
        cached_sequences = set(pd.read_parquet(table_path, columns=["sequence_name"])["sequence_name"])
        if cached_sequences == set(sequence_names):
            print(f"{table_path} already covers exactly the requested sequences, skipping build")
            return pd.read_parquet(table_path)
        print(f"{table_path} exists but covers different sequences than requested, rebuilding")

    df = build_points_table(sequence_names, sensor_id)
    n_instances = df.groupby(["sequence_name", "timestamp", "track_id"]).ngroups
    print(f"{len(df)} points across {n_instances} object instances, {len(sequence_names)} sequences")

    table_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(table_path, index=False)
    print(f"Saved points table to {table_path}")
    return df


if __name__ == "__main__":
    build_and_save_points_table()