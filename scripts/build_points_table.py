"""build the {points, attributes, label} dataset.

One row = one radar point belonging to a tracked (dynamic) object. Instance
identity (sequence_name, timestamp, track_id) is repeated on every point's row
group by those three columns to recover one instance's point set. Output is a parquet file.
"""
import h5py
import pandas as pd

from dataloader import DATA_ROOT, LABELS, OBJECT_ATTRS, RESULTS_DIR

SENSOR_ID = 2  # front-right corner radar (sensors.json: x=3.86, y=-0.7) - single sensor for now

ALL_SEQUENCES = sorted(
    (p.name for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.startswith("sequence_")),
    key=lambda name: int(name.split("_")[1]),
)

def sequence_points(sequence_name: str, sensor_id: int = SENSOR_ID) -> pd.DataFrame:
    """One row per point, restricted to one sensor's points belonging to a tracked object."""
    with h5py.File(DATA_ROOT / sequence_name / "radar_data.h5", "r") as f:
        radar_data = f["radar_data"][:]

    df = pd.DataFrame({attr: radar_data[attr] for attr in OBJECT_ATTRS})
    df["timestamp"] = radar_data["timestamp"]
    df["track_id"] = radar_data["track_id"]
    df["label_id"] = radar_data["label_id"]
    df = df[(df["track_id"] != b"") & (df["sensor_id"] == sensor_id)]
    df["sequence_name"] = sequence_name
    df["label_name"] = df["label_id"].map(lambda label_id: LABELS[label_id][0])
    return df


def build_points_table(sequence_names: list[str]) -> pd.DataFrame:
    return pd.concat([sequence_points(name) for name in sequence_names], ignore_index=True)


def build_and_save_points_table(sequence_names: list[str] | None = None, table_path=None) -> pd.DataFrame:
    """Build the points table and save it as a parquet file. Returns the table.

    Skips rebuilding if table_path already exists - delete it first to force a rebuild."""
    if sequence_names is None:
        sequence_names = ALL_SEQUENCES
    if table_path is None:
        table_path = RESULTS_DIR / "points_table.parquet"

    if table_path.exists():
        print(f"{table_path} already exists, skipping build")
        return pd.read_parquet(table_path)

    df = build_points_table(sequence_names)
    n_instances = df.groupby(["sequence_name", "timestamp", "track_id"]).ngroups
    print(f"{len(df)} points across {n_instances} object instances, {len(sequence_names)} sequences")

    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_parquet(table_path, index=False)
    print(f"Saved points table to {table_path}")
    return df


if __name__ == "__main__":
    build_and_save_points_table()