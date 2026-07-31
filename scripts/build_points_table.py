"""build the {points, attributes, label} dataset, long format.

One row = one radar point belonging to a tracked (dynamic) object. Instance
identity (sequence_name, timestamp, track_id) is repeated on every point's row -
group by those three columns to recover one instance's point set.
"""
import h5py
import pandas as pd

from dataloader import DATA_ROOT, LABELS, OBJECT_ATTRS, RESULTS_DIR

SENSOR_ID = 2  # front-right corner radar (sensors.json: x=3.86, y=-0.7) - single sensor for now


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


if __name__ == "__main__":
    sequence_names = [f"sequence_{i}" for i in range(1, 6)]  # small subset for a first pass
    df = build_points_table(sequence_names)
    n_instances = df.groupby(["sequence_name", "timestamp", "track_id"]).ngroups
    print(f"{len(df)} points across {n_instances} object instances, {len(sequence_names)} sequences")

    RESULTS_DIR.mkdir(exist_ok=True)
    table_path = RESULTS_DIR / "points_table.parquet"
    df.to_parquet(table_path, index=False)
    print(f"Saved points table to {table_path}")
