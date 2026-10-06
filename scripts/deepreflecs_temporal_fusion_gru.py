"""Parallel scan-level temporal encoder, fused with the frozen DeepReflecs per-scan
point embedding before the GRU (not glued on post-hoc at the head like
DeepReflecsGRUWithTemporal in deepreflecs_rnn_track_accumulation.py).

Motivation (track_accumulation.md, patent-inspired tracker-feature investigation): a
single-feature AUC check on window-level temporal aggregates (heading variance,
curvature, RCS/velocity variance, RCS/velocity signed-diff variance and mean) found no
one feature separating either confusion (two_wheeler->pedestrian, large_vehicle->car)
above ~0.78. A joint check (shallow gradient-boosted trees on all 8 features together,
5-fold CV) found much stronger joint separability (0.78-0.92 AUC), with velocity
variance/oscillation, curvature, and RCS variance carrying almost all the importance
and the drift features carrying none. That's the case for trying an architecture that
can combine these cues *and* learn end-to-end (rather than relying on exactly the 8
hand-engineered window-level aggregates the GBT saw), which is this module.

Design: per scan t, a small raw feature vector (that scan's own median RCS/vr_compensated,
their signed diff from scan t-1, and the scan's centroid displacement dx/dy from scan
t-1) feeds a trainable 2-layer MLP ("scan encoder", -> 32 dims). This is concatenated,
every timestep, with the frozen per-scan DeepReflecs point-encoder embedding (also 32
dims, precomputed, see deepreflecs_rnn_track_accumulation.compute_scan_embeddings). The
fused per-step vector feeds a GRU (freshly initialized, not the existing frozen one:
its input dimensionality has changed and it was never trained to use the temporal
channel). Scan encoder + GRU + classification head all train end-to-end; only the
point encoder stays frozen.

The raw temporal features are intentionally minimal (6 scalars, not the 8 window-level
aggregates from the AUC study): variance/curvature/oscillation are exactly the kind of
running statistic a GRU is built to accumulate from a raw per-step diff sequence itself,
so handing it the window-level aggregates directly would partly defeat the point of
testing whether an end-to-end sequence model can learn them."""
import copy
import json

import numpy as np
import pandas as pd
import torch
from torch import nn

from build_points_table import build_and_save_points_table
from deepreflecs_classifier import DEVICE, plot_training_curves
from deepreflecs_rnn_track_accumulation import (
    RNN_DIR, TRACK_COLS, WINDOW_N, STRIDE, HIDDEN_SIZE, GRU_LAYERS, MLP_HIDDEN_DIM,
    LEARNING_RATE, EPOCHS, BATCH_SIZE, RANDOM_STATE,
    get_or_compute_scan_embeddings, pad_sequences, _predict_in_batches,
)
from deepreflecs_track_accumulation import (
    POINTS_TABLE_SEQ_PATH, POINTS_TABLE_SEQ_ALLSENSORS_PATH, add_relative_features_seq,
)
from feature_distributions import MLP_CLASSES
from mlp_classifier import MLP, apply_mlp_class_groups
from separability_probe import class_weights
from sequence_split import load_split
from taxonomy_separability import INSTANCE_COLS

TEMPORAL_FUSION_DIR = RNN_DIR / "temporal_fusion_gru"
SCAN_ENCODER_DIM = 32
TEMPORAL_RAW_COLS = ["rcs", "vr_compensated", "d_rcs", "d_vr", "dx", "dy"]


def compute_scan_temporal_raw(df: pd.DataFrame, classes: list[str] = MLP_CLASSES) -> pd.DataFrame:
    """One row per scan (INSTANCE_COLS), six raw causal scalars: that scan's own
    median rcs/vr_compensated, their signed diff from the previous scan in the same
    track (0 for a track's first scan, no prior scan to diff against), and the
    centroid (mean x_cc, y_cc) displacement dx/dy from the previous scan (0 for the
    first scan). Computed once over the whole df, same convention as
    compute_scan_embeddings (one pass, cached by the caller, not recomputed per
    window)."""
    mask = df["group"].isin(classes)
    filtered = df.loc[mask]
    scan_stats = filtered.groupby(INSTANCE_COLS, sort=False).agg(
        rcs=("rcs", "median"), vr_compensated=("vr_compensated", "median"),
        x_cc=("x_cc", "mean"), y_cc=("y_cc", "mean"),
    ).reset_index()
    scan_stats = scan_stats.sort_values(TRACK_COLS + ["timestamp"])

    grouped = scan_stats.groupby(TRACK_COLS, sort=False)
    scan_stats["d_rcs"] = grouped["rcs"].diff().fillna(0.0)
    scan_stats["d_vr"] = grouped["vr_compensated"].diff().fillna(0.0)
    scan_stats["dx"] = grouped["x_cc"].diff().fillna(0.0)
    scan_stats["dy"] = grouped["y_cc"].diff().fillna(0.0)
    return scan_stats[INSTANCE_COLS + TEMPORAL_RAW_COLS]


def fit_temporal_standardization(temporal_df: pd.DataFrame, splits: dict[str, list[str]]) -> tuple[np.ndarray, np.ndarray]:
    train_rows = temporal_df.loc[temporal_df["sequence_name"].isin(splits["train"]), TEMPORAL_RAW_COLS]
    mean = train_rows.mean().to_numpy(dtype="float32")
    std = train_rows.std().to_numpy(dtype="float32")
    return mean, np.where(std > 0, std, 1.0).astype("float32")


def build_windowed_fused_sequences(
    combined_df: pd.DataFrame, point_cols: list[str], temporal_cols: list[str],
    classes: list[str] = MLP_CLASSES, n: int = WINDOW_N, stride: int = STRIDE,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Same windowing as build_windowed_embedding_sequences, but each step's vector
    is [point_embedding (point_cols), raw_temporal (temporal_cols)] concatenated;
    the model's forward splits them back apart (point_dim known at model construction)."""
    ordered = combined_df.sort_values(TRACK_COLS + ["timestamp"]).reset_index(drop=True)
    feat_matrix = ordered[point_cols + temporal_cols].to_numpy(dtype="float32")
    label_arr = ordered["label"].to_numpy()

    sequences, labels = [], []
    for _, positions in ordered.groupby(TRACK_COLS, sort=False).indices.items():
        for i in range(0, len(positions), stride):
            window_positions = positions[max(0, i - n + 1) : i + 1]
            sequences.append(feat_matrix[window_positions])
            labels.append(label_arr[window_positions[-1]])
    return sequences, np.array(labels, dtype="int64")


def prepare_temporal_fusion_splits(
    df: pd.DataFrame, embeddings_df: pd.DataFrame, classes: list[str] = MLP_CLASSES,
    n: int = WINDOW_N, stride: int = STRIDE, splits: dict[str, list[str]] | None = None,
):
    if splits is None:
        splits = load_split()

    point_cols = [c for c in embeddings_df.columns if c.startswith("e")]
    point_dim = len(point_cols)

    temporal_raw = compute_scan_temporal_raw(df, classes)
    mean, std = fit_temporal_standardization(temporal_raw, splits)
    temporal_raw[TEMPORAL_RAW_COLS] = (temporal_raw[TEMPORAL_RAW_COLS].to_numpy(dtype="float32") - mean) / std

    combined = embeddings_df.merge(temporal_raw, on=INSTANCE_COLS, how="inner")
    assert len(combined) == len(embeddings_df), "temporal raw features didn't cover every scan in embeddings_df"

    train_df = combined.loc[combined["sequence_name"].isin(splits["train"])]
    val_df = combined.loc[combined["sequence_name"].isin(splits["val"])]
    test_df = combined.loc[combined["sequence_name"].isin(splits["test"])]

    train_seqs, y_train = build_windowed_fused_sequences(train_df, point_cols, TEMPORAL_RAW_COLS, classes, n, stride)
    val_seqs, y_val = build_windowed_fused_sequences(val_df, point_cols, TEMPORAL_RAW_COLS, classes, n, stride)
    test_seqs, y_test = build_windowed_fused_sequences(test_df, point_cols, TEMPORAL_RAW_COLS, classes, n, stride)

    print(f"N={n} stride={stride}: windows train={len(train_seqs)} val={len(val_seqs)} test={len(test_seqs)}, "
          f"point_dim={point_dim}, temporal_dim={len(TEMPORAL_RAW_COLS)}")
    return train_seqs, y_train, val_seqs, y_val, test_seqs, y_test, point_dim


class TemporalFusionGRU(nn.Module):
    """Frozen per-scan point embedding (point_dim, passed through unchanged) fused,
    every timestep, with a trainable scan encoder's output on the raw temporal
    features (temporal_dim -> scan_encoder_dim), before a freshly-initialized GRU.
    Everything here (scan encoder, GRU, head) trains end-to-end; the point embedding
    itself was already frozen upstream when it was precomputed."""

    def __init__(
        self, point_dim: int, temporal_dim: int, scan_encoder_dim: int = SCAN_ENCODER_DIM,
        hidden_size: int = HIDDEN_SIZE, num_layers: int = GRU_LAYERS,
        num_classes: int = len(MLP_CLASSES), mlp_hidden_dim: int = MLP_HIDDEN_DIM,
    ):
        super().__init__()
        self.point_dim = point_dim
        self.scan_encoder = nn.Sequential(
            nn.Linear(temporal_dim, scan_encoder_dim), nn.ReLU(),
            nn.Linear(scan_encoder_dim, scan_encoder_dim), nn.ReLU(),
        )
        self.gru = nn.GRU(point_dim + scan_encoder_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = MLP(input_dim=hidden_size, hidden_dim=mlp_hidden_dim, num_classes=num_classes, n_hidden_layers=1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        point_part = x[..., : self.point_dim]
        temporal_part = x[..., self.point_dim :]
        scan_emb = self.scan_encoder(temporal_part)
        fused = torch.cat([point_part, scan_emb], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(fused, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        return self.head(h_n[-1])


def train_temporal_fusion_gru(
    train_seqs: list[np.ndarray], y_train: np.ndarray, val_seqs: list[np.ndarray], y_val: np.ndarray,
    point_dim: int, temporal_dim: int, classes: list[str] = MLP_CLASSES, epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE, lr: float = LEARNING_RATE, random_state: int = RANDOM_STATE,
    hidden_size: int = HIDDEN_SIZE, num_layers: int = GRU_LAYERS, mlp_hidden_dim: int = MLP_HIDDEN_DIM,
):
    """Same Adam + class-weighted CE convention as train_gru, model class and the
    point/temporal dims differ."""
    torch.manual_seed(random_state)

    weights_by_class = class_weights(pd.Series([classes[i] for i in y_train]))
    weight_tensor = torch.tensor([weights_by_class[cls] for cls in classes], dtype=torch.float32, device=DEVICE)
    print(f"class weights: {dict(zip(classes, weight_tensor.tolist()))}")

    model = TemporalFusionGRU(
        point_dim, temporal_dim, hidden_size=hidden_size, num_layers=num_layers,
        num_classes=len(classes), mlp_hidden_dim=mlp_hidden_dim,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    embedding_dim = point_dim + temporal_dim
    y_train_t = torch.tensor(y_train, device=DEVICE)
    y_val_t = torch.tensor(y_val, device=DEVICE)

    n = len(train_seqs)
    history = []
    best_val_acc = -1.0
    best_state = None
    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(n)
        epoch_loss, epoch_correct = 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            batch_seqs = [train_seqs[i] for i in idx]
            batch_m_max = max(len(s) for s in batch_seqs)
            batch_x, batch_lengths = pad_sequences(batch_seqs, batch_m_max, embedding_dim)
            batch_x_t = torch.tensor(batch_x, device=DEVICE)
            batch_lengths_t = torch.tensor(batch_lengths)
            batch_y = y_train_t[idx]

            optimizer.zero_grad()
            logits = model(batch_x_t, batch_lengths_t)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(idx)
            epoch_correct += (logits.argmax(dim=1) == batch_y).sum().item()

        train_loss = epoch_loss / n
        train_acc = epoch_correct / n

        val_logits = _predict_in_batches(model, val_seqs)
        val_acc = (val_logits.argmax(dim=1) == y_val_t).float().mean().item()

        history.append({"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc, "val_acc": val_acc})
        print(f"epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1

    print(f"restoring best checkpoint: epoch {best_epoch}, val_acc={best_val_acc:.4f} (final epoch was {history[-1]['val_acc']:.4f})")
    model.load_state_dict(best_state)
    return model, history


def evaluate_temporal_fusion_test_metrics(
    model: TemporalFusionGRU, test_seqs: list[np.ndarray], y_test: np.ndarray,
    classes: list[str] = MLP_CLASSES, output_dir=TEMPORAL_FUSION_DIR, title_suffix: str = "",
):
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_recall_fscore_support
    import matplotlib.pyplot as plt

    y_pred = _predict_in_batches(model, test_seqs).argmax(dim=1).cpu().numpy()
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=range(len(classes)), zero_division=0
    )
    metrics_df = pd.DataFrame({"precision": precision, "recall": recall, "f1": f1, "support": support}, index=classes)
    print(f"per-class precision/recall/f1 (test{title_suffix}):")
    print(metrics_df.round(3).to_string())
    print(f"macro F1: {f1.mean():.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_json(output_dir / "temporal_fusion_test_metrics.json", orient="index", indent=2)

    cm = confusion_matrix(y_test, y_pred, labels=range(len(classes)), normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, colorbar=False, values_format=".2f")
    ax.set_title(f"Temporal-fusion GRU: test confusion matrix{title_suffix}")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(output_dir / "temporal_fusion_test_confusion_matrix.png", dpi=150)
    print(f"Saved {output_dir / 'temporal_fusion_test_confusion_matrix.png'}")
    return metrics_df, y_pred


def run_temporal_fusion_training(
    df: pd.DataFrame, embeddings_df: pd.DataFrame, n: int = WINDOW_N, stride: int = STRIDE,
    classes: list[str] = MLP_CLASSES, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE, random_state: int = RANDOM_STATE, hidden_size: int = HIDDEN_SIZE,
    num_layers: int = GRU_LAYERS, mlp_hidden_dim: int = MLP_HIDDEN_DIM, output_dir=None,
    splits: dict[str, list[str]] | None = None,
):
    if output_dir is None:
        output_dir = TEMPORAL_FUSION_DIR
    if splits is None:
        splits = load_split()

    train_seqs, y_train, val_seqs, y_val, test_seqs, y_test, point_dim = prepare_temporal_fusion_splits(
        df, embeddings_df, classes, n, stride, splits,
    )
    temporal_dim = len(TEMPORAL_RAW_COLS)

    model, history = train_temporal_fusion_gru(
        train_seqs, y_train, val_seqs, y_val, point_dim, temporal_dim, classes, epochs,
        batch_size, lr, random_state, hidden_size, num_layers, mlp_hidden_dim,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "temporal_fusion_model.pt")
    history_cache = output_dir / "temporal_fusion_training_history.json"
    history_cache.write_text(json.dumps({"point_dim": point_dim, "temporal_dim": temporal_dim, "history": history}, indent=2))
    print(f"Saved {output_dir / 'temporal_fusion_model.pt'} and {history_cache}")
    plot_training_curves(history, output_dir=output_dir)
    return model, history, test_seqs, y_test


if __name__ == "__main__":
    N = 20
    output_dir = TEMPORAL_FUSION_DIR / f"N{N}_stride1_allsensors"

    sensor2_df = build_and_save_points_table(table_path=POINTS_TABLE_SEQ_PATH)
    sensor2_df = add_relative_features_seq(sensor2_df)
    sensor2_df = apply_mlp_class_groups(sensor2_df)

    df = build_and_save_points_table(table_path=POINTS_TABLE_SEQ_ALLSENSORS_PATH, sensor_id=None)
    df = add_relative_features_seq(df)
    df = apply_mlp_class_groups(df)

    embeddings_cache = RNN_DIR / "scan_embeddings_allsensors.parquet"
    embeddings_df = get_or_compute_scan_embeddings(df, cache_path=embeddings_cache, standardization_df=sensor2_df)

    model, history, test_seqs, y_test = run_temporal_fusion_training(df, embeddings_df, n=N, output_dir=output_dir)
    evaluate_temporal_fusion_test_metrics(model, test_seqs, y_test, output_dir=output_dir, title_suffix=f" (N={N}, all sensors)")
