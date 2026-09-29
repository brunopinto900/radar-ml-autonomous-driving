"""Point-level self-attention over a whole pooled track window (track_accumulation.md,
"Point-level attention over the whole track"). Every fusion variant tried so far glues
together two separately-trained branches after the fact: an order-blind pooled encoder
(DeepReflecs max-pool or a histogram MLP, which never sees more than one scan's points
at a time before pooling) and an order-aware GRU running over per-scan embeddings. This
instead treats every point across all N pooled scans as one token, tagged with its own
per-point recency (0 = target scan, up to N-1 = oldest scan in the window), and lets one
self-attention encoder jointly learn what to attend to and how recent each point is,
rather than splitting those two jobs across two networks trained separately.

Real point-count-per-window statistics (all-sensor N=20 train split, computed before
choosing POINT_CAP, see track_accumulation.md) are heavy-tailed: median 46, p99 269,
p99.9 498, max 692. Attention cost is O(M^2) in points-per-window M, so a fixed cap is
applied (POINT_CAP=300, keeps p99 intact, randomly subsamples the ~0.7% of windows above
it) rather than sizing every batch for the rare crowded-scene tail. Padding is still done
per-batch to that batch's own max (never a fixed dense whole-split array), the same
ragged discipline this branch's repeated OOM bugs forced everywhere else.

Mask convention: follows the scan-level DeepReflecsTransformer's own choice (a per-window
length scalar plus `arange(M) >= lengths`, True = padding, PyTorch's own
src_key_padding_mask convention) rather than DeepReflecs' classifier-style explicit bool
mask (True = real point). Real points are always packed into positions 0..length-1 (point
order within a window carries no meaning, only which scan a point came from does, so
left-aligned packing loses nothing), so a single length scalar is sufficient and avoids
juggling two incompatible mask conventions in one file.
"""
import numpy as np
import pandas as pd
import torch
from torch import nn

from dataloader import RESULTS_DIR
from deepreflecs_classifier import DEVICE, REFLECTION_FEATURES
from deepreflecs_track_accumulation import STRIDE, TRACK_COLS, WINDOW_N
from feature_distributions import MLP_CLASSES
from separability_probe import class_weights
from sequence_split import load_split
from taxonomy_separability import INSTANCE_COLS

POINT_ATTN_DIR = RESULTS_DIR / "track_accumulation_point_attention"

POINT_CAP = 300  # all-sensor N=20 train split: p99=269, p99.9=498, max=692; caps the O(M^2)
# cost of the rare crowded-scene tail instead of sizing every batch for it, truncates
# (via random subsampling) only the top ~0.7% of windows

D_MODEL = 32
NUM_HEADS = 4
NUM_LAYERS = 1
DIM_FEEDFORWARD = 64
EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 4e-5
RANDOM_STATE = 0


def build_windowed_point_sets_with_recency(
    df: pd.DataFrame,
    classes: list[str] = MLP_CLASSES,
    features: list[str] = REFLECTION_FEATURES,
    n: int = WINDOW_N,
    stride: int = STRIDE,
    range_sc_mode: str = "broadcast",
    point_cap: int = POINT_CAP,
    random_state: int = RANDOM_STATE,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Same windowing as deepreflecs_track_accumulation.build_windowed_point_sets (one
    point set per stride-th scan per track, target scan's own class as the label), but
    also returns a parallel per-point recency array (0 = target scan, up to n-1 = oldest
    scan pooled into the window) and caps each window's point count at `point_cap` via
    uniform random subsampling (points are unordered within a scan, so which points get
    dropped doesn't bias any one recency band more than another)."""
    if range_sc_mode not in ("broadcast", "raw"):
        raise ValueError(f"range_sc_mode must be 'broadcast' or 'raw', got {range_sc_mode!r}")

    rng = np.random.RandomState(random_state)
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    mask = df["group"].isin(classes)
    filtered = df.loc[mask]

    feat_matrix = filtered[features].to_numpy(dtype="float32")
    class_idx = filtered["group"].map(class_to_idx).to_numpy()
    range_sc_values = filtered["range_sc"].to_numpy(dtype="float32")
    range_col = features.index("range_sc")

    scan_positions = filtered.groupby(INSTANCE_COLS, sort=False).indices
    scan_arrays = list(scan_positions.values())
    scan_keys = pd.DataFrame(list(scan_positions.keys()), columns=INSTANCE_COLS)
    scan_keys["_scan_idx"] = np.arange(len(scan_keys))
    scan_keys = scan_keys.sort_values(TRACK_COLS + ["timestamp"])

    point_sets, recency_sets, labels = [], [], []
    for _, track_scans in scan_keys.groupby(TRACK_COLS, sort=False):
        ordered = track_scans["_scan_idx"].to_numpy()
        for i in range(0, len(ordered), stride):
            window_idx = ordered[max(0, i - n + 1) : i + 1]
            w = len(window_idx)
            positions = np.concatenate([scan_arrays[j] for j in window_idx])
            feats = feat_matrix[positions].copy()
            recency = np.concatenate(
                [np.full(len(scan_arrays[j]), w - 1 - k, dtype="int64") for k, j in enumerate(window_idx)]
            )

            target_positions = scan_arrays[ordered[i]]
            if range_sc_mode == "broadcast":
                feats[:, range_col] = range_sc_values[target_positions[0]]

            if len(feats) > point_cap:
                keep = rng.choice(len(feats), size=point_cap, replace=False)
                feats, recency = feats[keep], recency[keep]

            point_sets.append(feats)
            recency_sets.append(recency)
            labels.append(class_idx[target_positions[0]])

    return point_sets, recency_sets, np.array(labels, dtype="int64")


def pad_point_batch(
    point_sets: list[np.ndarray], recency_sets: list[np.ndarray], m_max: int, n_features: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Left-aligned per-batch padding (real points at positions 0..length-1), mirrors
    deepreflecs_rnn_track_accumulation.pad_sequences but also pads the parallel recency
    array with the same lengths."""
    n = len(point_sets)
    points = np.zeros((n, m_max, n_features), dtype="float32")
    recency = np.zeros((n, m_max), dtype="int64")
    lengths = np.zeros(n, dtype="int64")
    for i, (p, r) in enumerate(zip(point_sets, recency_sets)):
        points[i, : len(p)] = p
        recency[i, : len(r)] = r
        lengths[i] = len(p)
    return points, recency, lengths


def prepare_point_attention_splits(
    df: pd.DataFrame,
    classes: list[str] = MLP_CLASSES,
    features: list[str] = REFLECTION_FEATURES,
    n: int = WINDOW_N,
    stride: int = STRIDE,
    range_sc_mode: str = "broadcast",
    point_cap: int = POINT_CAP,
    splits: dict[str, list[str]] | None = None,
):
    """Returns ragged (point_sets, recency_sets, labels) per split, standardized on
    train's real points only (same fit-on-train-only discipline as every other split
    prep function in this branch)."""
    if splits is None:
        splits = load_split()

    out = {}
    for split_name in ("train", "val", "test"):
        split_df = df.loc[df["sequence_name"].isin(splits[split_name])]
        point_sets, recency_sets, labels = build_windowed_point_sets_with_recency(
            split_df, classes, features, n, stride, range_sc_mode, point_cap,
        )
        out[split_name] = (point_sets, recency_sets, labels)
        print(f"point-attention {split_name}: {len(labels)} windows, points/window median "
              f"{np.median([len(p) for p in point_sets]):.0f}")

    train_points = np.concatenate(out["train"][0], axis=0)
    mean = train_points.mean(axis=0)
    std = train_points.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    for split_name in ("train", "val", "test"):
        point_sets, recency_sets, labels = out[split_name]
        point_sets = [(p - mean) / std for p in point_sets]
        out[split_name] = (point_sets, recency_sets, labels)

    return out["train"], out["val"], out["test"]


class PointAttentionClassifier(nn.Module):
    """Per-point linear embedding + a learned recency embedding (indexed 0..n-1, the
    point-level analogue of DeepReflecsTransformer's per-scan positional embedding),
    self-attention over the whole pooled window's points, masked max-pool (matching
    DeepReflecs' own aggregation convention) into one classification vector."""

    def __init__(
        self, n_features: int, max_recency: int, d_model: int = D_MODEL, num_heads: int = NUM_HEADS,
        num_layers: int = NUM_LAYERS, dim_feedforward: int = DIM_FEEDFORWARD, num_classes: int = len(MLP_CLASSES),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.point_embed = nn.Linear(n_features, d_model)
        self.recency_embed = nn.Embedding(max_recency, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, points: torch.Tensor, recency: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        B, M, _ = points.shape
        h = self.point_embed(points) + self.recency_embed(recency)

        key_padding_mask = torch.arange(M, device=points.device).unsqueeze(0) >= lengths.unsqueeze(1)
        out = self.encoder(h, src_key_padding_mask=key_padding_mask)

        real_mask = ~key_padding_mask
        out = out.masked_fill(~real_mask.unsqueeze(-1), float("-inf"))
        pooled = out.max(dim=1).values
        return self.classifier(pooled)


def _predict_in_batches(
    model: PointAttentionClassifier, point_sets: list[np.ndarray], recency_sets: list[np.ndarray],
    n_features: int, batch_size: int = 512,
) -> torch.Tensor:
    model.eval()
    logits = []
    with torch.no_grad():
        for start in range(0, len(point_sets), batch_size):
            batch_p = point_sets[start : start + batch_size]
            batch_r = recency_sets[start : start + batch_size]
            batch_m_max = max(len(p) for p in batch_p)
            points, recency, lengths = pad_point_batch(batch_p, batch_r, batch_m_max, n_features)
            points_t = torch.tensor(points, device=DEVICE)
            recency_t = torch.tensor(recency, device=DEVICE)
            lengths_t = torch.tensor(lengths, device=DEVICE)
            logits.append(model(points_t, recency_t, lengths_t))
    return torch.cat(logits, dim=0)


def train_point_attention(
    train_points: list[np.ndarray], train_recency: list[np.ndarray], y_train: np.ndarray,
    val_points: list[np.ndarray], val_recency: list[np.ndarray], y_val: np.ndarray,
    classes: list[str] = MLP_CLASSES, n: int = WINDOW_N, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE, random_state: int = RANDOM_STATE, d_model: int = D_MODEL,
    num_heads: int = NUM_HEADS, num_layers: int = NUM_LAYERS, dim_feedforward: int = DIM_FEEDFORWARD,
):
    torch.manual_seed(random_state)
    n_features = train_points[0].shape[1]

    weights_by_class = class_weights(pd.Series([classes[i] for i in y_train]))
    weight_tensor = torch.tensor([weights_by_class[cls] for cls in classes], dtype=torch.float32, device=DEVICE)
    print(f"class weights: {dict(zip(classes, weight_tensor.tolist()))}")

    model = PointAttentionClassifier(
        n_features, max_recency=n, d_model=d_model, num_heads=num_heads, num_layers=num_layers,
        dim_feedforward=dim_feedforward, num_classes=len(classes),
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    n_train = len(train_points)
    y_train_arr = np.asarray(y_train)
    history = []
    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(n_train)
        epoch_loss, epoch_correct = 0.0, 0
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            batch_p = [train_points[i] for i in idx]
            batch_r = [train_recency[i] for i in idx]
            batch_m_max = max(len(p) for p in batch_p)
            points, recency, lengths = pad_point_batch(batch_p, batch_r, batch_m_max, n_features)
            points_t = torch.tensor(points, device=DEVICE)
            recency_t = torch.tensor(recency, device=DEVICE)
            lengths_t = torch.tensor(lengths, device=DEVICE)
            y_t = torch.tensor(y_train_arr[idx], device=DEVICE)

            optimizer.zero_grad()
            logits = model(points_t, recency_t, lengths_t)
            loss = criterion(logits, y_t)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(idx)
            epoch_correct += (logits.argmax(dim=1) == y_t).sum().item()

        train_loss = epoch_loss / n_train
        train_acc = epoch_correct / n_train

        val_logits = _predict_in_batches(model, val_points, val_recency, n_features)
        val_acc = (val_logits.argmax(dim=1).cpu().numpy() == np.asarray(y_val)).mean()

        history.append({"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc, "val_acc": float(val_acc)})
        print(f"epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}", flush=True)

    return model, history


def evaluate_point_attention_test_metrics(
    model: PointAttentionClassifier, test_points: list[np.ndarray], test_recency: list[np.ndarray],
    y_test: np.ndarray, classes: list[str] = MLP_CLASSES, output_dir=None, n: int = WINDOW_N, stride: int = STRIDE,
):
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_recall_fscore_support

    if output_dir is None:
        output_dir = POINT_ATTN_DIR / f"N{n}_stride{stride}"

    n_features = test_points[0].shape[1]
    y_pred = _predict_in_batches(model, test_points, test_recency, n_features).argmax(dim=1).cpu().numpy()

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=range(len(classes)), zero_division=0
    )
    metrics_df = pd.DataFrame({"precision": precision, "recall": recall, "f1": f1, "support": support}, index=classes)
    print(f"per-class precision/recall/f1 (test, point-attention, N={n}, stride={stride}):")
    print(metrics_df.round(3).to_string())

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_json(output_dir / "point_attention_test_metrics.json", orient="index", indent=2)
    print(f"Saved {output_dir / 'point_attention_test_metrics.json'}")

    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_test, y_pred, labels=range(len(classes)), normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, colorbar=False, values_format=".2f")
    ax.set_title(f"Point-attention: test confusion matrix (N={n})")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(output_dir / "point_attention_test_confusion_matrix.png", dpi=150)
    print(f"Saved {output_dir / 'point_attention_test_confusion_matrix.png'}")

    return metrics_df


def run_point_attention_training(
    df: pd.DataFrame, classes: list[str] = MLP_CLASSES, n: int = WINDOW_N, stride: int = STRIDE,
    range_sc_mode: str = "broadcast", point_cap: int = POINT_CAP, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE, random_state: int = RANDOM_STATE, output_dir=None,
    splits: dict[str, list[str]] | None = None,
):
    if output_dir is None:
        output_dir = POINT_ATTN_DIR / f"N{n}_stride{stride}"

    train_data, val_data, test_data = prepare_point_attention_splits(
        df, classes, REFLECTION_FEATURES, n, stride, range_sc_mode, point_cap, splits,
    )
    train_points, train_recency, y_train = train_data
    val_points, val_recency, y_val = val_data
    test_points, test_recency, y_test = test_data

    model, history = train_point_attention(
        train_points, train_recency, y_train, val_points, val_recency, y_val,
        classes=classes, n=n, epochs=epochs, batch_size=batch_size, lr=lr, random_state=random_state,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "point_attention_model.pt")
    print(f"Saved {output_dir / 'point_attention_model.pt'}")

    from deepreflecs_classifier import plot_training_curves
    plot_training_curves(history, output_dir=output_dir)

    return model, history, test_points, test_recency, y_test
