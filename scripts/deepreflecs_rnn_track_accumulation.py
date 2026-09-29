"""track-accumulation branch (track_accumulation.md): DeepReflecs per-scan point encoder
+ GRU, instead of pooling raw points across scans into one bigger point cloud (every
other variant in this branch). Each scan keeps its own point set; a frozen, already-
trained DeepReflecs encoder (the N=1 control from deepreflecs_track_accumulation.py,
x_seq/y_seq frame) reduces it to one POINT_DIM-wide embedding, and a GRU consumes the
sequence of a track's embeddings to carry information across scans instead of a hand-
engineered scalar (temporal variation, transition matrix, ...).

Precompute-then-train-RNN, not end-to-end: the encoder runs exactly once per scan
across the whole dataset (compute_scan_embeddings), cached to disk. Windowing at the
GRU-training stage (build_windowed_embedding_sequences) then just slices these cached
vectors, so a stride=1 sliding window's N-1 scan overlap costs nothing extra in encoder
compute, unlike naive end-to-end windowing which would re-run the point encoder on the
same scan's points up to N times. Trade-off accepted: the embeddings are whatever the
N=1 encoder already learned for single-scan classification, not fine-tuned for
temporal usefulness.

Unlike the fixed-length histogram/pooled encoders elsewhere in this branch, the GRU
needs no padding convention for short windows: it just runs fewer steps for a track
younger than N scans (`torch.nn.utils.rnn.pack_padded_sequence`), sidestepping the
padding-looks-like-real-stability bug that hurt the signed diff vector attempt.

Unidirectional/causal by construction, deliberately not mirroring the bidirectional
LSTM in Hassan et al. 2024 (EuRAD, "Classification of Tracked Objects Using Multiple
Frame Processing for Automotive Radar"): a verdict at scan t must only depend on scans
up to and including t, so it stays usable in real-time streaming inference, matching
every other variant in this branch."""
import functools
import json

import numpy as np
import pandas as pd
import torch
from torch import nn

from dataloader import RESULTS_DIR
from deepreflecs_classifier import (
    CONV_DIM,
    DEVICE,
    POINT_DIM,
    REFLECTION_FEATURES,
    DeepReflecs,
    build_point_sets,
    pad_to_fixed,
    plot_training_curves,
)
from deepreflecs_track_accumulation import POINTS_TABLE_SEQ_PATH, TRACK_ACC_DIR, TRACK_COLS, add_relative_features_seq
from feature_distributions import MLP_CLASSES
from mlp_classifier import MLP, apply_mlp_class_groups
from separability_probe import class_weights
from sequence_split import load_split
from taxonomy_separability import INSTANCE_COLS

# frozen encoder: the N=1 control DeepReflecs model already trained/saved by
# deepreflecs_track_accumulation.py (x_seq/y_seq, per-scan recentered, canonical split)
ENCODER_DIR = TRACK_ACC_DIR / "N1_stride1"

RNN_DIR = RESULTS_DIR / "track_accumulation_rnn"
EMBEDDINGS_CACHE = RNN_DIR / "scan_embeddings_sensor2.parquet"

# --- hyperparameters ---
WINDOW_N = 10
STRIDE = 1
HIDDEN_SIZE = 64
GRU_LAYERS = 1
MLP_HIDDEN_DIM = 16
LEARNING_RATE = 4e-5  # matches deepreflecs_classifier.py / mlp_classifier.py
EPOCHS = 100
BATCH_SIZE = 128
RANDOM_STATE = 0
# ------------------------


def embed_points(model: DeepReflecs, points: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """DeepReflecs.forward stopped right before the final classification dense layer:
    shared per-point layers, global context layer, masked global max pool. This
    POINT_DIM-wide vector is the frozen per-scan feature the GRU consumes."""
    x = torch.relu(model.point_conv1(points))
    x = model.context(x, mask)
    x = torch.relu(model.point_conv2(x))
    x = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
    return x.max(dim=1).values


def fit_reflection_standardization(
    df: pd.DataFrame, splits: dict[str, list[str]], classes: list[str] = MLP_CLASSES,
    features: list[str] = REFLECTION_FEATURES,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduces exactly the per-feature mean/std the N=1 control encoder was trained
    with (deepreflecs_classifier.prepare_split_point_sets: fit on train's real, non-
    padding points only). Must match training-time normalization exactly, not be
    refit here, since the frozen encoder's weights only make sense on inputs scaled
    the way it was trained on."""
    train_df = df.loc[df["sequence_name"].isin(splits["train"])]
    train_sets, _ = build_point_sets(train_df, classes, features)
    m_max = max(p.shape[0] for p in train_sets)
    X_train, mask_train = pad_to_fixed(train_sets, m_max)
    mean = X_train[mask_train].mean(axis=0)
    std = X_train[mask_train].std(axis=0)
    return mean, np.where(std > 0, std, 1.0)


def build_scan_point_sets_with_keys(
    df: pd.DataFrame, classes: list[str] = MLP_CLASSES, features: list[str] = REFLECTION_FEATURES,
) -> tuple[list[np.ndarray], np.ndarray, pd.DataFrame]:
    """Same per-scan grouping as deepreflecs_classifier.build_point_sets, but also
    returns each scan's own identity (sequence_name, timestamp, track_id) in matching
    order, needed to reassemble per-track embedding sequences afterward."""
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    mask = df["group"].isin(classes)
    filtered = df.loc[mask]
    feat_matrix = filtered[features].to_numpy(dtype="float32")
    class_idx = filtered["group"].map(class_to_idx).to_numpy()

    scan_positions = filtered.groupby(INSTANCE_COLS, sort=False).indices
    point_sets, labels, keys = [], [], []
    for key, positions in scan_positions.items():
        point_sets.append(feat_matrix[positions])
        labels.append(class_idx[positions[0]])
        keys.append(key)
    keys_df = pd.DataFrame(keys, columns=INSTANCE_COLS)
    return point_sets, np.array(labels, dtype="int64"), keys_df


def compute_scan_embeddings(
    df: pd.DataFrame,
    classes: list[str] = MLP_CLASSES,
    features: list[str] = REFLECTION_FEATURES,
    encoder_dir=ENCODER_DIR,
    conv_dim: int = CONV_DIM,
    point_dim: int = POINT_DIM,
    splits: dict[str, list[str]] | None = None,
    batch_size: int = 1024,
    standardization_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Runs the frozen N=1 control DeepReflecs encoder once per scan across all of df,
    not once per window: a scan sitting inside up to WINDOW_N overlapping stride=1
    windows still only gets encoded here exactly once, the entire point of precompute-
    then-train-RNN over end-to-end. Returns one row per scan: INSTANCE_COLS + `label`
    (class index into `classes`) + one embedding column per encoder output dim
    (`e0..e{point_dim-1}`).

    standardization_df: the encoder's normalization must match its training-time
    stats exactly (see fit_reflection_standardization), which were fit on sensor2
    data. Embedding an all-sensor df with stats refit from that same all-sensor df
    would silently feed the frozen encoder inputs scaled differently than it was
    trained on. Pass the original sensor2 df here to reproduce the correct stats
    when df is a different sensor scope; defaults to df itself (existing sensor2
    behavior, unchanged)."""
    if splits is None:
        splits = load_split()
    if standardization_df is None:
        standardization_df = df

    model_path = encoder_dir / "deepreflecs_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} doesn't exist, train the N=1 control encoder first")

    mean, std = fit_reflection_standardization(standardization_df, splits, classes, features)

    point_sets, labels, keys_df = build_scan_point_sets_with_keys(df, classes, features)
    point_sets_std = [(p - mean) / std for p in point_sets]

    model = DeepReflecs(len(features), num_classes=len(classes), conv_dim=conv_dim, point_dim=point_dim).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    embeddings = []
    with torch.no_grad():
        for start in range(0, len(point_sets_std), batch_size):
            batch_sets = point_sets_std[start : start + batch_size]
            batch_m_max = max(p.shape[0] for p in batch_sets)
            batch_x, batch_mask = pad_to_fixed(batch_sets, batch_m_max)
            batch_x_t = torch.tensor(batch_x, device=DEVICE)
            batch_mask_t = torch.tensor(batch_mask, device=DEVICE)
            embeddings.append(embed_points(model, batch_x_t, batch_mask_t))
    embeddings = torch.cat(embeddings, dim=0).cpu().numpy()

    result = keys_df.copy()
    result["label"] = labels
    for i in range(embeddings.shape[1]):
        result[f"e{i}"] = embeddings[:, i]
    return result


def get_or_compute_scan_embeddings(df: pd.DataFrame, cache_path=EMBEDDINGS_CACHE, **kwargs) -> pd.DataFrame:
    """Embedding the whole dataset once takes real time; retraining/tweaking the GRU
    afterward should never have to redo it, hence the cache."""
    if cache_path.exists():
        print(f"{cache_path} already exists, loading cached embeddings")
        return pd.read_parquet(cache_path)
    result = compute_scan_embeddings(df, **kwargs)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(cache_path)
    print(f"Saved {cache_path}")
    return result


def compute_scan_embeddings_mlp(
    df: pd.DataFrame,
    classes: list[str] = MLP_CLASSES,
    n_bins: int = 16,
    encoder_dir=None,
    mlp_hidden_dim: int = 16,
    splits: dict[str, list[str]] | None = None,
    batch_size: int = 1024,
) -> pd.DataFrame:
    """MLP-line counterpart to compute_scan_embeddings: instead of the frozen
    DeepReflecs point-set encoder, runs a frozen per-scan (N=1) histogram-MLP encoder
    (mlp_track_accumulation.run_windowed_mlp(n=1, range_sc_mode="raw"), trained
    separately) and takes its penultimate hidden-layer activation (MLP.forward_features)
    as the per-scan embedding. Same output convention as compute_scan_embeddings
    (INSTANCE_COLS + label + e0..e{mlp_hidden_dim-1}), so every downstream GRU/fusion
    function that consumes an embeddings_df works unchanged regardless of which encoder
    produced it, feeding the histogram-MLP's own per-scan representation into the GRU's
    recurrence at every timestep instead of only gluing it on post-hoc via fusion.

    Edges are refit here from build_scan_point_sets_with_keys(train_df, ...): at n=1,
    range_sc_mode="raw", build_windowed_point_sets never fires its broadcast override
    (there's no other scan in the window to broadcast from), so this per-scan point set
    is identical to what run_windowed_mlp(n=1) built at training time, and refitting
    edges here reproduces its exact quantile edges without needing them saved to disk
    (same discipline as fit_pooled_standardization/fit_quantile_edges elsewhere)."""
    from mlp_track_accumulation import FEATURES as MLP_FEATURES
    from mlp_track_accumulation import build_windowed_histogram_features, fit_quantile_edges

    if splits is None:
        splits = load_split()

    train_df = df.loc[df["sequence_name"].isin(splits["train"])]
    train_point_sets, _, _ = build_scan_point_sets_with_keys(train_df, classes, MLP_FEATURES)
    edges = fit_quantile_edges(train_point_sets, MLP_FEATURES, n_bins)
    del train_df, train_point_sets

    point_sets, labels, keys_df = build_scan_point_sets_with_keys(df, classes, MLP_FEATURES)
    X, _ = build_windowed_histogram_features(
        point_sets, MLP_FEATURES, n_bins, edges=edges, include_doppler_spread=False,
    )

    model = MLP(input_dim=X.shape[1], hidden_dim=mlp_hidden_dim, num_classes=len(classes)).to(DEVICE)
    model.load_state_dict(torch.load(encoder_dir / "mlp_model.pt", map_location=DEVICE))
    model.eval()

    embeddings = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch_x = torch.tensor(X[start : start + batch_size], device=DEVICE)
            embeddings.append(model.forward_features(batch_x))
    embeddings = torch.cat(embeddings, dim=0).cpu().numpy()

    result = keys_df.copy()
    result["label"] = labels
    for i in range(embeddings.shape[1]):
        result[f"e{i}"] = embeddings[:, i]
    return result


def get_or_compute_scan_embeddings_mlp(df: pd.DataFrame, cache_path, **kwargs) -> pd.DataFrame:
    """Same caching convention as get_or_compute_scan_embeddings, for the MLP-embedding
    line."""
    if cache_path.exists():
        print(f"{cache_path} already exists, loading cached embeddings")
        return pd.read_parquet(cache_path)
    result = compute_scan_embeddings_mlp(df, **kwargs)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(cache_path)
    print(f"Saved {cache_path}")
    return result


def build_windowed_embedding_sequences(
    embeddings_df: pd.DataFrame, classes: list[str] = MLP_CLASSES, n: int = WINDOW_N, stride: int = STRIDE,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Groups precomputed per-scan embeddings by track (sorted by timestamp), and for
    every stride-th scan, slices out the sequence of up to the last n scans' embeddings
    (this one included), oldest to newest. No padding here: a track with fewer than n
    scans just yields a shorter real sequence, the GRU runs however many steps that is
    via pack_padded_sequence at training time. Label is the target (last) scan's own
    class, same convention as every other windowed variant in this branch."""
    embedding_cols = [c for c in embeddings_df.columns if c.startswith("e")]
    ordered = embeddings_df.sort_values(TRACK_COLS + ["timestamp"]).reset_index(drop=True)
    emb_matrix = ordered[embedding_cols].to_numpy(dtype="float32")
    label_arr = ordered["label"].to_numpy()

    sequences, labels = [], []
    for _, positions in ordered.groupby(TRACK_COLS, sort=False).indices.items():
        for i in range(0, len(positions), stride):
            window_positions = positions[max(0, i - n + 1) : i + 1]
            sequences.append(emb_matrix[window_positions])
            labels.append(label_arr[window_positions[-1]])
    return sequences, np.array(labels, dtype="int64")


def pad_sequences(sequences: list[np.ndarray], m_max: int, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Left-aligned padding (real steps at positions 0..len-1, zeros after): the
    convention pack_padded_sequence expects, unlike the right-aligned zero-padding
    used for the fixed-length diff-vector feature elsewhere in this branch."""
    n = len(sequences)
    X = np.zeros((n, m_max, dim), dtype="float32")
    lengths = np.zeros(n, dtype="int64")
    for i, seq in enumerate(sequences):
        X[i, : len(seq)] = seq
        lengths[i] = len(seq)
    return X, lengths


def prepare_windowed_embedding_splits(
    embeddings_df: pd.DataFrame,
    classes: list[str] = MLP_CLASSES,
    n: int = WINDOW_N,
    stride: int = STRIDE,
    splits: dict[str, list[str]] | None = None,
):
    """Returns ragged per-window embedding sequences (list[np.ndarray], one array per
    window, length <= n, never padded here): at all-sensor N=50 scale (~870k train
    windows, m_max=50, dim=32) padding the whole split to one dense (n, m_max, dim)
    array costs ~5.6GB for train alone, with val/test and the ragged source lists
    still alive alongside it, which is what OOM-killed this exact computation (see
    track_accumulation.md). Padding now happens per mini-batch instead, same
    discipline as deepreflecs_track_accumulation's point-set ragged fix."""
    if splits is None:
        splits = load_split()

    train_df = embeddings_df.loc[embeddings_df["sequence_name"].isin(splits["train"])]
    val_df = embeddings_df.loc[embeddings_df["sequence_name"].isin(splits["val"])]
    test_df = embeddings_df.loc[embeddings_df["sequence_name"].isin(splits["test"])]

    train_seqs, y_train = build_windowed_embedding_sequences(train_df, classes, n, stride)
    val_seqs, y_val = build_windowed_embedding_sequences(val_df, classes, n, stride)
    test_seqs, y_test = build_windowed_embedding_sequences(test_df, classes, n, stride)

    print(f"N={n} stride={stride}: windows train={len(train_seqs)} val={len(val_seqs)} test={len(test_seqs)}")
    return train_seqs, y_train, val_seqs, y_val, test_seqs, y_test


class DeepReflecsGRU(nn.Module):
    """Unidirectional GRU over a track's per-scan DeepReflecs embeddings (frozen,
    precomputed, see compute_scan_embeddings). Causal: a window's prediction only ever
    depends on the GRU's hidden state after its own last real step (via
    pack_padded_sequence, so a padded tail never contributes), matching this project's
    real-time deployment constraint. That final hidden state feeds mlp_classifier.MLP
    for the actual classification, reusing that class rather than hand-rolling another
    linear+ReLU head."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = GRU_LAYERS,
        num_classes: int = len(MLP_CLASSES),
        mlp_hidden_dim: int = MLP_HIDDEN_DIM,
    ):
        super().__init__()
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = MLP(input_dim=hidden_size, hidden_dim=mlp_hidden_dim, num_classes=num_classes, n_hidden_layers=1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        return self.head(h_n[-1])


def _predict_in_batches(
    model: DeepReflecsGRU, sequences: list[np.ndarray], batch_size: int = 1024,
) -> torch.Tensor:
    """sequences is a ragged list, padded fresh per mini-batch (this batch's own max
    length) and moved to DEVICE per batch: at all-sensor scale (roughly 2.5x
    sensor2's window count) padding the whole split to one dense array first no
    longer fits in plain CPU RAM, let alone the 6GB GPU, same class of bug as the
    point-set trainer (see track_accumulation.md)."""
    model.eval()
    dim = sequences[0].shape[1]
    logits = []
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch_seqs = sequences[start : start + batch_size]
            batch_m_max = max(len(s) for s in batch_seqs)
            batch_x, batch_lengths = pad_sequences(batch_seqs, batch_m_max, dim)
            batch_x_t = torch.tensor(batch_x, device=DEVICE)
            batch_lengths_t = torch.tensor(batch_lengths)
            logits.append(model(batch_x_t, batch_lengths_t))
    return torch.cat(logits, dim=0)


def train_gru(
    train_seqs: list[np.ndarray],
    y_train: np.ndarray,
    val_seqs: list[np.ndarray],
    y_val: np.ndarray,
    classes: list[str] = MLP_CLASSES,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    random_state: int = RANDOM_STATE,
    hidden_size: int = HIDDEN_SIZE,
    num_layers: int = GRU_LAYERS,
    mlp_hidden_dim: int = MLP_HIDDEN_DIM,
):
    """Trains DeepReflecsGRU with Adam, class-count-weighted cross-entropy, same
    weighting scheme as train_deepreflecs/train_mlp. train_seqs/val_seqs are ragged
    lists, padded fresh per mini-batch (see _predict_in_batches)."""
    torch.manual_seed(random_state)

    weights_by_class = class_weights(pd.Series([classes[i] for i in y_train]))
    weight_tensor = torch.tensor(
        [weights_by_class[cls] for cls in classes], dtype=torch.float32, device=DEVICE
    )
    print(f"class weights: {dict(zip(classes, weight_tensor.tolist()))}")

    embedding_dim = train_seqs[0].shape[1]
    model = DeepReflecsGRU(
        embedding_dim, hidden_size=hidden_size, num_layers=num_layers, num_classes=len(classes),
        mlp_hidden_dim=mlp_hidden_dim,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    y_train_t = torch.tensor(y_train, device=DEVICE)
    y_val_t = torch.tensor(y_val, device=DEVICE)

    n = len(train_seqs)
    history = []
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

    return model, history


def _evaluate_gru_metrics(
    model: DeepReflecsGRU,
    sequences: list[np.ndarray],
    y_true: np.ndarray,
    classes: list[str],
    metrics_cache,
    model_cache,
    confusion_matrix_path,
    metrics_bar_path,
    split_name: str,
):
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_recall_fscore_support

    y_pred = _predict_in_batches(model, sequences).argmax(dim=1).cpu().numpy()

    if metrics_cache.exists() and metrics_cache.stat().st_mtime >= model_cache.stat().st_mtime:
        metrics_df = pd.read_json(metrics_cache, orient="index")
        print(f"{metrics_cache} already cached, reusing")
    else:
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=range(len(classes)), zero_division=0
        )
        metrics_df = pd.DataFrame(
            {"precision": precision, "recall": recall, "f1": f1, "support": support}, index=classes
        )
        metrics_cache.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_json(metrics_cache, orient="index", indent=2)
        print(f"Saved {metrics_cache}")
    print(f"per-class precision/recall/f1 ({split_name}):")
    print(metrics_df.round(3).to_string())

    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)), normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, colorbar=False, values_format=".2f")
    ax.set_title(f"DeepReflecs+GRU: {split_name} confusion matrix (row-normalized)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    confusion_matrix_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(confusion_matrix_path, dpi=150)
    print(f"Saved {confusion_matrix_path}")

    bar_fig, bar_ax = plt.subplots(figsize=(9, 5))
    metrics_df[["precision", "recall", "f1"]].plot(kind="bar", ax=bar_ax, edgecolor="k")
    bar_ax.set_ylim(0, 1)
    bar_ax.set_ylabel("score")
    bar_ax.set_title(f"DeepReflecs+GRU: per-class precision/recall/f1 ({split_name})")
    bar_ax.legend(loc="lower right")
    plt.setp(bar_ax.get_xticklabels(), rotation=45, ha="right")
    bar_fig.tight_layout()
    bar_fig.savefig(metrics_bar_path, dpi=150)
    print(f"Saved {metrics_bar_path}")

    return metrics_df, fig, bar_fig


def run_gru_training(
    embeddings_df: pd.DataFrame,
    n: int = WINDOW_N,
    stride: int = STRIDE,
    classes: list[str] = MLP_CLASSES,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    random_state: int = RANDOM_STATE,
    hidden_size: int = HIDDEN_SIZE,
    num_layers: int = GRU_LAYERS,
    mlp_hidden_dim: int = MLP_HIDDEN_DIM,
    output_dir=None,
    splits: dict[str, list[str]] | None = None,
):
    if output_dir is None:
        output_dir = RNN_DIR / f"N{n}_stride{stride}_gru_h{hidden_size}"
    if splits is None:
        splits = load_split()

    cache_key = {
        "n": n, "stride": stride, "classes": classes, "splits": splits, "hidden_size": hidden_size,
        "num_layers": num_layers, "mlp_hidden_dim": mlp_hidden_dim, "epochs": epochs, "batch_size": batch_size,
        "lr": lr, "random_state": random_state,
    }
    history_cache = output_dir / "gru_training_history.json"
    model_cache = output_dir / "gru_model.pt"

    train_seqs, y_train, val_seqs, y_val, test_seqs, y_test = prepare_windowed_embedding_splits(
        embeddings_df, classes=classes, n=n, stride=stride, splits=splits,
    )
    embedding_dim = train_seqs[0].shape[1]

    if history_cache.exists() and model_cache.exists():
        cached = json.loads(history_cache.read_text())
        if cached.get("key") == cache_key:
            print(f"{history_cache} already matches this config, loading cached model + history")
            model = DeepReflecsGRU(
                embedding_dim, hidden_size=hidden_size, num_layers=num_layers, num_classes=len(classes),
                mlp_hidden_dim=mlp_hidden_dim,
            ).to(DEVICE)
            model.load_state_dict(torch.load(model_cache, map_location=DEVICE))
            plot_training_curves(cached["history"], output_dir=output_dir)
            return model, cached["history"], test_seqs, y_test
        print(f"{history_cache} doesn't match this config, retraining")

    model, history = train_gru(
        train_seqs, y_train, val_seqs, y_val, classes=classes, epochs=epochs,
        batch_size=batch_size, lr=lr, random_state=random_state, hidden_size=hidden_size,
        num_layers=num_layers, mlp_hidden_dim=mlp_hidden_dim,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    history_cache.write_text(json.dumps({"key": cache_key, "history": history}, indent=2))
    torch.save(model.state_dict(), model_cache)
    print(f"Saved {history_cache} and {model_cache}")

    plot_training_curves(history, output_dir=output_dir)
    return model, history, test_seqs, y_test


def evaluate_gru_test_metrics(
    model: DeepReflecsGRU,
    test_seqs: list[np.ndarray],
    y_test: np.ndarray,
    classes: list[str] = MLP_CLASSES,
    output_dir=None,
    n: int = WINDOW_N,
    stride: int = STRIDE,
    hidden_size: int = HIDDEN_SIZE,
):
    if output_dir is None:
        output_dir = RNN_DIR / f"N{n}_stride{stride}_gru_h{hidden_size}"
    model_cache = output_dir / "gru_model.pt"
    return _evaluate_gru_metrics(
        model, test_seqs, y_test, classes,
        metrics_cache=output_dir / "gru_test_metrics.json", model_cache=model_cache,
        confusion_matrix_path=output_dir / "gru_test_confusion_matrix.png",
        metrics_bar_path=output_dir / "gru_test_precision_recall_f1.png",
        split_name=f"test (N={n}, stride={stride}, hidden_size={hidden_size})",
    )


# --- causal transformer variant: same precomputed embeddings/windowing as the GRU,
# self-attention instead of recurrence over the FIFO buffer's sequence ---
D_MODEL = 32  # == point_dim, no input projection needed
NUM_HEADS = 4  # 32 / 4 = 8 per head
NUM_LAYERS = 1
DIM_FEEDFORWARD = 64  # 2x d_model, modest given the short sequence and small embedding


class DeepReflecsTransformer(nn.Module):
    """Causal self-attention counterpart to DeepReflecsGRU: same frozen, precomputed
    per-scan embeddings (compute_scan_embeddings) and same windowing, consumed by a
    small causal Transformer encoder instead of a GRU. Needs two things a GRU gets for
    free: an explicit causal mask (attention has no inherent notion of order or
    direction, unlike recurrence) and a learned positional embedding per FIFO-buffer
    slot (0..max_seq_len-1), added to the input embeddings before the first layer.
    The output at each sequence's own last real position (mirrors the GRU's final
    hidden state h_t) feeds the same MLP head."""

    def __init__(
        self,
        embedding_dim: int,
        max_seq_len: int,
        d_model: int = D_MODEL,
        num_heads: int = NUM_HEADS,
        num_layers: int = NUM_LAYERS,
        dim_feedforward: int = DIM_FEEDFORWARD,
        num_classes: int = len(MLP_CLASSES),
        mlp_hidden_dim: int = MLP_HIDDEN_DIM,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_proj = nn.Identity() if embedding_dim == d_model else nn.Linear(embedding_dim, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = MLP(input_dim=d_model, hidden_dim=mlp_hidden_dim, num_classes=num_classes, n_hidden_layers=1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.input_proj(x) + self.pos_embedding(positions)

        causal_mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        lengths = lengths.to(x.device)
        key_padding_mask = torch.arange(T, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)

        out = self.encoder(h, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        last_idx = lengths - 1
        last_hidden = out[torch.arange(B, device=x.device), last_idx]
        return self.head(last_hidden)


def _predict_transformer_in_batches(
    model: DeepReflecsTransformer, sequences: list[np.ndarray], batch_size: int = 1024,
) -> torch.Tensor:
    """sequences is a ragged list, padded fresh per mini-batch, same reason as
    _predict_in_batches."""
    model.eval()
    dim = sequences[0].shape[1]
    logits = []
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch_seqs = sequences[start : start + batch_size]
            batch_m_max = max(len(s) for s in batch_seqs)
            batch_x, batch_lengths = pad_sequences(batch_seqs, batch_m_max, dim)
            batch_x_t = torch.tensor(batch_x, device=DEVICE)
            batch_lengths_t = torch.tensor(batch_lengths)
            logits.append(model(batch_x_t, batch_lengths_t))
    return torch.cat(logits, dim=0)


def train_transformer(
    train_seqs: list[np.ndarray],
    y_train: np.ndarray,
    val_seqs: list[np.ndarray],
    y_val: np.ndarray,
    classes: list[str] = MLP_CLASSES,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    random_state: int = RANDOM_STATE,
    max_seq_len: int = WINDOW_N,
    d_model: int = D_MODEL,
    num_heads: int = NUM_HEADS,
    num_layers: int = NUM_LAYERS,
    dim_feedforward: int = DIM_FEEDFORWARD,
    mlp_hidden_dim: int = MLP_HIDDEN_DIM,
):
    """Same Adam + class-count-weighted cross-entropy convention as train_gru.
    train_seqs/val_seqs are ragged lists, padded fresh per mini-batch (see
    _predict_transformer_in_batches). max_seq_len sizes the positional embedding
    table and must be >= any real sequence length (defaults to the window size n,
    which every sequence is built to be at most)."""
    torch.manual_seed(random_state)

    weights_by_class = class_weights(pd.Series([classes[i] for i in y_train]))
    weight_tensor = torch.tensor(
        [weights_by_class[cls] for cls in classes], dtype=torch.float32, device=DEVICE
    )
    print(f"class weights: {dict(zip(classes, weight_tensor.tolist()))}")

    embedding_dim = train_seqs[0].shape[1]
    model = DeepReflecsTransformer(
        embedding_dim, max_seq_len, d_model=d_model, num_heads=num_heads, num_layers=num_layers,
        dim_feedforward=dim_feedforward, num_classes=len(classes), mlp_hidden_dim=mlp_hidden_dim,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    y_train_t = torch.tensor(y_train, device=DEVICE)
    y_val_t = torch.tensor(y_val, device=DEVICE)

    n = len(train_seqs)
    history = []
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

        val_logits = _predict_transformer_in_batches(model, val_seqs)
        val_acc = (val_logits.argmax(dim=1) == y_val_t).float().mean().item()

        history.append({"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc, "val_acc": val_acc})
        print(f"epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

    return model, history


def _evaluate_transformer_metrics(
    model: DeepReflecsTransformer,
    sequences: list[np.ndarray],
    y_true: np.ndarray,
    classes: list[str],
    metrics_cache,
    model_cache,
    confusion_matrix_path,
    metrics_bar_path,
    split_name: str,
):
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_recall_fscore_support

    y_pred = _predict_transformer_in_batches(model, sequences).argmax(dim=1).cpu().numpy()

    if metrics_cache.exists() and metrics_cache.stat().st_mtime >= model_cache.stat().st_mtime:
        metrics_df = pd.read_json(metrics_cache, orient="index")
        print(f"{metrics_cache} already cached, reusing")
    else:
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=range(len(classes)), zero_division=0
        )
        metrics_df = pd.DataFrame(
            {"precision": precision, "recall": recall, "f1": f1, "support": support}, index=classes
        )
        metrics_cache.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_json(metrics_cache, orient="index", indent=2)
        print(f"Saved {metrics_cache}")
    print(f"per-class precision/recall/f1 ({split_name}):")
    print(metrics_df.round(3).to_string())

    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)), normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, colorbar=False, values_format=".2f")
    ax.set_title(f"DeepReflecs+Transformer: {split_name} confusion matrix (row-normalized)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    confusion_matrix_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(confusion_matrix_path, dpi=150)
    print(f"Saved {confusion_matrix_path}")

    bar_fig, bar_ax = plt.subplots(figsize=(9, 5))
    metrics_df[["precision", "recall", "f1"]].plot(kind="bar", ax=bar_ax, edgecolor="k")
    bar_ax.set_ylim(0, 1)
    bar_ax.set_ylabel("score")
    bar_ax.set_title(f"DeepReflecs+Transformer: per-class precision/recall/f1 ({split_name})")
    bar_ax.legend(loc="lower right")
    plt.setp(bar_ax.get_xticklabels(), rotation=45, ha="right")
    bar_fig.tight_layout()
    bar_fig.savefig(metrics_bar_path, dpi=150)
    print(f"Saved {metrics_bar_path}")

    return metrics_df, fig, bar_fig


def run_transformer_training(
    embeddings_df: pd.DataFrame,
    n: int = WINDOW_N,
    stride: int = STRIDE,
    classes: list[str] = MLP_CLASSES,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    random_state: int = RANDOM_STATE,
    d_model: int = D_MODEL,
    num_heads: int = NUM_HEADS,
    num_layers: int = NUM_LAYERS,
    dim_feedforward: int = DIM_FEEDFORWARD,
    mlp_hidden_dim: int = MLP_HIDDEN_DIM,
    output_dir=None,
    splits: dict[str, list[str]] | None = None,
):
    if output_dir is None:
        output_dir = RNN_DIR / f"N{n}_stride{stride}_transformer_d{d_model}"
    if splits is None:
        splits = load_split()

    cache_key = {
        "n": n, "stride": stride, "classes": classes, "splits": splits, "d_model": d_model,
        "num_heads": num_heads, "num_layers": num_layers, "dim_feedforward": dim_feedforward,
        "mlp_hidden_dim": mlp_hidden_dim, "epochs": epochs, "batch_size": batch_size, "lr": lr,
        "random_state": random_state,
    }
    history_cache = output_dir / "transformer_training_history.json"
    model_cache = output_dir / "transformer_model.pt"

    train_seqs, y_train, val_seqs, y_val, test_seqs, y_test = prepare_windowed_embedding_splits(
        embeddings_df, classes=classes, n=n, stride=stride, splits=splits,
    )
    embedding_dim = train_seqs[0].shape[1]
    max_seq_len = n

    if history_cache.exists() and model_cache.exists():
        cached = json.loads(history_cache.read_text())
        if cached.get("key") == cache_key:
            print(f"{history_cache} already matches this config, loading cached model + history")
            model = DeepReflecsTransformer(
                embedding_dim, max_seq_len, d_model=d_model, num_heads=num_heads, num_layers=num_layers,
                dim_feedforward=dim_feedforward, num_classes=len(classes), mlp_hidden_dim=mlp_hidden_dim,
            ).to(DEVICE)
            model.load_state_dict(torch.load(model_cache, map_location=DEVICE))
            plot_training_curves(cached["history"], output_dir=output_dir)
            return model, cached["history"], test_seqs, y_test
        print(f"{history_cache} doesn't match this config, retraining")

    model, history = train_transformer(
        train_seqs, y_train, val_seqs, y_val, classes=classes, epochs=epochs,
        batch_size=batch_size, lr=lr, random_state=random_state, max_seq_len=max_seq_len, d_model=d_model,
        num_heads=num_heads, num_layers=num_layers, dim_feedforward=dim_feedforward, mlp_hidden_dim=mlp_hidden_dim,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    history_cache.write_text(json.dumps({"key": cache_key, "history": history}, indent=2))
    torch.save(model.state_dict(), model_cache)
    print(f"Saved {history_cache} and {model_cache}")

    plot_training_curves(history, output_dir=output_dir)
    return model, history, test_seqs, y_test


def evaluate_transformer_test_metrics(
    model: DeepReflecsTransformer,
    test_seqs: list[np.ndarray],
    y_test: np.ndarray,
    classes: list[str] = MLP_CLASSES,
    output_dir=None,
    n: int = WINDOW_N,
    stride: int = STRIDE,
    d_model: int = D_MODEL,
):
    if output_dir is None:
        output_dir = RNN_DIR / f"N{n}_stride{stride}_transformer_d{d_model}"
    model_cache = output_dir / "transformer_model.pt"
    return _evaluate_transformer_metrics(
        model, test_seqs, y_test, classes,
        metrics_cache=output_dir / "transformer_test_metrics.json", model_cache=model_cache,
        confusion_matrix_path=output_dir / "transformer_test_confusion_matrix.png",
        metrics_bar_path=output_dir / "transformer_test_precision_recall_f1.png",
        split_name=f"test (N={n}, stride={stride}, d_model={d_model})",
    )


# --- end-to-end variant: encoder not frozen, gradients flow through it every window ---
# (see track_accumulation.md, "DeepReflecs + GRU": precompute-then-train was chosen
# first because it costs nothing extra to re-use a scan across N overlapping stride=1
# windows; end-to-end reintroduces that redundant compute deliberately, to test
# whether letting the encoder adapt to the temporal task beats its frozen embeddings.)


def standardize_features(
    df: pd.DataFrame, mean: np.ndarray, std: np.ndarray, features: list[str] = REFLECTION_FEATURES,
) -> pd.DataFrame:
    df = df.copy()
    df[features] = (df[features].to_numpy(dtype="float32") - mean) / std
    return df


def build_windowed_scan_point_sequences(
    df: pd.DataFrame, classes: list[str] = MLP_CLASSES, features: list[str] = REFLECTION_FEATURES,
    n: int = WINDOW_N, stride: int = STRIDE,
) -> tuple[list[list[np.ndarray]], np.ndarray]:
    """Same windowing as deepreflecs_track_accumulation.build_windowed_point_sets, but
    keeps each window's scans separate (a list of per-scan point arrays) instead of
    concatenating them into one flat pooled point set: end-to-end training still needs
    each scan's own point-level mask for its own encoder forward pass, pooling would
    throw that boundary away."""
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    mask = df["group"].isin(classes)
    filtered = df.loc[mask]

    feat_matrix = filtered[features].to_numpy(dtype="float32")
    class_idx = filtered["group"].map(class_to_idx).to_numpy()

    scan_positions = filtered.groupby(INSTANCE_COLS, sort=False).indices
    scan_arrays = list(scan_positions.values())
    scan_keys = pd.DataFrame(list(scan_positions.keys()), columns=INSTANCE_COLS)
    scan_keys["_scan_idx"] = np.arange(len(scan_keys))
    scan_keys = scan_keys.sort_values(TRACK_COLS + ["timestamp"])

    sequences, labels = [], []
    for _, track_scans in scan_keys.groupby(TRACK_COLS, sort=False):
        ordered = track_scans["_scan_idx"].to_numpy()
        for i in range(0, len(ordered), stride):
            window_idx = ordered[max(0, i - n + 1) : i + 1]
            sequences.append([feat_matrix[scan_arrays[j]] for j in window_idx])
            labels.append(class_idx[scan_arrays[ordered[i]][0]])
    return sequences, np.array(labels, dtype="int64")


def collate_scan_sequences(
    batch_seqs: list[list[np.ndarray]], max_seq_len: int, n_features: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pads a single mini-batch's windows to max_seq_len (fixed, =N, small) and to
    THIS BATCH's own max points-per-scan, not the whole dataset's worst case.
    Points-per-scan is heavily skewed in this dataset (median 2, 99th pct 13, one
    single scan somewhere hits 45), so padding every one of N scan-slots in every one
    of hundreds of thousands of windows to the global 45 wastes roughly 45/2.9 ~ 15x
    the memory a typical window actually needs, that's what OOM-killed the first
    attempt at building one dense array for a whole split up front (see
    track_accumulation.md). Collating fresh per mini-batch keeps the resident array
    small (this batch's own worst case, usually nowhere near 45) at the cost of a
    small amount of repeated Python-level padding work per step, cheap next to the
    encoder forward/backward pass that step already does."""
    B = len(batch_seqs)
    max_points = max(scan.shape[0] for window in batch_seqs for scan in window)
    X = np.zeros((B, max_seq_len, max_points, n_features), dtype="float32")
    point_mask = np.zeros((B, max_seq_len, max_points), dtype=bool)
    valid_scan_mask = np.zeros((B, max_seq_len), dtype=bool)
    lengths = np.zeros(B, dtype="int64")
    for i, window in enumerate(batch_seqs):
        lengths[i] = len(window)
        for t, scan in enumerate(window):
            X[i, t, : len(scan)] = scan
            point_mask[i, t, : len(scan)] = True
            valid_scan_mask[i, t] = True
    return X, point_mask, valid_scan_mask, lengths


def prepare_end_to_end_splits(
    df: pd.DataFrame, classes: list[str] = MLP_CLASSES, features: list[str] = REFLECTION_FEATURES,
    n: int = WINDOW_N, stride: int = STRIDE, splits: dict[str, list[str]] | None = None,
):
    """Returns the raw per-window scan-sequence lists and labels for each split, NOT a
    pre-padded dense array: collate_scan_sequences pads per mini-batch instead (see
    its docstring for why building one dense array up front OOMs on the full
    dataset)."""
    if splits is None:
        splits = load_split()

    mean, std = fit_reflection_standardization(df, splits, classes, features)
    df_std = standardize_features(df, mean, std, features)

    train_df = df_std.loc[df_std["sequence_name"].isin(splits["train"])]
    val_df = df_std.loc[df_std["sequence_name"].isin(splits["val"])]
    test_df = df_std.loc[df_std["sequence_name"].isin(splits["test"])]

    train_seqs, y_train = build_windowed_scan_point_sequences(train_df, classes, features, n, stride)
    val_seqs, y_val = build_windowed_scan_point_sequences(val_df, classes, features, n, stride)
    test_seqs, y_test = build_windowed_scan_point_sequences(test_df, classes, features, n, stride)

    max_seq_len = max(len(w) for w in (*train_seqs, *val_seqs, *test_seqs))

    print(
        f"end-to-end N={n} stride={stride}: windows train={len(train_seqs)} val={len(val_seqs)} "
        f"test={len(test_seqs)}, max_seq_len={max_seq_len}"
    )
    return train_seqs, y_train, val_seqs, y_val, test_seqs, y_test, max_seq_len


class DeepReflecsGRUEndToEnd(nn.Module):
    """End-to-end version of DeepReflecsGRU: the point encoder is not frozen or
    precomputed, it runs inside forward() on every real scan of every window, so
    gradients from the classification loss flow all the way back into the per-point
    layers. Only real (non-padding) scans are ever passed through the encoder
    (valid_scan_mask), both to avoid wasted compute on scans that don't exist and to
    avoid running the masked max-pool on an all-padding point mask (would reduce to
    -inf for every feature, better to just never construct that case)."""

    def __init__(
        self,
        n_features: int,
        num_classes: int,
        conv_dim: int = CONV_DIM,
        point_dim: int = POINT_DIM,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = GRU_LAYERS,
        mlp_hidden_dim: int = MLP_HIDDEN_DIM,
    ):
        super().__init__()
        self.encoder = DeepReflecs(n_features, num_classes=num_classes, conv_dim=conv_dim, point_dim=point_dim)
        self.gru = nn.GRU(point_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = MLP(input_dim=hidden_size, hidden_dim=mlp_hidden_dim, num_classes=num_classes, n_hidden_layers=1)

    def forward(
        self, points: torch.Tensor, point_mask: torch.Tensor, valid_scan_mask: torch.Tensor, lengths: torch.Tensor,
    ) -> torch.Tensor:
        B, T, M, F = points.shape
        flat_valid = valid_scan_mask.reshape(-1)
        valid_points = points.reshape(B * T, M, F)[flat_valid]
        valid_point_mask = point_mask.reshape(B * T, M)[flat_valid]
        valid_embeddings = embed_points(self.encoder, valid_points, valid_point_mask)

        flat_embeddings = torch.zeros(B * T, valid_embeddings.shape[1], device=points.device, dtype=valid_embeddings.dtype)
        flat_embeddings[flat_valid] = valid_embeddings
        embeddings = flat_embeddings.reshape(B, T, -1)

        packed = nn.utils.rnn.pack_padded_sequence(embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        return self.head(h_n[-1])


def _predict_e2e_in_batches(
    model: DeepReflecsGRUEndToEnd, seqs: list[list[np.ndarray]], max_seq_len: int, n_features: int,
    batch_size: int = 256,
) -> torch.Tensor:
    model.eval()
    logits = []
    with torch.no_grad():
        for start in range(0, len(seqs), batch_size):
            X, mask, valid, lengths = collate_scan_sequences(seqs[start : start + batch_size], max_seq_len, n_features)
            logits.append(model(
                torch.tensor(X, device=DEVICE), torch.tensor(mask, device=DEVICE),
                torch.tensor(valid, device=DEVICE), torch.tensor(lengths),
            ))
    return torch.cat(logits, dim=0)


def train_end_to_end(
    train_seqs: list[list[np.ndarray]], y_train: np.ndarray,
    val_seqs: list[list[np.ndarray]], y_val: np.ndarray,
    max_seq_len: int,
    n_features: int = len(REFLECTION_FEATURES),
    classes: list[str] = MLP_CLASSES,
    epochs: int = EPOCHS,
    batch_size: int = 64,
    lr: float = LEARNING_RATE,
    random_state: int = RANDOM_STATE,
    conv_dim: int = CONV_DIM,
    point_dim: int = POINT_DIM,
    hidden_size: int = HIDDEN_SIZE,
    num_layers: int = GRU_LAYERS,
    mlp_hidden_dim: int = MLP_HIDDEN_DIM,
    init: str = "warmstart",
    encoder_dir=ENCODER_DIR,
    precompute_model_dir=None,
):
    """init="warmstart": encoder starts from the N=1 control checkpoint, GRU+head
    start from the already-trained precompute GRU checkpoint (precompute_model_dir),
    nothing in the model is random at step 0, isolating whether adapting an already-
    good encoder to the temporal task beats its frozen embeddings. init="random":
    everything (encoder, GRU, head) is randomly initialized, a harder and different
    question, whether the whole stack can learn shape from raw points with no single-
    scan supervision at all.

    train_seqs/val_seqs are the raw per-window scan-sequence lists (see
    prepare_end_to_end_splits), collated into a padded batch fresh at every step
    (collate_scan_sequences), never pre-built as one dense array for the whole split:
    that pre-built version is what OOM-killed the first attempt at this on the full
    dataset (points-per-scan is heavily skewed, padding every scan slot in every
    window to the dataset-wide worst case wastes roughly an order of magnitude of
    memory, see collate_scan_sequences' docstring)."""
    if init not in ("warmstart", "random"):
        raise ValueError(f"init must be 'warmstart' or 'random', got {init!r}")
    torch.manual_seed(random_state)

    weights_by_class = class_weights(pd.Series([classes[i] for i in y_train]))
    weight_tensor = torch.tensor(
        [weights_by_class[cls] for cls in classes], dtype=torch.float32, device=DEVICE
    )
    print(f"class weights: {dict(zip(classes, weight_tensor.tolist()))}")

    model = DeepReflecsGRUEndToEnd(
        n_features, num_classes=len(classes), conv_dim=conv_dim, point_dim=point_dim, hidden_size=hidden_size,
        num_layers=num_layers, mlp_hidden_dim=mlp_hidden_dim,
    ).to(DEVICE)

    if init == "warmstart":
        model.encoder.load_state_dict(torch.load(encoder_dir / "deepreflecs_model.pt", map_location=DEVICE))
        precompute_state = torch.load(precompute_model_dir / "gru_model.pt", map_location=DEVICE)
        missing, unexpected = model.load_state_dict(precompute_state, strict=False)
        assert not unexpected, f"unexpected keys loading precompute GRU checkpoint: {unexpected}"
        print(f"warm-started encoder from {encoder_dir}, gru+head from {precompute_model_dir}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    y_train_t = torch.tensor(y_train, device=DEVICE)
    y_val_t = torch.tensor(y_val, device=DEVICE)

    n = len(train_seqs)
    history = []
    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(n)
        epoch_loss, epoch_correct = 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            batch_seqs = [train_seqs[i] for i in idx]
            X, mask, valid, lengths = collate_scan_sequences(batch_seqs, max_seq_len, n_features)
            batch_x = torch.tensor(X, device=DEVICE)
            batch_mask = torch.tensor(mask, device=DEVICE)
            batch_valid = torch.tensor(valid, device=DEVICE)
            batch_lengths = torch.tensor(lengths)
            batch_y = y_train_t[idx]

            optimizer.zero_grad()
            logits = model(batch_x, batch_mask, batch_valid, batch_lengths)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(idx)
            epoch_correct += (logits.argmax(dim=1) == batch_y).sum().item()

        train_loss = epoch_loss / n
        train_acc = epoch_correct / n

        val_logits = _predict_e2e_in_batches(model, val_seqs, max_seq_len, n_features)
        val_acc = (val_logits.argmax(dim=1) == y_val_t).float().mean().item()

        history.append({"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc, "val_acc": val_acc})
        print(f"epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

    return model, history


def run_end_to_end_training(
    df: pd.DataFrame,
    n: int = WINDOW_N,
    stride: int = STRIDE,
    classes: list[str] = MLP_CLASSES,
    features: list[str] = REFLECTION_FEATURES,
    epochs: int = EPOCHS,
    batch_size: int = 64,
    lr: float = LEARNING_RATE,
    random_state: int = RANDOM_STATE,
    conv_dim: int = CONV_DIM,
    point_dim: int = POINT_DIM,
    hidden_size: int = HIDDEN_SIZE,
    num_layers: int = GRU_LAYERS,
    mlp_hidden_dim: int = MLP_HIDDEN_DIM,
    init: str = "warmstart",
    encoder_dir=ENCODER_DIR,
    precompute_model_dir=None,
    output_dir=None,
    splits: dict[str, list[str]] | None = None,
):
    if precompute_model_dir is None:
        precompute_model_dir = RNN_DIR / f"N{n}_stride{stride}_gru_h{hidden_size}"
    if output_dir is None:
        output_dir = RNN_DIR / f"N{n}_stride{stride}_e2e_{init}_h{hidden_size}"
    if splits is None:
        splits = load_split()

    cache_key = {
        "n": n, "stride": stride, "classes": classes, "splits": splits, "hidden_size": hidden_size,
        "num_layers": num_layers, "mlp_hidden_dim": mlp_hidden_dim, "conv_dim": conv_dim, "point_dim": point_dim,
        "init": init, "epochs": epochs, "batch_size": batch_size, "lr": lr, "random_state": random_state,
    }
    history_cache = output_dir / "e2e_training_history.json"
    model_cache = output_dir / "e2e_model.pt"
    n_features = len(features)

    train_seqs, y_train, val_seqs, y_val, test_seqs, y_test, max_seq_len = prepare_end_to_end_splits(
        df, classes=classes, features=features, n=n, stride=stride, splits=splits,
    )

    if history_cache.exists() and model_cache.exists():
        cached = json.loads(history_cache.read_text())
        if cached.get("key") == cache_key:
            print(f"{history_cache} already matches this config, loading cached model + history")
            model = DeepReflecsGRUEndToEnd(
                n_features, num_classes=len(classes), conv_dim=conv_dim, point_dim=point_dim,
                hidden_size=hidden_size, num_layers=num_layers, mlp_hidden_dim=mlp_hidden_dim,
            ).to(DEVICE)
            model.load_state_dict(torch.load(model_cache, map_location=DEVICE))
            plot_training_curves(cached["history"], output_dir=output_dir)
            return model, cached["history"], test_seqs, y_test, max_seq_len, n_features
        print(f"{history_cache} doesn't match this config, retraining")

    model, history = train_end_to_end(
        train_seqs, y_train, val_seqs, y_val, max_seq_len, n_features=n_features,
        classes=classes, epochs=epochs, batch_size=batch_size, lr=lr, random_state=random_state,
        conv_dim=conv_dim, point_dim=point_dim, hidden_size=hidden_size, num_layers=num_layers,
        mlp_hidden_dim=mlp_hidden_dim, init=init, encoder_dir=encoder_dir, precompute_model_dir=precompute_model_dir,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    history_cache.write_text(json.dumps({"key": cache_key, "history": history}, indent=2))
    torch.save(model.state_dict(), model_cache)
    print(f"Saved {history_cache} and {model_cache}")

    plot_training_curves(history, output_dir=output_dir)
    return model, history, test_seqs, y_test, max_seq_len, n_features


def evaluate_end_to_end_test_metrics(
    model: DeepReflecsGRUEndToEnd,
    test_seqs: list[list[np.ndarray]],
    y_test: np.ndarray,
    max_seq_len: int,
    n_features: int,
    classes: list[str] = MLP_CLASSES,
    output_dir=None,
    n: int = WINDOW_N,
    stride: int = STRIDE,
    hidden_size: int = HIDDEN_SIZE,
    init: str = "warmstart",
):
    if output_dir is None:
        output_dir = RNN_DIR / f"N{n}_stride{stride}_e2e_{init}_h{hidden_size}"
    model_cache = output_dir / "e2e_model.pt"

    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_recall_fscore_support

    y_pred = _predict_e2e_in_batches(model, test_seqs, max_seq_len, n_features).argmax(dim=1).cpu().numpy()

    metrics_cache = output_dir / "e2e_test_metrics.json"
    if metrics_cache.exists() and metrics_cache.stat().st_mtime >= model_cache.stat().st_mtime:
        metrics_df = pd.read_json(metrics_cache, orient="index")
        print(f"{metrics_cache} already cached, reusing")
    else:
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, labels=range(len(classes)), zero_division=0
        )
        metrics_df = pd.DataFrame(
            {"precision": precision, "recall": recall, "f1": f1, "support": support}, index=classes
        )
        metrics_cache.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_json(metrics_cache, orient="index", indent=2)
        print(f"Saved {metrics_cache}")

    split_name = f"test (N={n}, stride={stride}, hidden_size={hidden_size}, init={init})"
    print(f"per-class precision/recall/f1 ({split_name}):")
    print(metrics_df.round(3).to_string())

    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_test, y_pred, labels=range(len(classes)), normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, colorbar=False, values_format=".2f")
    ax.set_title(f"DeepReflecs+GRU (end-to-end, {init}): {split_name} confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    confusion_matrix_path = output_dir / "e2e_test_confusion_matrix.png"
    fig.savefig(confusion_matrix_path, dpi=150)
    print(f"Saved {confusion_matrix_path}")

    return metrics_df, fig


# --- fusion variant: pooled DeepReflecs (all N scans concatenated, order-blind) +
# GRU's final hidden state (sequence-aware), concatenated into one feature vector for
# a fresh MLP head. Direct test of whether the sequence branch knows anything the
# pooled branch doesn't: both frozen, reused from already-trained checkpoints, only
# the small fusion MLP head is new/trained here. See track_accumulation.md's "late
# fusion" discussion for the reasoning (U-Net-style skip-connection analogy: splice
# in a view that the other branch's own bottleneck would otherwise discard). ---
from deepreflecs_track_accumulation import build_windowed_point_sets as build_pooled_point_sets

POOLED_ENCODER_DIR = TRACK_ACC_DIR / f"N{WINDOW_N}_stride{STRIDE}"  # already-trained pooled DeepReflecs, broadcast range_sc
FUSION_MLP_HIDDEN_DIM = 16


def fit_pooled_standardization(
    df: pd.DataFrame, splits: dict[str, list[str]], classes: list[str] = MLP_CLASSES,
    features: list[str] = REFLECTION_FEATURES, n: int = WINDOW_N, stride: int = STRIDE,
    range_sc_mode: str = "broadcast",
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduces exactly the mean/std the pooled N=10 DeepReflecs model was trained
    with (deepreflecs_track_accumulation.prepare_windowed_split_point_sets): fit on
    train's real, non-padding points only. Same discipline as fit_reflection_
    standardization, just windowed (N scans pooled) instead of single-scan.

    Concatenates the ragged point sets directly instead of padding to one shared
    m_max first: mean/std only need the real points, padding would only add zeros
    that then get masked back out anyway, so padding here is pure wasted memory. At
    large N (m_max in the thousands, hundreds of thousands of windows) padding a
    whole split first is what OOM-killed this exact computation (see
    track_accumulation.md, "Pooled DeepReflecs point-set at N=50")."""
    train_df = df.loc[df["sequence_name"].isin(splits["train"])]
    train_sets, _ = build_pooled_point_sets(train_df, classes, features, n, stride, range_sc_mode)
    all_points = np.concatenate(train_sets)
    mean = all_points.mean(axis=0).astype("float32")
    std = all_points.std(axis=0).astype("float32")
    return mean, np.where(std > 0, std, 1.0).astype("float32")


def compute_pooled_embeddings(
    df: pd.DataFrame, split_sequences: list[str], mean: np.ndarray, std: np.ndarray,
    classes: list[str] = MLP_CLASSES, features: list[str] = REFLECTION_FEATURES, n: int = WINDOW_N,
    stride: int = STRIDE, range_sc_mode: str = "broadcast", encoder_dir=POOLED_ENCODER_DIR,
    conv_dim: int = CONV_DIM, point_dim: int = POINT_DIM, batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Runs the already-trained, frozen pooled N=10 DeepReflecs encoder on one split's
    windows (all N scans' points concatenated per window, same as that model's own
    training data), returns its pre-classifier embedding (point_dim-wide) and labels.

    Pads fresh per mini-batch (this batch's own max points), never the whole split at
    once: at large N a single outlier window can push the split-wide m_max into the
    thousands, and padding every window in a 350k+-window split to that one shared
    value OOMs in plain CPU memory before any GPU work starts, same mechanism as
    deepreflecs_track_accumulation's point-set trainer (see track_accumulation.md,
    "Pooled DeepReflecs point-set at N=50"). This function had the identical bug,
    just one level up (embedding inference instead of training)."""
    split_df = df.loc[df["sequence_name"].isin(split_sequences)]
    point_sets, labels = build_pooled_point_sets(split_df, classes, features, n, stride, range_sc_mode)
    point_sets_std = [(p - mean) / std for p in point_sets]

    model = DeepReflecs(len(features), num_classes=len(classes), conv_dim=conv_dim, point_dim=point_dim).to(DEVICE)
    model.load_state_dict(torch.load(encoder_dir / "deepreflecs_model.pt", map_location=DEVICE))
    model.eval()

    embeddings = []
    with torch.no_grad():
        for start in range(0, len(point_sets_std), batch_size):
            batch_sets = point_sets_std[start : start + batch_size]
            batch_m_max = max(p.shape[0] for p in batch_sets)
            batch_x, batch_mask = pad_to_fixed(batch_sets, batch_m_max)
            batch_x_t = torch.tensor(batch_x, device=DEVICE)
            batch_mask_t = torch.tensor(batch_mask, device=DEVICE)
            embeddings.append(embed_points(model, batch_x_t, batch_mask_t))
    return torch.cat(embeddings, dim=0).cpu().numpy(), labels


def compute_gru_hidden_states(
    embeddings_df: pd.DataFrame, split_sequences: list[str], classes: list[str] = MLP_CLASSES,
    n: int = WINDOW_N, stride: int = STRIDE, gru_model_dir=None, hidden_size: int = HIDDEN_SIZE,
    num_layers: int = GRU_LAYERS, batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Runs the already-trained, frozen precompute GRU on one split's windows,
    returns its final hidden state h_t (not the MLP head's output) and labels.

    Pads fresh per mini-batch, never the whole split at once, and never puts the
    whole split on the GPU: at all-sensor scale a dense (n, m_max, dim) array for
    a 350k+-window split doesn't fit in plain CPU RAM, let alone the 6GB GPU
    directly, same class of bug fixed everywhere else in this module (see
    track_accumulation.md)."""
    if gru_model_dir is None:
        gru_model_dir = RNN_DIR / f"N{n}_stride{stride}_gru_h{hidden_size}"
    split_df = embeddings_df.loc[embeddings_df["sequence_name"].isin(split_sequences)]
    sequences, labels = build_windowed_embedding_sequences(split_df, classes, n, stride)
    dim = sequences[0].shape[1]

    model = DeepReflecsGRU(dim, hidden_size=hidden_size, num_layers=num_layers, num_classes=len(classes)).to(DEVICE)
    model.load_state_dict(torch.load(gru_model_dir / "gru_model.pt", map_location=DEVICE))
    model.eval()

    hidden_states = []
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch_seqs = sequences[start : start + batch_size]
            batch_m_max = max(len(s) for s in batch_seqs)
            batch_x, batch_lengths = pad_sequences(batch_seqs, batch_m_max, dim)
            batch_x_t = torch.tensor(batch_x, device=DEVICE)
            batch_lengths_t = torch.tensor(batch_lengths)
            packed = nn.utils.rnn.pack_padded_sequence(
                batch_x_t, batch_lengths_t, batch_first=True, enforce_sorted=False
            )
            _, h_n = model.gru(packed)
            hidden_states.append(h_n[-1])
    return torch.cat(hidden_states, dim=0).cpu().numpy(), labels


def _prepare_fusion_splits_core(
    embeddings_df: pd.DataFrame,
    splits: dict[str, list[str]],
    branch_fn,
    branch_name: str,
    classes: list[str],
    n: int,
    stride: int,
    gru_model_dir,
    gru_hidden_size: int,
    gru_num_layers: int,
    print_prefix: str = "",
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Shared core of every prepare_fusion_splits_* variant (pooled DeepReflecs, pooled
    MLP, raw histogram, ...): the two branches (a "pooled" embedding built by
    `branch_fn(split_sequences) -> (embedding, labels)`, and the GRU hidden state,
    always computed the same way) are built independently per split and concatenated.
    Alignment is verified explicitly (assert equal label arrays) rather than trusted
    from matching iteration order alone: both branches derive from the same underlying
    scan universe (same points table, same class filter, same n/stride), so identical
    labels in identical positions is the correct, checkable invariant. Every variant
    differs only in what `branch_fn` builds; this loop, the assertions, and the printed
    shape summary are otherwise identical across all of them."""
    results = {}
    for split_name in ("train", "val", "test"):
        seqs = splits[split_name]
        branch_embed, y_branch = branch_fn(seqs)
        gru_hidden, y_gru = compute_gru_hidden_states(
            embeddings_df, seqs, classes, n, stride, gru_model_dir, gru_hidden_size, gru_num_layers,
        )

        assert len(y_branch) == len(y_gru), f"{split_name}: window count mismatch, {len(y_branch)} vs {len(y_gru)}"
        assert np.array_equal(y_branch, y_gru), f"{split_name}: label mismatch, {branch_name}/GRU windows are misaligned"

        X = np.hstack([branch_embed, gru_hidden])
        results[split_name] = (X, y_branch)
        print(
            f"fusion{print_prefix} {split_name}: {X.shape[0]} windows, feature dim {X.shape[1]} "
            f"({branch_name} {branch_embed.shape[1]} + gru {gru_hidden.shape[1]})"
        )

    return results


def prepare_fusion_splits(
    df: pd.DataFrame, embeddings_df: pd.DataFrame, classes: list[str] = MLP_CLASSES, n: int = WINDOW_N,
    stride: int = STRIDE, splits: dict[str, list[str]] | None = None, range_sc_mode: str = "broadcast",
    pooled_encoder_dir=POOLED_ENCODER_DIR, gru_model_dir=None, gru_hidden_size: int = HIDDEN_SIZE,
    gru_num_layers: int = GRU_LAYERS,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Builds [pooled_embedding ; gru_hidden_state] per window for each split (pooled
    DeepReflecs branch, see _prepare_fusion_splits_core for the shared mechanics)."""
    if splits is None:
        splits = load_split()
    if gru_model_dir is None:
        gru_model_dir = RNN_DIR / f"N{n}_stride{stride}_gru_h{gru_hidden_size}"

    mean, std = fit_pooled_standardization(df, splits, classes, REFLECTION_FEATURES, n, stride, range_sc_mode)
    branch_fn = functools.partial(
        compute_pooled_embeddings, df, mean=mean, std=std, classes=classes, features=REFLECTION_FEATURES,
        n=n, stride=stride, range_sc_mode=range_sc_mode, encoder_dir=pooled_encoder_dir,
    )
    return _prepare_fusion_splits_core(
        embeddings_df, splits, branch_fn, "pooled", classes, n, stride, gru_model_dir, gru_hidden_size,
        gru_num_layers,
    )


def compute_pooled_mlp_embeddings(
    df: pd.DataFrame, split_sequences: list[str], edges: dict[str, np.ndarray], classes: list[str] = MLP_CLASSES,
    features: list[str] | None = None, n: int = WINDOW_N, stride: int = STRIDE, range_sc_mode: str = "raw",
    encoder_dir=None, n_bins: int = 16, include_doppler_spread: bool = False, hidden_dim: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """MLP-line counterpart to compute_pooled_embeddings: instead of running DeepReflecs'
    point-set conv network, builds the same quantile-bin histogram features the pooled
    MLP classifier was trained on (mlp_track_accumulation.build_windowed_histogram_
    features), then takes that trained MLP's penultimate hidden-layer activation
    (MLP.forward_features) as the learned embedding, the direct analogue of DeepReflecs'
    point_dim output. `edges` must be the quantile edges fit on this same encoder's own
    train split (fit_quantile_edges), reused as-is here, matching training-time exactly
    (same discipline as fit_reflection_standardization elsewhere in this module).
    range_sc_mode defaults to "raw", not "broadcast": the pooled MLP line was trained
    with range_sc_mode="raw" (mlp_track_accumulation.run_windowed_mlp's own default),
    unlike DeepReflecs' pooled point-set line which defaults to "broadcast"."""
    from mlp_track_accumulation import FEATURES as MLP_FEATURES, build_windowed_histogram_features

    if features is None:
        features = MLP_FEATURES

    split_df = df.loc[df["sequence_name"].isin(split_sequences)]
    point_sets, labels = build_pooled_point_sets(split_df, classes, features, n, stride, range_sc_mode)
    X, _ = build_windowed_histogram_features(point_sets, features, n_bins, edges=edges, include_doppler_spread=include_doppler_spread)

    model = MLP(input_dim=X.shape[1], hidden_dim=hidden_dim, num_classes=len(classes)).to(DEVICE)
    model.load_state_dict(torch.load(encoder_dir / "mlp_model.pt", map_location=DEVICE))
    model.eval()
    with torch.no_grad():
        embeddings = model.forward_features(torch.tensor(X, device=DEVICE)).cpu().numpy()
    return embeddings, labels


def prepare_fusion_splits_mlp(
    df: pd.DataFrame, embeddings_df: pd.DataFrame, mlp_encoder_dir, classes: list[str] = MLP_CLASSES,
    n: int = WINDOW_N, stride: int = STRIDE, splits: dict[str, list[str]] | None = None,
    range_sc_mode: str = "raw", gru_model_dir=None, gru_hidden_size: int = HIDDEN_SIZE,
    n_bins: int = 16, include_doppler_spread: bool = False,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Builds [pooled_mlp_embedding ; gru_hidden_state] per window for each split, the
    "MLP instead of DeepReflecs" counterpart to prepare_fusion_splits: same GRU branch,
    same alignment discipline, only the pooled branch's source model/features change."""
    from mlp_track_accumulation import FEATURES as MLP_FEATURES, fit_quantile_edges

    if splits is None:
        splits = load_split()
    if gru_model_dir is None:
        gru_model_dir = RNN_DIR / f"N{n}_stride{stride}_gru_h{gru_hidden_size}"

    train_df = df.loc[df["sequence_name"].isin(splits["train"])]
    train_point_sets, _ = build_pooled_point_sets(train_df, classes, MLP_FEATURES, n, stride, range_sc_mode)
    edges = fit_quantile_edges(train_point_sets, MLP_FEATURES, n_bins)
    del train_df, train_point_sets  # only needed for edge-fitting; each split rebuilds its own point sets below,
    # and at all-sensor N=50 scale leaving these bound for the rest of the function nearly doubles peak RSS
    # during the train split's own point-set build (see track_accumulation.md's OOM writeup)

    branch_fn = functools.partial(
        compute_pooled_mlp_embeddings, df, edges=edges, classes=classes, features=MLP_FEATURES, n=n, stride=stride,
        range_sc_mode=range_sc_mode, encoder_dir=mlp_encoder_dir, n_bins=n_bins,
        include_doppler_spread=include_doppler_spread,
    )
    return _prepare_fusion_splits_core(
        embeddings_df, splits, branch_fn, "mlp", classes, n, stride, gru_model_dir, gru_hidden_size,
        GRU_LAYERS, print_prefix=" (mlp)",
    )


def train_fusion_mlp(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, classes: list[str] = MLP_CLASSES,
    epochs: int = EPOCHS, batch_size: int = BATCH_SIZE, lr: float = LEARNING_RATE, random_state: int = RANDOM_STATE,
    hidden_dim: int = FUSION_MLP_HIDDEN_DIM, n_hidden_layers: int = 2,
):
    """Same Adam + class-count-weighted cross-entropy convention as every other
    trainer in this module. Plain MLP.MLP on the concatenated feature vector, no
    masking/packing needed, both branches already reduced each window to one fixed
    vector."""
    torch.manual_seed(random_state)

    weights_by_class = class_weights(pd.Series([classes[i] for i in y_train]))
    weight_tensor = torch.tensor([weights_by_class[cls] for cls in classes], dtype=torch.float32, device=DEVICE)
    print(f"class weights: {dict(zip(classes, weight_tensor.tolist()))}")

    model = MLP(
        input_dim=X_train.shape[1], hidden_dim=hidden_dim, num_classes=len(classes), n_hidden_layers=n_hidden_layers,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    X_train_t = torch.tensor(X_train, device=DEVICE)
    y_train_t = torch.tensor(y_train, device=DEVICE)
    X_val_t = torch.tensor(X_val, device=DEVICE)
    y_val_t = torch.tensor(y_val, device=DEVICE)

    n = len(X_train_t)
    history = []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        epoch_loss, epoch_correct = 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            batch_x, batch_y = X_train_t[idx], y_train_t[idx]

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(idx)
            epoch_correct += (logits.argmax(dim=1) == batch_y).sum().item()

        train_loss = epoch_loss / n
        train_acc = epoch_correct / n

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
        val_acc = (val_logits.argmax(dim=1) == y_val_t).float().mean().item()

        history.append({"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc, "val_acc": val_acc})
        print(f"epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

    return model, history


def _run_fusion_training_core(
    splits_data: dict[str, tuple[np.ndarray, np.ndarray]],
    classes: list[str],
    epochs: int,
    batch_size: int,
    lr: float,
    random_state: int,
    hidden_dim: int,
    n_hidden_layers: int,
    output_dir,
):
    """Shared tail of every run_fusion_training_* variant: train the fusion head on
    whatever splits_data a prepare_fusion_splits_* variant produced, save, plot, and
    return. Identical across variants; only how splits_data was built differs."""
    X_train, y_train = splits_data["train"]
    X_val, y_val = splits_data["val"]
    X_test, y_test = splits_data["test"]

    model, history = train_fusion_mlp(
        X_train, y_train, X_val, y_val, classes, epochs, batch_size, lr, random_state, hidden_dim, n_hidden_layers,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "fusion_model.pt")
    print(f"Saved {output_dir / 'fusion_model.pt'}")
    plot_training_curves(history, output_dir=output_dir)
    return model, history, X_test, y_test


def run_fusion_training(
    df: pd.DataFrame, embeddings_df: pd.DataFrame, n: int = WINDOW_N, stride: int = STRIDE,
    classes: list[str] = MLP_CLASSES, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE, lr: float = LEARNING_RATE,
    random_state: int = RANDOM_STATE, hidden_dim: int = FUSION_MLP_HIDDEN_DIM, n_hidden_layers: int = 2,
    pooled_encoder_dir=POOLED_ENCODER_DIR, gru_model_dir=None, gru_hidden_size: int = HIDDEN_SIZE,
    gru_num_layers: int = GRU_LAYERS, output_dir=None, splits: dict[str, list[str]] | None = None,
):
    if gru_model_dir is None:
        gru_model_dir = RNN_DIR / f"N{n}_stride{stride}_gru_h{gru_hidden_size}"
    if output_dir is None:
        output_dir = RNN_DIR / f"N{n}_stride{stride}_fusion_pooled_gru{gru_hidden_size}"
    if splits is None:
        splits = load_split()

    splits_data = prepare_fusion_splits(
        df, embeddings_df, classes, n, stride, splits,
        pooled_encoder_dir=pooled_encoder_dir, gru_model_dir=gru_model_dir, gru_hidden_size=gru_hidden_size,
        gru_num_layers=gru_num_layers,
    )
    return _run_fusion_training_core(
        splits_data, classes, epochs, batch_size, lr, random_state, hidden_dim, n_hidden_layers, output_dir,
    )


def run_fusion_training_mlp(
    df: pd.DataFrame, embeddings_df: pd.DataFrame, mlp_encoder_dir, n: int = WINDOW_N, stride: int = STRIDE,
    classes: list[str] = MLP_CLASSES, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE, lr: float = LEARNING_RATE,
    random_state: int = RANDOM_STATE, hidden_dim: int = FUSION_MLP_HIDDEN_DIM, n_hidden_layers: int = 2,
    gru_model_dir=None, gru_hidden_size: int = HIDDEN_SIZE, output_dir=None, splits: dict[str, list[str]] | None = None,
    range_sc_mode: str = "raw", n_bins: int = 16, include_doppler_spread: bool = False,
):
    """"MLP instead of DeepReflecs" fusion: pooled branch comes from the histogram
    MLP's own hidden-layer embedding (compute_pooled_mlp_embeddings) instead of
    DeepReflecs' point-set network, same GRU branch and same fusion-head training as
    run_fusion_training otherwise."""
    if gru_model_dir is None:
        gru_model_dir = RNN_DIR / f"N{n}_stride{stride}_gru_h{gru_hidden_size}"
    if output_dir is None:
        output_dir = RNN_DIR / f"N{n}_stride{stride}_fusion_mlp_gru{gru_hidden_size}"
    if splits is None:
        splits = load_split()

    splits_data = prepare_fusion_splits_mlp(
        df, embeddings_df, mlp_encoder_dir, classes, n, stride, splits,
        range_sc_mode=range_sc_mode, gru_model_dir=gru_model_dir, gru_hidden_size=gru_hidden_size,
        n_bins=n_bins, include_doppler_spread=include_doppler_spread,
    )
    return _run_fusion_training_core(
        splits_data, classes, epochs, batch_size, lr, random_state, hidden_dim, n_hidden_layers, output_dir,
    )


def compute_pooled_histogram_features(
    df: pd.DataFrame, split_sequences: list[str], edges: dict[str, np.ndarray], classes: list[str] = MLP_CLASSES,
    features: list[str] | None = None, n: int = WINDOW_N, stride: int = STRIDE, range_sc_mode: str = "raw",
    n_bins: int = 16, include_doppler_spread: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Raw-feature counterpart to compute_pooled_mlp_embeddings: the fixed quantile-bin
    histogram vector itself (no MLP model, no learned encoding step at all), used
    directly as the pooled branch. Isolates whether compute_pooled_mlp_embeddings'
    gain (if any) comes from the MLP's learned hidden layer specifically, or is
    already present in the raw hand-engineered feature vector it was computed from."""
    from mlp_track_accumulation import FEATURES as MLP_FEATURES, build_windowed_histogram_features

    if features is None:
        features = MLP_FEATURES

    split_df = df.loc[df["sequence_name"].isin(split_sequences)]
    point_sets, labels = build_pooled_point_sets(split_df, classes, features, n, stride, range_sc_mode)
    X, _ = build_windowed_histogram_features(point_sets, features, n_bins, edges=edges, include_doppler_spread=include_doppler_spread)
    return X, labels


def prepare_fusion_splits_histogram(
    df: pd.DataFrame, embeddings_df: pd.DataFrame, classes: list[str] = MLP_CLASSES, n: int = WINDOW_N,
    stride: int = STRIDE, splits: dict[str, list[str]] | None = None, range_sc_mode: str = "raw",
    gru_model_dir=None, gru_hidden_size: int = HIDDEN_SIZE, n_bins: int = 16, include_doppler_spread: bool = False,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Builds [raw_histogram_features ; gru_hidden_state] per window for each split,
    same alignment discipline as prepare_fusion_splits/prepare_fusion_splits_mlp, no
    MLP model in the pooled branch at all."""
    from mlp_track_accumulation import FEATURES as MLP_FEATURES, fit_quantile_edges

    if splits is None:
        splits = load_split()
    if gru_model_dir is None:
        gru_model_dir = RNN_DIR / f"N{n}_stride{stride}_gru_h{gru_hidden_size}"

    train_df = df.loc[df["sequence_name"].isin(splits["train"])]
    train_point_sets, _ = build_pooled_point_sets(train_df, classes, MLP_FEATURES, n, stride, range_sc_mode)
    edges = fit_quantile_edges(train_point_sets, MLP_FEATURES, n_bins)
    del train_df, train_point_sets  # only needed for edge-fitting; see prepare_fusion_splits_mlp's comment

    branch_fn = functools.partial(
        compute_pooled_histogram_features, df, edges=edges, classes=classes, features=MLP_FEATURES, n=n,
        stride=stride, range_sc_mode=range_sc_mode, n_bins=n_bins, include_doppler_spread=include_doppler_spread,
    )
    return _prepare_fusion_splits_core(
        embeddings_df, splits, branch_fn, "histogram", classes, n, stride, gru_model_dir, gru_hidden_size,
        GRU_LAYERS, print_prefix=" (histogram)",
    )


def run_fusion_training_histogram(
    df: pd.DataFrame, embeddings_df: pd.DataFrame, n: int = WINDOW_N, stride: int = STRIDE,
    classes: list[str] = MLP_CLASSES, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE, lr: float = LEARNING_RATE,
    random_state: int = RANDOM_STATE, hidden_dim: int = FUSION_MLP_HIDDEN_DIM, n_hidden_layers: int = 2,
    gru_model_dir=None, gru_hidden_size: int = HIDDEN_SIZE, output_dir=None, splits: dict[str, list[str]] | None = None,
    range_sc_mode: str = "raw", n_bins: int = 16, include_doppler_spread: bool = False,
):
    """"Raw quantile-bin histogram instead of any learned encoder" fusion: pooled
    branch is the fixed hand-engineered feature vector itself, same GRU branch and
    same fusion-head training as run_fusion_training/run_fusion_training_mlp
    otherwise."""
    if gru_model_dir is None:
        gru_model_dir = RNN_DIR / f"N{n}_stride{stride}_gru_h{gru_hidden_size}"
    if output_dir is None:
        output_dir = RNN_DIR / f"N{n}_stride{stride}_fusion_histogram_gru{gru_hidden_size}"
    if splits is None:
        splits = load_split()

    splits_data = prepare_fusion_splits_histogram(
        df, embeddings_df, classes, n, stride, splits, range_sc_mode=range_sc_mode, gru_model_dir=gru_model_dir,
        gru_hidden_size=gru_hidden_size, n_bins=n_bins, include_doppler_spread=include_doppler_spread,
    )
    return _run_fusion_training_core(
        splits_data, classes, epochs, batch_size, lr, random_state, hidden_dim, n_hidden_layers, output_dir,
    )


def evaluate_fusion_test_metrics(
    model: MLP, X_test: np.ndarray, y_test: np.ndarray, classes: list[str] = MLP_CLASSES, output_dir=None,
    n: int = WINDOW_N, stride: int = STRIDE, gru_hidden_size: int = HIDDEN_SIZE,
):
    if output_dir is None:
        output_dir = RNN_DIR / f"N{n}_stride{stride}_fusion_pooled_gru{gru_hidden_size}"

    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_recall_fscore_support

    model.eval()
    with torch.no_grad():
        y_pred = model(torch.tensor(X_test, device=DEVICE)).argmax(dim=1).cpu().numpy()

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=range(len(classes)), zero_division=0
    )
    metrics_df = pd.DataFrame({"precision": precision, "recall": recall, "f1": f1, "support": support}, index=classes)
    print(f"per-class precision/recall/f1 (test, fusion pooled+gru{gru_hidden_size}):")
    print(metrics_df.round(3).to_string())

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_json(output_dir / "fusion_test_metrics.json", orient="index", indent=2)
    print(f"Saved {output_dir / 'fusion_test_metrics.json'}")

    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_test, y_pred, labels=range(len(classes)), normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, colorbar=False, values_format=".2f")
    ax.set_title(f"Fusion (pooled DeepReflecs + GRU h={gru_hidden_size}): test confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    confusion_matrix_path = output_dir / "fusion_test_confusion_matrix.png"
    fig.savefig(confusion_matrix_path, dpi=150)
    print(f"Saved {confusion_matrix_path}")

    return metrics_df


def prepare_fusion_splits_mlp_e2e(
    df: pd.DataFrame, embeddings_df: pd.DataFrame, classes: list[str] = MLP_CLASSES, n: int = WINDOW_N,
    stride: int = STRIDE, splits: dict[str, list[str]] | None = None, range_sc_mode: str = "raw",
    gru_model_dir=None, gru_hidden_size: int = HIDDEN_SIZE, n_bins: int = 16, include_doppler_spread: bool = False,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Like prepare_fusion_splits_histogram, but keeps the raw histogram features and
    GRU hidden states as two separate arrays per split instead of concatenating them:
    the MLP encoder is about to become a trainable sub-module (see
    FusionEndToEndMLP/train_fusion_mlp_e2e below), fed the raw features directly, not
    a precomputed fixed embedding. The GRU branch stays frozen/precomputed exactly as
    in every other fusion variant, only the MLP branch becomes end-to-end trainable,
    mirroring how "End-to-end GRU, warmstart" only unfroze the single-scan encoder
    while the rest of that pipeline stayed as-is."""
    from mlp_track_accumulation import FEATURES as MLP_FEATURES, fit_quantile_edges

    if splits is None:
        splits = load_split()
    if gru_model_dir is None:
        gru_model_dir = RNN_DIR / f"N{n}_stride{stride}_gru_h{gru_hidden_size}"

    train_df = df.loc[df["sequence_name"].isin(splits["train"])]
    train_point_sets, _ = build_pooled_point_sets(train_df, classes, MLP_FEATURES, n, stride, range_sc_mode)
    edges = fit_quantile_edges(train_point_sets, MLP_FEATURES, n_bins)
    del train_df, train_point_sets  # only needed for edge-fitting; see prepare_fusion_splits_mlp's comment

    results = {}
    for split_name in ("train", "val", "test"):
        seqs = splits[split_name]
        hist_features, y_hist = compute_pooled_histogram_features(
            df, seqs, edges, classes, MLP_FEATURES, n, stride, range_sc_mode, n_bins, include_doppler_spread,
        )
        gru_hidden, y_gru = compute_gru_hidden_states(embeddings_df, seqs, classes, n, stride, gru_model_dir, gru_hidden_size)

        assert len(y_hist) == len(y_gru), f"{split_name}: window count mismatch, {len(y_hist)} vs {len(y_gru)}"
        assert np.array_equal(y_hist, y_gru), f"{split_name}: label mismatch, histogram/GRU windows are misaligned"

        results[split_name] = (hist_features, gru_hidden, y_hist)
        print(
            f"fusion (mlp e2e) {split_name}: {len(y_hist)} windows, "
            f"histogram dim {hist_features.shape[1]}, gru dim {gru_hidden.shape[1]}"
        )

    return results


class FusionEndToEndMLP(nn.Module):
    """The MLP encoder (its penultimate hidden layer, via forward_features) is a
    trainable sub-module here, not a frozen precompute step: gradients flow from the
    fusion head back through it. The GRU branch is not part of this module at all,
    its hidden state arrives already fixed/precomputed, same as every other fusion
    variant."""

    def __init__(
        self, hist_dim: int, gru_dim: int, mlp_hidden_dim: int = 16, fusion_hidden_dim: int = FUSION_MLP_HIDDEN_DIM,
        n_hidden_layers: int = 2, num_classes: int = len(MLP_CLASSES),
    ):
        super().__init__()
        self.mlp_encoder = MLP(input_dim=hist_dim, hidden_dim=mlp_hidden_dim, num_classes=num_classes)
        self.head = MLP(
            input_dim=mlp_hidden_dim + gru_dim, hidden_dim=fusion_hidden_dim, num_classes=num_classes,
            n_hidden_layers=n_hidden_layers,
        )

    def forward(self, x_hist: torch.Tensor, gru_hidden: torch.Tensor) -> torch.Tensor:
        mlp_embed = self.mlp_encoder.forward_features(x_hist)
        combined = torch.cat([mlp_embed, gru_hidden], dim=-1)
        return self.head(combined)


def train_fusion_mlp_e2e(
    hist_train: np.ndarray, gru_train: np.ndarray, y_train: np.ndarray,
    hist_val: np.ndarray, gru_val: np.ndarray, y_val: np.ndarray, mlp_encoder_dir,
    classes: list[str] = MLP_CLASSES, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE, lr: float = LEARNING_RATE,
    random_state: int = RANDOM_STATE, mlp_hidden_dim: int = 16, hidden_dim: int = FUSION_MLP_HIDDEN_DIM,
    n_hidden_layers: int = 2, warmstart: bool = True,
):
    """Same Adam + class-weighted CE convention as every other trainer here.
    warmstart=True (default, matching "End-to-end GRU, warmstart" beating its
    random-init counterpart) loads the already-trained MLP classifier's weights into
    model.mlp_encoder before training; warmstart=False leaves it randomly
    initialized, for the same random-init control this branch already ran for the
    GRU line."""
    torch.manual_seed(random_state)

    weights_by_class = class_weights(pd.Series([classes[i] for i in y_train]))
    weight_tensor = torch.tensor([weights_by_class[cls] for cls in classes], dtype=torch.float32, device=DEVICE)
    print(f"class weights: {dict(zip(classes, weight_tensor.tolist()))}")

    model = FusionEndToEndMLP(
        hist_train.shape[1], gru_train.shape[1], mlp_hidden_dim, hidden_dim, n_hidden_layers, len(classes),
    ).to(DEVICE)
    if warmstart:
        model.mlp_encoder.load_state_dict(torch.load(mlp_encoder_dir / "mlp_model.pt", map_location=DEVICE))
        print(f"warmstarted mlp_encoder from {mlp_encoder_dir / 'mlp_model.pt'}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    hist_train_t = torch.tensor(hist_train, device=DEVICE)
    gru_train_t = torch.tensor(gru_train, device=DEVICE)
    y_train_t = torch.tensor(y_train, device=DEVICE)
    hist_val_t = torch.tensor(hist_val, device=DEVICE)
    gru_val_t = torch.tensor(gru_val, device=DEVICE)
    y_val_t = torch.tensor(y_val, device=DEVICE)

    import resource
    print(f"[DEBUG] after tensors on device: peak RSS = {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.3f} GB", flush=True)

    n = len(hist_train_t)
    history = []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        epoch_loss, epoch_correct = 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            batch_hist, batch_gru, batch_y = hist_train_t[idx], gru_train_t[idx], y_train_t[idx]

            optimizer.zero_grad()
            logits = model(batch_hist, batch_gru)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(idx)
            epoch_correct += (logits.argmax(dim=1) == batch_y).sum().item()
            if epoch == 0 and start == 0:
                print(f"[DEBUG] after first batch: peak RSS = {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.3f} GB", flush=True)

        train_loss = epoch_loss / n
        train_acc = epoch_correct / n

        model.eval()
        with torch.no_grad():
            val_logits = model(hist_val_t, gru_val_t)
        val_acc = (val_logits.argmax(dim=1) == y_val_t).float().mean().item()

        history.append({"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc, "val_acc": val_acc})
        print(f"epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}", flush=True)

    return model, history


def run_fusion_training_mlp_e2e(
    df: pd.DataFrame, embeddings_df: pd.DataFrame, mlp_encoder_dir, n: int = WINDOW_N, stride: int = STRIDE,
    classes: list[str] = MLP_CLASSES, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE, lr: float = LEARNING_RATE,
    random_state: int = RANDOM_STATE, mlp_hidden_dim: int = 16, hidden_dim: int = FUSION_MLP_HIDDEN_DIM,
    n_hidden_layers: int = 2, gru_model_dir=None, gru_hidden_size: int = HIDDEN_SIZE, output_dir=None,
    splits: dict[str, list[str]] | None = None, range_sc_mode: str = "raw", n_bins: int = 16,
    include_doppler_spread: bool = False, warmstart: bool = True,
):
    if gru_model_dir is None:
        gru_model_dir = RNN_DIR / f"N{n}_stride{stride}_gru_h{gru_hidden_size}"
    if output_dir is None:
        tag = "warmstart" if warmstart else "random"
        output_dir = RNN_DIR / f"N{n}_stride{stride}_fusion_mlp_e2e_{tag}_gru{gru_hidden_size}"
    if splits is None:
        splits = load_split()

    splits_data = prepare_fusion_splits_mlp_e2e(
        df, embeddings_df, classes, n, stride, splits, range_sc_mode=range_sc_mode, gru_model_dir=gru_model_dir,
        gru_hidden_size=gru_hidden_size, n_bins=n_bins, include_doppler_spread=include_doppler_spread,
    )
    hist_train, gru_train, y_train = splits_data["train"]
    hist_val, gru_val, y_val = splits_data["val"]
    hist_test, gru_test, y_test = splits_data["test"]

    model, history = train_fusion_mlp_e2e(
        hist_train, gru_train, y_train, hist_val, gru_val, y_val, mlp_encoder_dir, classes, epochs, batch_size, lr,
        random_state, mlp_hidden_dim, hidden_dim, n_hidden_layers, warmstart,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "fusion_model.pt")
    print(f"Saved {output_dir / 'fusion_model.pt'}")
    plot_training_curves(history, output_dir=output_dir)
    return model, history, hist_test, gru_test, y_test


def evaluate_fusion_mlp_e2e_test_metrics(
    model: FusionEndToEndMLP, hist_test: np.ndarray, gru_test: np.ndarray, y_test: np.ndarray,
    classes: list[str] = MLP_CLASSES, output_dir=None, n: int = WINDOW_N, stride: int = STRIDE,
    gru_hidden_size: int = HIDDEN_SIZE, warmstart: bool = True,
):
    if output_dir is None:
        tag = "warmstart" if warmstart else "random"
        output_dir = RNN_DIR / f"N{n}_stride{stride}_fusion_mlp_e2e_{tag}_gru{gru_hidden_size}"

    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_recall_fscore_support

    model.eval()
    with torch.no_grad():
        y_pred = model(
            torch.tensor(hist_test, device=DEVICE), torch.tensor(gru_test, device=DEVICE)
        ).argmax(dim=1).cpu().numpy()

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=range(len(classes)), zero_division=0
    )
    metrics_df = pd.DataFrame({"precision": precision, "recall": recall, "f1": f1, "support": support}, index=classes)
    print(f"per-class precision/recall/f1 (test, fusion mlp-e2e ({'warmstart' if warmstart else 'random'})+gru{gru_hidden_size}):")
    print(metrics_df.round(3).to_string())

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_json(output_dir / "fusion_test_metrics.json", orient="index", indent=2)
    print(f"Saved {output_dir / 'fusion_test_metrics.json'}")

    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_test, y_pred, labels=range(len(classes)), normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, colorbar=False, values_format=".2f")
    ax.set_title(f"Fusion (MLP end-to-end + GRU h={gru_hidden_size}): test confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    confusion_matrix_path = output_dir / "fusion_test_confusion_matrix.png"
    fig.savefig(confusion_matrix_path, dpi=150)
    print(f"Saved {confusion_matrix_path}")

    return metrics_df


# --- temporal-variation scalar, appended to the GRU/transformer's own summary ---
# (see track_accumulation.md discussion): the cheap hand-built summed-diff statistic
# concatenated onto the sequence model's final representation, sidestepping whatever
# the frozen N=1 encoder's classification bottleneck might have discarded about raw
# feature dynamics. Two scalars (rcs, vr_compensated), not a whole extra branch.
from mlp_track_accumulation import TEMPORAL_FEATURES, build_windowed_temporal_features


def _build_windowed_labels(
    df: pd.DataFrame, classes: list[str] = MLP_CLASSES, n: int = WINDOW_N, stride: int = STRIDE,
) -> np.ndarray:
    """Cheap label-only mirror of build_windowed_point_sets' windowing (same
    TRACK_COLS + timestamp sort, same per-track stride slicing, same "target scan's
    own class" label rule), without building any pooled point features. Used only to
    verify window count/order alignment against another branch's own windowing (see
    compute_temporal_variation_for_split): building the full point sets just to
    discard them and keep the labels would, at all-sensor N=50 scale, rebuild
    exactly the large pooled-point-set structure that already OOM-killed this
    branch once (see track_accumulation.md) purely to throw it away."""
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    mask = df["group"].isin(classes)
    filtered = df.loc[mask]
    class_idx = filtered["group"].map(class_to_idx).to_numpy()

    scan_positions = filtered.groupby(INSTANCE_COLS, sort=False).indices
    scan_arrays = list(scan_positions.values())
    scan_keys = pd.DataFrame(list(scan_positions.keys()), columns=INSTANCE_COLS)
    scan_keys["_scan_idx"] = np.arange(len(scan_keys))
    scan_keys = scan_keys.sort_values(TRACK_COLS + ["timestamp"])

    labels = []
    for _, track_scans in scan_keys.groupby(TRACK_COLS, sort=False):
        ordered = track_scans["_scan_idx"].to_numpy()
        for i in range(0, len(ordered), stride):
            target_positions = scan_arrays[ordered[i]]
            labels.append(class_idx[target_positions[0]])
    return np.array(labels, dtype="int64")


def compute_temporal_variation_for_split(
    df: pd.DataFrame, split_sequences: list[str], y_reference: np.ndarray, classes: list[str] = MLP_CLASSES,
    temporal_features: list[str] = TEMPORAL_FEATURES, n: int = WINDOW_N, stride: int = STRIDE,
) -> np.ndarray:
    """build_windowed_temporal_features for one split, verified aligned against
    y_reference (that split's labels from the GRU/transformer embedding-sequence
    branch) before trusting the two can be concatenated positionally: both are
    documented/established to iterate the same underlying scan universe in the same
    order, but this is exactly the kind of silent-misalignment risk worth checking
    rather than assuming (see the fusion variant's same discipline above)."""
    split_df = df.loc[df["sequence_name"].isin(split_sequences)]
    temporal = build_windowed_temporal_features(split_df, classes, temporal_features, n, stride).astype("float32")
    y_check = _build_windowed_labels(split_df, classes, n, stride)
    assert len(y_check) == len(y_reference) == len(temporal), "window count mismatch building temporal variation"
    assert np.array_equal(y_check, y_reference), "temporal variation windows misaligned with embedding-sequence windows"
    return temporal


def prepare_windowed_embedding_splits_with_temporal(
    df: pd.DataFrame, embeddings_df: pd.DataFrame, classes: list[str] = MLP_CLASSES, n: int = WINDOW_N,
    stride: int = STRIDE, splits: dict[str, list[str]] | None = None,
):
    if splits is None:
        splits = load_split()
    train_seqs, y_train, val_seqs, y_val, test_seqs, y_test = prepare_windowed_embedding_splits(
        embeddings_df, classes=classes, n=n, stride=stride, splits=splits,
    )
    temporal_train = compute_temporal_variation_for_split(df, splits["train"], y_train, classes, n=n, stride=stride)
    temporal_val = compute_temporal_variation_for_split(df, splits["val"], y_val, classes, n=n, stride=stride)
    temporal_test = compute_temporal_variation_for_split(df, splits["test"], y_test, classes, n=n, stride=stride)
    return (
        train_seqs, temporal_train, y_train,
        val_seqs, temporal_val, y_val,
        test_seqs, temporal_test, y_test,
    )


class DeepReflecsGRUWithTemporal(nn.Module):
    """DeepReflecsGRU with the temporal-variation scalars concatenated onto the
    final hidden state before the MLP head."""

    def __init__(
        self, embedding_dim: int, temporal_dim: int, hidden_size: int = HIDDEN_SIZE, num_layers: int = GRU_LAYERS,
        num_classes: int = len(MLP_CLASSES), mlp_hidden_dim: int = MLP_HIDDEN_DIM,
    ):
        super().__init__()
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = MLP(
            input_dim=hidden_size + temporal_dim, hidden_dim=mlp_hidden_dim, num_classes=num_classes, n_hidden_layers=1,
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, temporal: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        combined = torch.cat([h_n[-1], temporal], dim=-1)
        return self.head(combined)


class DeepReflecsTransformerWithTemporal(nn.Module):
    """DeepReflecsTransformer with the temporal-variation scalars concatenated onto
    the final-position output before the MLP head."""

    def __init__(
        self,
        embedding_dim: int,
        max_seq_len: int,
        temporal_dim: int,
        d_model: int = D_MODEL,
        num_heads: int = NUM_HEADS,
        num_layers: int = NUM_LAYERS,
        dim_feedforward: int = DIM_FEEDFORWARD,
        num_classes: int = len(MLP_CLASSES),
        mlp_hidden_dim: int = MLP_HIDDEN_DIM,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_proj = nn.Identity() if embedding_dim == d_model else nn.Linear(embedding_dim, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = MLP(
            input_dim=d_model + temporal_dim, hidden_dim=mlp_hidden_dim, num_classes=num_classes, n_hidden_layers=1,
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, temporal: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.input_proj(x) + self.pos_embedding(positions)

        causal_mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        lengths = lengths.to(x.device)
        key_padding_mask = torch.arange(T, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)

        out = self.encoder(h, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        last_idx = lengths - 1
        last_hidden = out[torch.arange(B, device=x.device), last_idx]
        combined = torch.cat([last_hidden, temporal], dim=-1)
        return self.head(combined)


def _predict_with_temporal_in_batches(
    model, sequences: list[np.ndarray], temporal: torch.Tensor, batch_size: int = 1024,
) -> torch.Tensor:
    """sequences is a ragged list, padded fresh per mini-batch, same reason as
    _predict_in_batches. temporal is small (2 scalars/window) and stays a plain
    tensor, only moved to DEVICE per batch."""
    model.eval()
    dim = sequences[0].shape[1]
    logits = []
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            sl = slice(start, start + batch_size)
            batch_seqs = sequences[sl]
            batch_m_max = max(len(s) for s in batch_seqs)
            batch_x, batch_lengths = pad_sequences(batch_seqs, batch_m_max, dim)
            batch_x_t = torch.tensor(batch_x, device=DEVICE)
            batch_lengths_t = torch.tensor(batch_lengths)
            batch_temporal = temporal[sl].to(DEVICE)
            logits.append(model(batch_x_t, batch_lengths_t, batch_temporal))
    return torch.cat(logits, dim=0)


def _train_sequence_with_temporal(
    model_cls,
    model_kwargs: dict,
    train_seqs: list[np.ndarray], temporal_train: np.ndarray, y_train: np.ndarray,
    val_seqs: list[np.ndarray], temporal_val: np.ndarray, y_val: np.ndarray,
    classes: list[str] = MLP_CLASSES, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE, lr: float = LEARNING_RATE,
    random_state: int = RANDOM_STATE,
):
    """Shared training loop body for DeepReflecsGRUWithTemporal and
    DeepReflecsTransformerWithTemporal: identical Adam + class-weighted CE
    convention as every other trainer here, only the model class/kwargs differ.
    train_seqs/val_seqs are ragged lists, padded fresh per mini-batch (see
    _predict_with_temporal_in_batches); temporal_train/temporal_val are small dense
    arrays (2 scalars/window), kept on CPU and moved to DEVICE per batch."""
    torch.manual_seed(random_state)

    weights_by_class = class_weights(pd.Series([classes[i] for i in y_train]))
    weight_tensor = torch.tensor([weights_by_class[cls] for cls in classes], dtype=torch.float32, device=DEVICE)
    print(f"class weights: {dict(zip(classes, weight_tensor.tolist()))}")

    embedding_dim = train_seqs[0].shape[1]
    model = model_cls(**model_kwargs).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    temporal_train_t = torch.tensor(temporal_train)
    y_train_t = torch.tensor(y_train, device=DEVICE)
    temporal_val_t = torch.tensor(temporal_val)
    y_val_t = torch.tensor(y_val, device=DEVICE)

    n = len(train_seqs)
    history = []
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
            batch_temporal = temporal_train_t[idx].to(DEVICE)
            batch_y = y_train_t[idx]

            optimizer.zero_grad()
            logits = model(batch_x_t, batch_lengths_t, batch_temporal)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(idx)
            epoch_correct += (logits.argmax(dim=1) == batch_y).sum().item()

        train_loss = epoch_loss / n
        train_acc = epoch_correct / n

        val_logits = _predict_with_temporal_in_batches(model, val_seqs, temporal_val_t)
        val_acc = (val_logits.argmax(dim=1) == y_val_t).float().mean().item()

        history.append({"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc, "val_acc": val_acc})
        print(f"epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

    return model, history


def _evaluate_with_temporal_metrics(
    model, sequences, temporal, y_true, classes, metrics_cache, model_cache, confusion_matrix_path, title,
):
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_recall_fscore_support

    y_pred = _predict_with_temporal_in_batches(model, sequences, temporal).argmax(dim=1).cpu().numpy()

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(classes)), zero_division=0
    )
    metrics_df = pd.DataFrame({"precision": precision, "recall": recall, "f1": f1, "support": support}, index=classes)
    print(f"per-class precision/recall/f1 ({title}):")
    print(metrics_df.round(3).to_string())

    metrics_cache.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_json(metrics_cache, orient="index", indent=2)
    print(f"Saved {metrics_cache}")

    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)), normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, colorbar=False, values_format=".2f")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    confusion_matrix_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(confusion_matrix_path, dpi=150)
    print(f"Saved {confusion_matrix_path}")

    return metrics_df


def run_gru_with_temporal_training(
    df: pd.DataFrame, embeddings_df: pd.DataFrame, n: int = WINDOW_N, stride: int = STRIDE,
    classes: list[str] = MLP_CLASSES, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE, lr: float = LEARNING_RATE,
    random_state: int = RANDOM_STATE, hidden_size: int = HIDDEN_SIZE, num_layers: int = GRU_LAYERS,
    mlp_hidden_dim: int = MLP_HIDDEN_DIM, output_dir=None, splits: dict[str, list[str]] | None = None,
):
    if output_dir is None:
        output_dir = RNN_DIR / f"N{n}_stride{stride}_gru_h{hidden_size}_temporalvar"
    if splits is None:
        splits = load_split()

    (
        train_seqs, temporal_train, y_train,
        val_seqs, temporal_val, y_val,
        test_seqs, temporal_test, y_test,
    ) = prepare_windowed_embedding_splits_with_temporal(df, embeddings_df, classes, n, stride, splits)

    model, history = _train_sequence_with_temporal(
        DeepReflecsGRUWithTemporal,
        {
            "embedding_dim": train_seqs[0].shape[1], "temporal_dim": temporal_train.shape[1], "hidden_size": hidden_size,
            "num_layers": num_layers, "num_classes": len(classes), "mlp_hidden_dim": mlp_hidden_dim,
        },
        train_seqs, temporal_train, y_train, val_seqs, temporal_val, y_val,
        classes, epochs, batch_size, lr, random_state,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "gru_temporal_model.pt")
    print(f"Saved {output_dir / 'gru_temporal_model.pt'}")
    plot_training_curves(history, output_dir=output_dir)
    return model, history, test_seqs, temporal_test, y_test


def evaluate_gru_with_temporal_test_metrics(
    model, test_seqs, temporal_test, y_test, classes: list[str] = MLP_CLASSES, output_dir=None,
    n: int = WINDOW_N, stride: int = STRIDE, hidden_size: int = HIDDEN_SIZE,
):
    if output_dir is None:
        output_dir = RNN_DIR / f"N{n}_stride{stride}_gru_h{hidden_size}_temporalvar"
    temporal_t = torch.tensor(temporal_test)
    return _evaluate_with_temporal_metrics(
        model, test_seqs, temporal_t, y_test, classes,
        metrics_cache=output_dir / "gru_temporal_test_metrics.json",
        model_cache=output_dir / "gru_temporal_model.pt",
        confusion_matrix_path=output_dir / "gru_temporal_test_confusion_matrix.png",
        title=f"DeepReflecs+GRU+temporal variation: test confusion matrix (N={n}, stride={stride}, h={hidden_size})",
    )


def run_transformer_with_temporal_training(
    df: pd.DataFrame, embeddings_df: pd.DataFrame, n: int = WINDOW_N, stride: int = STRIDE,
    classes: list[str] = MLP_CLASSES, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE, lr: float = LEARNING_RATE,
    random_state: int = RANDOM_STATE, d_model: int = D_MODEL, num_heads: int = NUM_HEADS, num_layers: int = NUM_LAYERS,
    dim_feedforward: int = DIM_FEEDFORWARD, mlp_hidden_dim: int = MLP_HIDDEN_DIM, output_dir=None,
    splits: dict[str, list[str]] | None = None,
):
    if output_dir is None:
        output_dir = RNN_DIR / f"N{n}_stride{stride}_transformer_d{d_model}_temporalvar"
    if splits is None:
        splits = load_split()

    (
        train_seqs, temporal_train, y_train,
        val_seqs, temporal_val, y_val,
        test_seqs, temporal_test, y_test,
    ) = prepare_windowed_embedding_splits_with_temporal(df, embeddings_df, classes, n, stride, splits)

    model, history = _train_sequence_with_temporal(
        DeepReflecsTransformerWithTemporal,
        {
            "embedding_dim": train_seqs[0].shape[1], "max_seq_len": n, "temporal_dim": temporal_train.shape[1],
            "d_model": d_model, "num_heads": num_heads, "num_layers": num_layers, "dim_feedforward": dim_feedforward,
            "num_classes": len(classes), "mlp_hidden_dim": mlp_hidden_dim,
        },
        train_seqs, temporal_train, y_train, val_seqs, temporal_val, y_val,
        classes, epochs, batch_size, lr, random_state,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "transformer_temporal_model.pt")
    print(f"Saved {output_dir / 'transformer_temporal_model.pt'}")
    plot_training_curves(history, output_dir=output_dir)
    return model, history, test_seqs, temporal_test, y_test


def evaluate_transformer_with_temporal_test_metrics(
    model, test_seqs, temporal_test, y_test, classes: list[str] = MLP_CLASSES, output_dir=None,
    n: int = WINDOW_N, stride: int = STRIDE, d_model: int = D_MODEL,
):
    if output_dir is None:
        output_dir = RNN_DIR / f"N{n}_stride{stride}_transformer_d{d_model}_temporalvar"
    temporal_t = torch.tensor(temporal_test)
    return _evaluate_with_temporal_metrics(
        model, test_seqs, temporal_t, y_test, classes,
        metrics_cache=output_dir / "transformer_temporal_test_metrics.json",
        model_cache=output_dir / "transformer_temporal_model.pt",
        confusion_matrix_path=output_dir / "transformer_temporal_test_confusion_matrix.png",
        title=f"DeepReflecs+Transformer+temporal variation: test confusion matrix (N={n}, stride={stride}, d={d_model})",
    )


# --- orthogonalized three-way fusion: pooled DeepReflecs (with its temporal-
# predictable component regressed out) + GRU hidden state + temporal variation
# scalars, concatenated into one MLP head. The diagnostic probe below found partial
# overlap (R^2 0.33-0.39) between the pooled embedding and the temporal scalars, so
# a raw three-way concat would double-count some of that shared variance;
# orthogonalizing removes exactly the redundant part first, so any measured gain is
# guaranteed to come from information the temporal scalars don't already carry. ---
def orthogonalize_pooled_embedding(
    pooled_train: np.ndarray, temporal_train: np.ndarray, *eval_pairs: tuple[np.ndarray, np.ndarray],
) -> list[np.ndarray]:
    """Fits one linear regression predicting the pooled embedding from the temporal
    scalars on train only, returns residuals (pooled embedding minus its
    temporal-predictable component) for train and every (pooled, temporal) pair in
    eval_pairs, all using the train-fit regressor (same fit-on-train-only discipline
    as fit_pooled_standardization)."""
    from sklearn.linear_model import LinearRegression

    reg = LinearRegression()
    reg.fit(temporal_train, pooled_train)

    residuals = [pooled_train - reg.predict(temporal_train)]
    for pooled_eval, temporal_eval in eval_pairs:
        residuals.append(pooled_eval - reg.predict(temporal_eval))
    return residuals


def prepare_fusion_splits_with_temporal(
    df: pd.DataFrame, embeddings_df: pd.DataFrame, classes: list[str] = MLP_CLASSES, n: int = WINDOW_N,
    stride: int = STRIDE, splits: dict[str, list[str]] | None = None, range_sc_mode: str = "broadcast",
    pooled_encoder_dir=POOLED_ENCODER_DIR, gru_model_dir=None, gru_hidden_size: int = HIDDEN_SIZE,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Builds [orthogonalized_pooled_embedding ; gru_hidden_state ; temporal_variation]
    per window for each split. Pooled embedding is orthogonalized against the
    temporal scalars (fit on train) before concatenation. All three sources are
    verified aligned (assert equal label arrays) before concatenating, same
    discipline as prepare_fusion_splits."""
    if splits is None:
        splits = load_split()
    if gru_model_dir is None:
        gru_model_dir = RNN_DIR / f"N{n}_stride{stride}_gru_h{gru_hidden_size}"

    mean, std = fit_pooled_standardization(df, splits, classes, REFLECTION_FEATURES, n, stride, range_sc_mode)

    raw = {}
    for split_name in ("train", "val", "test"):
        seqs = splits[split_name]
        pooled_embed, y_pooled = compute_pooled_embeddings(
            df, seqs, mean, std, classes, REFLECTION_FEATURES, n, stride, range_sc_mode, pooled_encoder_dir,
        )
        gru_hidden, y_gru = compute_gru_hidden_states(embeddings_df, seqs, classes, n, stride, gru_model_dir, gru_hidden_size)
        temporal = compute_temporal_variation_for_split(df, seqs, y_pooled, classes, n=n, stride=stride)

        assert len(y_pooled) == len(y_gru), f"{split_name}: window count mismatch, {len(y_pooled)} vs {len(y_gru)}"
        assert np.array_equal(y_pooled, y_gru), f"{split_name}: label mismatch, pooled/GRU windows are misaligned"

        raw[split_name] = {"pooled": pooled_embed, "gru": gru_hidden, "temporal": temporal, "y": y_pooled}

    pooled_train_resid, pooled_val_resid, pooled_test_resid = orthogonalize_pooled_embedding(
        raw["train"]["pooled"], raw["train"]["temporal"],
        (raw["val"]["pooled"], raw["val"]["temporal"]),
        (raw["test"]["pooled"], raw["test"]["temporal"]),
    )
    residuals = {"train": pooled_train_resid, "val": pooled_val_resid, "test": pooled_test_resid}

    results = {}
    for split_name in ("train", "val", "test"):
        X = np.hstack([residuals[split_name], raw[split_name]["gru"], raw[split_name]["temporal"]])
        results[split_name] = (X, raw[split_name]["y"])
        print(
            f"fusion+temporal (orthogonalized) {split_name}: {X.shape[0]} windows, feature dim {X.shape[1]} "
            f"(pooled residual {residuals[split_name].shape[1]} + gru {raw[split_name]['gru'].shape[1]} "
            f"+ temporal {raw[split_name]['temporal'].shape[1]})"
        )
    return results


def run_fusion_with_temporal_training(
    df: pd.DataFrame, embeddings_df: pd.DataFrame, n: int = WINDOW_N, stride: int = STRIDE,
    classes: list[str] = MLP_CLASSES, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE, lr: float = LEARNING_RATE,
    random_state: int = RANDOM_STATE, hidden_dim: int = FUSION_MLP_HIDDEN_DIM, n_hidden_layers: int = 2,
    pooled_encoder_dir=POOLED_ENCODER_DIR, gru_model_dir=None, gru_hidden_size: int = HIDDEN_SIZE,
    output_dir=None, splits: dict[str, list[str]] | None = None,
):
    if gru_model_dir is None:
        gru_model_dir = RNN_DIR / f"N{n}_stride{stride}_gru_h{gru_hidden_size}"
    if output_dir is None:
        output_dir = RNN_DIR / f"N{n}_stride{stride}_fusion_orthogonalized_temporalvar"
    if splits is None:
        splits = load_split()

    splits_data = prepare_fusion_splits_with_temporal(
        df, embeddings_df, classes, n, stride, splits,
        pooled_encoder_dir=pooled_encoder_dir, gru_model_dir=gru_model_dir, gru_hidden_size=gru_hidden_size,
    )
    X_train, y_train = splits_data["train"]
    X_val, y_val = splits_data["val"]
    X_test, y_test = splits_data["test"]

    model, history = train_fusion_mlp(
        X_train, y_train, X_val, y_val, classes, epochs, batch_size, lr, random_state, hidden_dim, n_hidden_layers,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "fusion_temporal_model.pt")
    print(f"Saved {output_dir / 'fusion_temporal_model.pt'}")
    plot_training_curves(history, output_dir=output_dir)
    return model, history, X_test, y_test


def evaluate_fusion_with_temporal_test_metrics(
    model: MLP, X_test: np.ndarray, y_test: np.ndarray, classes: list[str] = MLP_CLASSES, output_dir=None,
    n: int = WINDOW_N, stride: int = STRIDE, gru_hidden_size: int = HIDDEN_SIZE,
):
    if output_dir is None:
        output_dir = RNN_DIR / f"N{n}_stride{stride}_fusion_orthogonalized_temporalvar"

    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_recall_fscore_support

    model.eval()
    with torch.no_grad():
        y_pred = model(torch.tensor(X_test, device=DEVICE)).argmax(dim=1).cpu().numpy()

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=range(len(classes)), zero_division=0
    )
    metrics_df = pd.DataFrame({"precision": precision, "recall": recall, "f1": f1, "support": support}, index=classes)
    print(f"per-class precision/recall/f1 (test, orthogonalized fusion+temporal, gru{gru_hidden_size}):")
    print(metrics_df.round(3).to_string())

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_json(output_dir / "fusion_temporal_test_metrics.json", orient="index", indent=2)
    print(f"Saved {output_dir / 'fusion_temporal_test_metrics.json'}")

    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_test, y_pred, labels=range(len(classes)), normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, colorbar=False, values_format=".2f")
    ax.set_title(f"Orthogonalized fusion + temporal variation (GRU h={gru_hidden_size}): test confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    confusion_matrix_path = output_dir / "fusion_temporal_test_confusion_matrix.png"
    fig.savefig(confusion_matrix_path, dpi=150)
    print(f"Saved {confusion_matrix_path}")

    return metrics_df


# --- diagnostic: how much of the temporal variation scalars is already recoverable
# linearly from the pooled DeepReflecs embedding? Answers whether the fusion skip
# and the temporal-variation addition (both +0.006/+0.013 over the plain GRU) are
# independent signals or two patches for the same gap, before spending a training
# run on a three-way fusion. See track_accumulation.md's "Design note" section. ---
def probe_temporal_from_pooled_embedding(
    df: pd.DataFrame, splits: dict[str, list[str]] | None = None, classes: list[str] = MLP_CLASSES,
    n: int = WINDOW_N, stride: int = STRIDE,
) -> pd.DataFrame:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    if splits is None:
        splits = load_split()

    mean, std = fit_pooled_standardization(df, splits, classes, REFLECTION_FEATURES, n, stride)

    pooled = {}
    temporal = {}
    for split_name in ("train", "test"):
        seqs = splits[split_name]
        pooled_embed, y_pooled = compute_pooled_embeddings(df, seqs, mean, std, classes, REFLECTION_FEATURES, n, stride)
        temporal_split = compute_temporal_variation_for_split(df, seqs, y_pooled, classes, n=n, stride=stride)
        pooled[split_name] = pooled_embed
        temporal[split_name] = temporal_split

    reg = LinearRegression()
    reg.fit(pooled["train"], temporal["train"])

    r2_train = r2_score(temporal["train"], reg.predict(pooled["train"]), multioutput="raw_values")
    r2_test = r2_score(temporal["test"], reg.predict(pooled["test"]), multioutput="raw_values")

    result = pd.DataFrame({"r2_train": r2_train, "r2_test": r2_test}, index=TEMPORAL_FEATURES)
    print("linear probe, pooled DeepReflecs embedding -> temporal variation scalar:")
    print(result.round(3).to_string())
    return result


if __name__ == "__main__":
    from build_points_table import build_and_save_points_table

    df = build_and_save_points_table(table_path=POINTS_TABLE_SEQ_PATH)
    df = add_relative_features_seq(df)
    df = apply_mlp_class_groups(df)

    embeddings_df = get_or_compute_scan_embeddings(df)

    model, history, test_seqs, y_test = run_gru_training(embeddings_df)
    metrics_df, _, _ = evaluate_gru_test_metrics(model, test_seqs, y_test)
    print(f"macro F1: {metrics_df['f1'].mean():.4f}")
