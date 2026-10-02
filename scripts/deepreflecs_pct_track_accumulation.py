"""PCT (Guo, Cai, Liu, Mu, Martin & Hu, "PCT: Point Cloud Transformer", Computational
Visual Media 2021), adapted for the per-window point sets this branch's point-attention
line already built. deepreflecs_point_attention_track_accumulation.py's own classifier
(plain self-attention, one layer, straight max-pool) scored 0.8847 macro F1 at N=20
all-sensor, worse than the GRU (0.8895) and fusion (0.8897). This module adds the two
things that distinguish actual PCT from that simplified version: offset-attention
(Sec 3.2 of the paper) and a neighbor-embedding local-aggregation step before any global
attention. Everything else (windowing, recency tagging, the point cap bounding
attention's O(M^2) cost, ragged per-batch padding, split prep) is reused unchanged from
the existing module, same relationship deepreflecs_mamba_track_accumulation.py has to
the GRU line: new model/train/eval only, same proven data pipeline."""
import numpy as np
import pandas as pd
import torch
from torch import nn

from dataloader import RESULTS_DIR
from deepreflecs_classifier import DEVICE, REFLECTION_FEATURES
from deepreflecs_point_attention_track_accumulation import (
    POINT_CAP,
    STRIDE,
    WINDOW_N,
    pad_point_batch,
    prepare_point_attention_splits,
)
from feature_distributions import MLP_CLASSES
from separability_probe import class_weights

PCT_DIR = RESULTS_DIR / "track_accumulation_pct"

D_MODEL = 32
K_NEIGHBORS = 8
NUM_LAYERS = 4
EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 4e-5
RANDOM_STATE = 0

# REFLECTION_FEATURES = ["x_rel", "y_rel", "rcs", "vr_compensated", "range_sc"]: the
# neighbor-embedding step's spatial coordinates are just the first two feature columns,
# no separate data needed.
XY_COLS = slice(0, 2)


class OffsetAttention(nn.Module):
    """PCT's offset-attention (paper Sec 3.2): standard scaled dot-product attention,
    but the attention map gets a second L1 renormalization over the query axis after
    softmax (each key's total incoming attention sums to 1 across all real queries,
    not just each query's own distribution over keys), and the block adds back
    input-minus-attention-output (the "offset") through a Linear+LayerNorm+ReLU,
    instead of adding the attention output directly. Both are reported in the paper to
    behave like a learned Laplacian/graph-smoothing operator, more robust to sparse/
    noisy points than the plain self-attention already tried in
    deepreflecs_point_attention_track_accumulation.py.

    LayerNorm, not the paper's original BatchNorm: batch statistics over a ragged,
    masked, per-batch-padded point axis would be contaminated by how much padding a
    given batch happens to have, the same reason masked_pool (deepreflecs_classifier.py)
    never uses an unmasked reduction.

    Padded query rows are explicitly zeroed out of the softmax before the column-sum
    L1 renormalization: otherwise their attention distribution (computed from a
    zero-padded, not actually absent, input) would leak into real queries' own
    normalization by an amount that depends on how much padding this particular batch
    happens to contain, a batch-composition-dependent bug the mask is there to
    prevent, not merely a cosmetic detail."""

    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.lbr = nn.Sequential(nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.ReLU())
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        # x: (B, M, D), key_padding_mask: (B, M) bool, True = padding
        Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        energy = Q @ K.transpose(-1, -2) / (Q.shape[-1] ** 0.5)  # (B, M_query, M_key)
        energy = energy.masked_fill(key_padding_mask.unsqueeze(1), float("-inf"))

        attn = torch.softmax(energy, dim=-1)
        attn = attn.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)  # zero padded QUERY rows
        attn = attn / (attn.sum(dim=-2, keepdim=True) + 1e-8)  # L1 renorm over the query axis
        attn = self.dropout(attn)

        out = attn @ V
        offset = x - out
        return self.lbr(offset) + x


class NeighborEmbedding(nn.Module):
    """Local feature aggregation before any global attention (PCT's Neighbor Embedding,
    simplified: no furthest-point-sampling, this branch's windows already cap at
    POINT_CAP=300 points, unlike LiDAR-scale PCT's tens-of-thousands-of-points scenes).
    For each point, gathers its k nearest real neighbors by (x_rel, y_rel) within the
    same window, builds an EdgeConv-style [point, neighbor-point] feature, and
    max-pools over neighbors. Gives the model access to local geometric structure
    before attention mixes anything globally, which the existing point-attention
    module (straight to global self-attention) never had at all.

    A window with fewer than k real points (short tracks) falls back gracefully: the
    unavailable neighbor slots get a zero offset (contributes nothing distinguishing,
    not NaN/inf) rather than needing a separate code path."""

    def __init__(self, in_dim: int, out_dim: int, k: int = K_NEIGHBORS):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(nn.Linear(in_dim * 2, out_dim), nn.ReLU())

    def forward(self, feats: torch.Tensor, points_xy: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        # feats: (B, M, D), points_xy: (B, M, 2), key_padding_mask: (B, M) bool, True = padding
        B, M, D = feats.shape
        real_mask = ~key_padding_mask

        diff = points_xy.unsqueeze(2) - points_xy.unsqueeze(1)  # (B, M, M, 2)
        dist = (diff**2).sum(-1)  # (B, M, M)

        self_mask = torch.eye(M, dtype=torch.bool, device=feats.device).unsqueeze(0)
        invalid = self_mask | (~real_mask).unsqueeze(1)  # candidate j is self, or padding
        dist = dist.masked_fill(invalid, float("inf"))

        k_eff = min(self.k, M)
        knn_dist, knn_idx = dist.topk(k_eff, dim=-1, largest=False)  # (B, M, k_eff)

        knn_idx_flat = knn_idx.reshape(B, -1)
        gathered = torch.gather(feats, 1, knn_idx_flat.unsqueeze(-1).expand(-1, -1, D))
        neighbor_feats = gathered.reshape(B, M, k_eff, D)

        center = feats.unsqueeze(2).expand(-1, -1, k_eff, -1)
        no_neighbor = torch.isinf(knn_dist).unsqueeze(-1)  # (B, M, k_eff, 1)
        offset = torch.where(no_neighbor, torch.zeros_like(neighbor_feats), neighbor_feats - center)

        edge_feat = torch.cat([center, offset], dim=-1)  # (B, M, k_eff, 2D)
        return self.mlp(edge_feat).max(dim=2).values  # (B, M, out_dim)


class PCTClassifier(nn.Module):
    """point_embed+recency_embed (same convention as PointAttentionClassifier) ->
    NeighborEmbedding (local structure) -> num_layers stacked OffsetAttention blocks,
    concatenated (not just the last layer's output, PCT's own choice: lets the
    classifier draw on both early/local and late/global representations) -> fuse ->
    masked max-pool (matches DeepReflecs' own aggregation convention) -> classifier."""

    def __init__(
        self, n_features: int, max_recency: int, d_model: int = D_MODEL, k: int = K_NEIGHBORS,
        num_layers: int = NUM_LAYERS, num_classes: int = len(MLP_CLASSES), dropout: float = 0.0,
    ):
        super().__init__()
        self.point_embed = nn.Linear(n_features, d_model)
        self.recency_embed = nn.Embedding(max_recency, d_model)
        self.neighbor_embed = NeighborEmbedding(d_model, d_model, k=k)
        self.oa_layers = nn.ModuleList([OffsetAttention(d_model, dropout=dropout) for _ in range(num_layers)])
        self.fuse = nn.Linear(d_model * num_layers, d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, points: torch.Tensor, recency: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        B, M, _ = points.shape
        key_padding_mask = torch.arange(M, device=points.device).unsqueeze(0) >= lengths.unsqueeze(1)

        h = self.point_embed(points) + self.recency_embed(recency)
        h = self.neighbor_embed(h, points[:, :, XY_COLS], key_padding_mask)

        outs = []
        for layer in self.oa_layers:
            h = layer(h, key_padding_mask)
            outs.append(h)
        h = self.fuse(torch.cat(outs, dim=-1))

        real_mask = ~key_padding_mask
        h = h.masked_fill(~real_mask.unsqueeze(-1), float("-inf"))
        pooled = h.max(dim=1).values
        return self.classifier(pooled)


def _predict_in_batches(
    model: PCTClassifier, point_sets: list[np.ndarray], recency_sets: list[np.ndarray],
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


def train_pct(
    train_points: list[np.ndarray], train_recency: list[np.ndarray], y_train: np.ndarray,
    val_points: list[np.ndarray], val_recency: list[np.ndarray], y_val: np.ndarray,
    classes: list[str] = MLP_CLASSES, n: int = WINDOW_N, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE, random_state: int = RANDOM_STATE, d_model: int = D_MODEL,
    k: int = K_NEIGHBORS, num_layers: int = NUM_LAYERS,
):
    torch.manual_seed(random_state)
    n_features = train_points[0].shape[1]

    weights_by_class = class_weights(pd.Series([classes[i] for i in y_train]))
    weight_tensor = torch.tensor([weights_by_class[cls] for cls in classes], dtype=torch.float32, device=DEVICE)
    print(f"class weights: {dict(zip(classes, weight_tensor.tolist()))}")

    model = PCTClassifier(
        n_features, max_recency=n, d_model=d_model, k=k, num_layers=num_layers, num_classes=len(classes),
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


def evaluate_pct_test_metrics(
    model: PCTClassifier, test_points: list[np.ndarray], test_recency: list[np.ndarray],
    y_test: np.ndarray, classes: list[str] = MLP_CLASSES, output_dir=None, n: int = WINDOW_N, stride: int = STRIDE,
):
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_recall_fscore_support

    if output_dir is None:
        output_dir = PCT_DIR / f"N{n}_stride{stride}"

    n_features = test_points[0].shape[1]
    y_pred = _predict_in_batches(model, test_points, test_recency, n_features).argmax(dim=1).cpu().numpy()

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=range(len(classes)), zero_division=0
    )
    metrics_df = pd.DataFrame({"precision": precision, "recall": recall, "f1": f1, "support": support}, index=classes)
    print(f"per-class precision/recall/f1 (test, PCT, N={n}, stride={stride}):")
    print(metrics_df.round(3).to_string())

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_json(output_dir / "pct_test_metrics.json", orient="index", indent=2)
    print(f"Saved {output_dir / 'pct_test_metrics.json'}")

    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_test, y_pred, labels=range(len(classes)), normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, colorbar=False, values_format=".2f")
    ax.set_title(f"PCT: test confusion matrix (N={n})")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(output_dir / "pct_test_confusion_matrix.png", dpi=150)
    print(f"Saved {output_dir / 'pct_test_confusion_matrix.png'}")

    return metrics_df


def run_pct_training(
    df: pd.DataFrame, classes: list[str] = MLP_CLASSES, n: int = WINDOW_N, stride: int = STRIDE,
    range_sc_mode: str = "broadcast", point_cap: int = POINT_CAP, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE, random_state: int = RANDOM_STATE, output_dir=None,
    splits: dict[str, list[str]] | None = None,
):
    if output_dir is None:
        output_dir = PCT_DIR / f"N{n}_stride{stride}"

    train_data, val_data, test_data = prepare_point_attention_splits(
        df, classes, REFLECTION_FEATURES, n, stride, range_sc_mode, point_cap, splits,
    )
    train_points, train_recency, y_train = train_data
    val_points, val_recency, y_val = val_data
    test_points, test_recency, y_test = test_data

    model, history = train_pct(
        train_points, train_recency, y_train, val_points, val_recency, y_val,
        classes=classes, n=n, epochs=epochs, batch_size=batch_size, lr=lr, random_state=random_state,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "pct_model.pt")
    print(f"Saved {output_dir / 'pct_model.pt'}")

    from deepreflecs_classifier import plot_training_curves
    plot_training_curves(history, output_dir=output_dir)

    return model, history, test_points, test_recency, y_test
