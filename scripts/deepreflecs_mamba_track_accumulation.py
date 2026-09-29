"""Mamba (selective state-space model) counterpart to DeepReflecsGRU/DeepReflecsTransformer:
same frozen, precomputed per-scan embeddings (deepreflecs_rnn_track_accumulation.get_or_
compute_scan_embeddings) and same windowing, consumed by a selective SSM instead of a GRU
or self-attention. Motivation (see track_accumulation.md): a real deployment classifies
every incoming scan in real time under an automotive SoC's memory budget. A GRU already
fits that (O(1) state per update); a Transformer needs an explicit KV cache to match it.
Mamba is the third natural option: same O(1)-state-per-step streaming property as a GRU,
but the state update (and what to remember/forget) is input-dependent per channel, not a
single shared gate, closer in spirit to a bank of adaptive filters than one blended gate.

The official `mamba-ssm` package needs compiled CUDA kernels (causal_conv1d,
selective_scan_cuda) for its parallel associative scan, built for sequences thousands of
steps long. Every sequence here is at most WINDOW_N=20 steps, so a naive sequential Python
loop over time is already fast (no custom kernel needed) and this file implements the
selective SSM (S6) recurrence directly in plain PyTorch, avoiding a fragile CUDA-extension
build for a case that doesn't need the speed it buys."""
import numpy as np
import pandas as pd
import torch
from torch import nn

from deepreflecs_rnn_track_accumulation import (
    DEVICE,
    MLP_HIDDEN_DIM,
    RNN_DIR,
    STRIDE,
    WINDOW_N,
    _predict_transformer_in_batches as _predict_in_batches,
    pad_sequences,
    plot_training_curves,
    prepare_windowed_embedding_splits,
)
from feature_distributions import MLP_CLASSES
from mlp_classifier import MLP
from separability_probe import class_weights

D_MODEL = 32  # == point_dim, matches DeepReflecsTransformer, no input projection needed
D_STATE = 16  # Mamba paper default
D_CONV = 4  # causal depthwise conv kernel width, Mamba paper default
EXPAND = 2  # d_inner = EXPAND * D_MODEL
NUM_LAYERS = 2
EPOCHS = 100
BATCH_SIZE = 2048  # much larger than the branch's usual 128: the sequential per-timestep scan's
# cost is dominated by fixed per-step kernel-launch overhead, not FLOPs, and Mamba's per-sample
# memory footprint (T<=20, d_inner=64, d_state=16) is tiny, so a big batch amortizes that fixed
# cost over far more data per launch instead of paying it once per 128 samples
LEARNING_RATE = 4e-5
RANDOM_STATE = 0


class MambaBlock(nn.Module):
    """Minimal selective SSM (S6): input-dependent Delta/B/C (the "selective" part that
    sets Mamba apart from a fixed-parameter linear SSM like S4), computed via small
    projections from the input; a causal depthwise conv mixes in short local context
    before the projections, since a diagonal per-channel SSM alone can't. Sequential
    scan over time, correct but not the optimized parallel-scan kernel; fine at
    seq_len<=20."""

    def __init__(self, d_model: int, d_state: int = D_STATE, d_conv: int = D_CONV, expand: int = EXPAND):
        super().__init__()
        d_inner = expand * d_model
        dt_rank = max(1, d_model // 16)
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank

        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=d_conv, groups=d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))  # A = -exp(A_log), per (channel, state), > 0 ensures stability
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        x_in, res = self.in_proj(x).chunk(2, dim=-1)  # each (B, T, d_inner)

        x_conv = self.conv1d(x_in.transpose(1, 2))[:, :, :T]  # causal: trim the extra right-side padding
        x_conv = torch.nn.functional.silu(x_conv.transpose(1, 2))  # (B, T, d_inner)

        dt, B_mat, C_mat = torch.split(self.x_proj(x_conv), [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = torch.nn.functional.softplus(self.dt_proj(dt))  # (B, T, d_inner)
        A = -torch.exp(self.A_log)  # (d_inner, d_state), always < 0

        # Sequential scan for the linear recurrence h_t = a_t*h_{t-1} + b_t: a vectorized
        # closed form via cumsum in log-space (h_t = a_cum_t * cumsum_k(b_k/a_cum_k)) was tried
        # and rejected, exp(-log_a_cum) overflows once cumulative decay exceeds float32's exp
        # range (A up to -16, T=20 steps gets there easily), producing NaN loss silently. A
        # numerically stable vectorized version needs a proper chunked associative-scan combine
        # rule, real complexity this short a sequence doesn't need. T<=20 here, so the loop
        # itself is cheap in FLOPs; the real cost is Python-level kernel-launch overhead per
        # step, fixed per batch regardless of batch size, so a bigger BATCH_SIZE amortizes it
        # over more data per launch instead (see module docstring / BATCH_SIZE).
        h = x.new_zeros(B, self.d_inner, self.d_state)
        ys = []
        for t in range(T):
            dA = torch.exp(dt[:, t].unsqueeze(-1) * A)  # (B, d_inner, d_state)
            dB_u = dt[:, t].unsqueeze(-1) * B_mat[:, t].unsqueeze(1) * x_conv[:, t].unsqueeze(-1)
            h = h * dA + dB_u
            ys.append((h * C_mat[:, t].unsqueeze(1)).sum(-1))  # (B, d_inner)
        y = torch.stack(ys, dim=1) + x_conv * self.D  # (B, T, d_inner), D is a direct skip/feedthrough term

        y = y * torch.nn.functional.silu(res)  # gate, same role as a GRU's update gate
        return self.out_proj(y)


class MambaResidualBlock(nn.Module):
    def __init__(self, d_model: int, **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = MambaBlock(d_model, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mamba(self.norm(x))


class DeepReflecsMamba(nn.Module):
    """Same frozen per-scan embedding input and same final-real-position readout as
    DeepReflecsTransformer, sequence-mixing layer swapped for a stack of MambaResidualBlocks.
    No causal mask or key-padding mask needed unlike attention: the scan is inherently
    sequential (each step only ever depends on earlier steps), so left-aligned zero-padded
    steps after a sequence's real length simply never influence the state at that real
    length, reading out at `lengths - 1` is sufficient by construction."""

    def __init__(
        self, embedding_dim: int, d_model: int = D_MODEL, num_layers: int = NUM_LAYERS, d_state: int = D_STATE,
        d_conv: int = D_CONV, expand: int = EXPAND, num_classes: int = len(MLP_CLASSES),
        mlp_hidden_dim: int = MLP_HIDDEN_DIM,
    ):
        super().__init__()
        self.input_proj = nn.Identity() if embedding_dim == d_model else nn.Linear(embedding_dim, d_model)
        self.layers = nn.ModuleList(
            [MambaResidualBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand) for _ in range(num_layers)]
        )
        self.norm_f = nn.LayerNorm(d_model)
        self.head = MLP(input_dim=d_model, hidden_dim=mlp_hidden_dim, num_classes=num_classes, n_hidden_layers=1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h)
        h = self.norm_f(h)
        B = h.shape[0]
        last_idx = lengths.to(h.device) - 1
        last_hidden = h[torch.arange(B, device=h.device), last_idx]
        return self.head(last_hidden)


def train_mamba(
    train_seqs: list[np.ndarray], y_train: np.ndarray, val_seqs: list[np.ndarray], y_val: np.ndarray,
    classes: list[str] = MLP_CLASSES, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE, lr: float = LEARNING_RATE,
    random_state: int = RANDOM_STATE, d_model: int = D_MODEL, num_layers: int = NUM_LAYERS, d_state: int = D_STATE,
    d_conv: int = D_CONV, expand: int = EXPAND,
):
    torch.manual_seed(random_state)
    embedding_dim = train_seqs[0].shape[1]

    weights_by_class = class_weights(pd.Series([classes[i] for i in y_train]))
    weight_tensor = torch.tensor([weights_by_class[cls] for cls in classes], dtype=torch.float32, device=DEVICE)
    print(f"class weights: {dict(zip(classes, weight_tensor.tolist()))}")

    model = DeepReflecsMamba(
        embedding_dim, d_model=d_model, num_layers=num_layers, d_state=d_state, d_conv=d_conv, expand=expand,
        num_classes=len(classes),
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

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
            batch_y = torch.tensor(y_train[idx], device=DEVICE)

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
        val_acc = (val_logits.argmax(dim=1).cpu().numpy() == y_val).mean()

        history.append({"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc, "val_acc": float(val_acc)})
        print(f"epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}", flush=True)

    return model, history


def evaluate_mamba_test_metrics(
    model: DeepReflecsMamba, test_seqs: list[np.ndarray], y_test: np.ndarray, classes: list[str] = MLP_CLASSES,
    output_dir=None, n: int = WINDOW_N, stride: int = STRIDE,
):
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_recall_fscore_support

    if output_dir is None:
        output_dir = RNN_DIR / f"N{n}_stride{stride}_mamba"

    y_pred = _predict_in_batches(model, test_seqs).argmax(dim=1).cpu().numpy()

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=range(len(classes)), zero_division=0
    )
    metrics_df = pd.DataFrame({"precision": precision, "recall": recall, "f1": f1, "support": support}, index=classes)
    print(f"per-class precision/recall/f1 (test, mamba, N={n}, stride={stride}):")
    print(metrics_df.round(3).to_string())

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_json(output_dir / "mamba_test_metrics.json", orient="index", indent=2)
    print(f"Saved {output_dir / 'mamba_test_metrics.json'}")

    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_test, y_pred, labels=range(len(classes)), normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, colorbar=False, values_format=".2f")
    ax.set_title(f"Mamba: test confusion matrix (N={n})")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(output_dir / "mamba_test_confusion_matrix.png", dpi=150)
    print(f"Saved {output_dir / 'mamba_test_confusion_matrix.png'}")

    return metrics_df


def run_mamba_training(
    embeddings_df: pd.DataFrame, n: int = WINDOW_N, stride: int = STRIDE, classes: list[str] = MLP_CLASSES,
    epochs: int = EPOCHS, batch_size: int = BATCH_SIZE, lr: float = LEARNING_RATE, random_state: int = RANDOM_STATE,
    output_dir=None, splits: dict[str, list[str]] | None = None,
):
    if output_dir is None:
        output_dir = RNN_DIR / f"N{n}_stride{stride}_mamba"

    train_seqs, y_train, val_seqs, y_val, test_seqs, y_test = prepare_windowed_embedding_splits(
        embeddings_df, classes=classes, n=n, stride=stride, splits=splits,
    )

    model, history = train_mamba(
        train_seqs, y_train, val_seqs, y_val, classes=classes, epochs=epochs, batch_size=batch_size, lr=lr,
        random_state=random_state,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "mamba_model.pt")
    print(f"Saved {output_dir / 'mamba_model.pt'}")
    plot_training_curves(history, output_dir=output_dir)

    return model, history, test_seqs, y_test
