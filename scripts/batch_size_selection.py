"""Picks BATCH_SIZE (and a matching LEARNING_RATE) from the train split's actual class
frequencies, instead of guessing. A batch of size bs misses a class entirely with probability
(1-p)^bs (binomial P(zero successes) - each of the bs draws independently misses the class with
probability 1-p), where p is that class's fraction of the training set. Rare classes
(large_vehicle, bus, two_wheeler) are the ones that matter here - a batch that never sees `bus`
contributes nothing to its weighted loss term that step.

This picks the smallest batch size that gets every class's miss-probability under a threshold
(rounded up to the next power of two, for GPU-friendly sizing), then scales the learning rate
with it via the linear scaling rule (Goyal et al.: LR should grow with batch size, since a bigger
batch means fewer, less noisy gradient updates per epoch - see the mlp_classifier.py batch-size
discussion). BASE_BATCH_SIZE/BASE_LR is mlp_classifier.py's original hand-picked, empirically
working config (Design_Decisions.md) - the scaling is anchored to a known-stable point, not
derived from nothing.

This is a diagnostic, not a training input: it prints a recommendation, it does not write to
mlp_classifier.py's BATCH_SIZE/LEARNING_RATE MACROS itself - that stays a deliberate, reviewed
edit.
"""
import math

import pandas as pd

from feature_distributions import FINAL_CLASSES
from sequence_split import load_split

BASE_BATCH_SIZE = 32
BASE_LR = 1e-5
CANDIDATE_BATCH_SIZES = (16, 32, 64, 128, 256, 512)


def train_class_frequencies(df: pd.DataFrame, classes: list[str] = FINAL_CLASSES) -> pd.Series:
    """Fraction of train-split instances belonging to each class (train only - val/test must
    never influence a hyperparameter choice, same discipline as the bin edges)."""
    splits = load_split()
    instances = df.drop_duplicates(["sequence_name", "timestamp", "track_id"])
    train = instances.loc[instances["sequence_name"].isin(splits["train"]) & instances["group"].isin(classes)]
    return train["group"].value_counts(normalize=True).reindex(classes)


def miss_probability_table(freqs: pd.Series, batch_sizes: tuple[int, ...] = CANDIDATE_BATCH_SIZES) -> pd.DataFrame:
    """P(a batch of size bs contains zero examples of this class) = (1-p)^bs, for each
    candidate batch size - the table that motivates the recommendation."""
    return pd.DataFrame({bs: (1 - freqs) ** bs for bs in batch_sizes})


def required_batch_size(freqs: pd.Series, target_p_zero: float = 0.10) -> int:
    """Smallest batch size such that EVERY class's miss-probability is <= target_p_zero - driven
    by the rarest class, since it's the binding constraint. Solves (1-p)^bs <= target for bs,
    then rounds up to the next power of two (GPU-friendly, and a batch size no longer needs to
    be read as a precisely-tuned number)."""
    p_rarest = freqs.min()
    bs_exact = math.log(target_p_zero) / math.log(1 - p_rarest)
    return 1 << (math.ceil(bs_exact) - 1).bit_length()


def scale_lr(batch_size: int, base_batch_size: int = BASE_BATCH_SIZE, base_lr: float = BASE_LR) -> float:
    """Linear scaling rule: LR grows proportionally with batch size, so fewer/larger gradient
    updates per epoch don't also mean smaller steps on top of that."""
    return base_lr * (batch_size / base_batch_size)


def select_hyperparameters(
    df: pd.DataFrame, classes: list[str] = FINAL_CLASSES, target_p_zero: float = 0.10
) -> tuple[int, float]:
    """Prints the miss-probability table and the resulting (batch_size, lr) recommendation.
    Returns (batch_size, lr) - does not write these anywhere; that's a deliberate manual edit to
    mlp_classifier.py's MACROS."""
    freqs = train_class_frequencies(df, classes)
    print(f"train-set class frequencies:\n{freqs.round(4).to_string()}\n")

    table = miss_probability_table(freqs)
    print(f"P(batch misses class entirely) by candidate batch size:\n{table.round(3).to_string()}\n")

    batch_size = required_batch_size(freqs, target_p_zero)
    lr = scale_lr(batch_size)
    rarest = freqs.idxmin()
    print(
        f"rarest class: {rarest} (p={freqs[rarest]:.4f})\n"
        f"target P(zero)<={target_p_zero:.0%} -> batch_size={batch_size}\n"
        f"linear-scaled from base (batch_size={BASE_BATCH_SIZE}, lr={BASE_LR:.0e}) -> lr={lr:.2e}"
    )
    return batch_size, lr


if __name__ == "__main__":
    from build_points_table import build_and_save_points_table
    from feature_distributions import apply_class_groups
    from taxonomy_separability import add_relative_features

    df = build_and_save_points_table()
    df = add_relative_features(df)
    df = apply_class_groups(df)

    select_hyperparameters(df)
