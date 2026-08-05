"""Resolves the truck-vs-bus grouping conflict from MLP_FINDINGS.md section 10: extent says
truck looks like car and bus is the outlier; RCS says the opposite (bus closer to car, truck
the outlier). Single-feature comparison can't settle which merge is actually supported -this
runs the same separability probe used in section 6/car_large_vehicle_feature_overlap.py (a
logistic regression on the full feature vector, not one stat at a time), across all pairs of
car/truck/bus/large_vehicle (raw), in both the sparse (n_points<=3, the regime that drives the
actual confusion) and overall regimes - specifically to check a proposed {car,truck} vs.
{bus,large_vehicle(raw)} grouping, which the extent numbers alone don't obviously support
(large_vehicle(raw)'s sparse median extent, 0.69m, is closer to car's, 0.83m, than to bus's,
1.81m).

Builds fresh 80-dim histogram feature vectors (same FEATURES/bin_edges as the canonical
baseline) split by the pre-merge label_name column, rather than reusing train_X.npy (which is
keyed by the merged class_name and can't distinguish truck from bus).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from dataloader import RESULTS_DIR
from histogram_features import (FEATURES, GROUP_KEY, N_BINS, add_relative_xy, fit_bin_edges,
                                 instance_vector)

OUT_DIR = RESULTS_DIR / "mlp_full_run" / "truck_bus_separability"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42
SPARSE_MAX_POINTS = 3


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def probe(X_a: np.ndarray, X_b: np.ndarray, label: str, lines: list[str]) -> None:
    X = np.concatenate([X_a, X_b])
    y = np.concatenate([np.zeros(len(X_a)), np.ones(len(X_b))])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]
    preds = clf.predict(X_test)

    auc = roc_auc_score(y_test, probs)
    bal_acc = balanced_accuracy_score(y_test, preds)
    centroid_a, centroid_b = X_a.mean(axis=0), X_b.mean(axis=0)
    cos = cosine_sim(centroid_a, centroid_b)
    euc = float(np.linalg.norm(centroid_a - centroid_b))
    line = (f"{label}: n_a={len(X_a)} n_b={len(X_b)}  held-out AUC={auc:.3f}  "
            f"balanced_acc={bal_acc:.3f}  centroid cosine_sim={cos:.4f}  euclidean={euc:.3f}")
    print(line)
    lines.append(line)


if __name__ == "__main__":
    train_df = add_relative_xy(pd.read_parquet(RESULTS_DIR / "train_points.parquet"))
    bin_edges = fit_bin_edges(train_df)  # same edges the canonical baseline uses

    n_points = train_df.groupby(GROUP_KEY).size().rename("n_points")

    GROUPS = ("car", "truck", "bus", "large_vehicle")
    rows = []
    for key, pts in train_df.groupby(GROUP_KEY):
        label = pts["label_name"].iloc[0]
        if label not in GROUPS:
            continue
        vec = instance_vector(pts, bin_edges)
        rows.append((label, len(pts), vec))

    labels = np.array([r[0] for r in rows])
    n_pts = np.array([r[1] for r in rows])
    X = np.stack([r[2] for r in rows])

    import itertools
    report_lines = []
    for regime, mask_fn in [("overall", lambda n: np.ones_like(n, dtype=bool)),
                             (f"sparse (n_points<={SPARSE_MAX_POINTS})",
                              lambda n: n <= SPARSE_MAX_POINTS)]:
        mask = mask_fn(n_pts)
        print(f"\n--- {regime} ---")
        report_lines.append(f"--- {regime} ---")
        group_X = {g: X[mask & (labels == g)] for g in GROUPS}
        for a, b in itertools.combinations(GROUPS, 2):
            probe(group_X[a], group_X[b], f"{a} vs {b}", report_lines)

    (OUT_DIR / "results.txt").write_text("\n".join(report_lines) + "\n")
    print(f"\nSaved {OUT_DIR / 'results.txt'}")
