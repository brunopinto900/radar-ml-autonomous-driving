"""Generic class-separability probe: class-count-weighted logistic regression + random forest
on a sequence-grouped held-out split, with per-class one-vs-rest and pairwise ROC-AUC. Feature-
set agnostic - callers (taxonomy_separability.py, histogram_separability.py) build their own per-instance
feature matrix and hand it to run_probe.
"""
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from dataloader import RESULTS_DIR


def class_weights(labels: pd.Series) -> dict[str, float]:
    """weight_i = (largest class's count) / (class i's count) - the biggest class gets
    weight 1, a class with 1/10th its count gets weight 10."""
    counts = labels.value_counts()
    return (counts.max() / counts).to_dict()


def pairwise_roc_auc(clf, X, y_true, classes: list[str]) -> dict:
    """One-vs-one ROC-AUC per class pair: for each pair (A, B), restrict to the subset of X/y
    where the true label is A or B, and rank by the model's raw P(A) score (not renormalized
    over just the pair). Unlike one-vs-rest, this answers "how separable is A from B
    specifically" without pooling the third class into "rest"."""
    y_proba = clf.predict_proba(X)
    class_to_idx = {c: i for i, c in enumerate(clf.classes_)}
    y_true = np.asarray(y_true)

    results = {}
    for a, b in combinations(classes, 2):
        mask = np.isin(y_true, [a, b])
        y_binary = (y_true[mask] == a).astype(int)
        score = y_proba[mask, class_to_idx[a]]
        results[(a, b)] = roc_auc_score(y_binary, score)
    return results


def run_probe(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    classes: list[str],
    n_splits: int = 5,
    random_state: int = 0,
    verbose: bool = True,
    save_confusion: bool = True,
    tag: str = "probe",
) -> dict:
    """Sequence-grouped split (StratifiedGroupKFold on `groups`, e.g. sequence_name, ~1/n_splits
    held out - instances from the same sequence share background/weather/vehicle, so an
    instance-level split would leak sequence-specific signal into "held-out" test data) with
    leakage asserts, then class-count-weighted LogisticRegression + RandomForestClassifier,
    per-class one-vs-rest ROC-AUC. `verbose` gates the classification_report / pairwise-AUC
    prints - off for sweeps over many configurations where per-run detail is just noise.
    `save_confusion` gates the row-normalized confusion matrix plot, saved to
    results/{tag}_confusion_matrix_<model>.png. Returns {model_name: (report, auc_per_class,
    fig_or_None)}."""
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    assert set(y_train) == set(classes) and set(y_test) == set(classes), (
        "sequence-grouped split dropped a class from train or test - try a different random_state"
    )
    assert not set(groups[train_idx]) & set(groups[test_idx]), "train/test share a sequence - split is leaking"

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    weights = class_weights(pd.Series(y_train))
    majority_class = pd.Series(y_train).value_counts().idxmax()
    baseline_acc = (y_test == majority_class).mean()
    if verbose:
        print(f"class weights: {weights}")

    models = {
        "logistic_regression": (LogisticRegression(max_iter=1000, class_weight=weights), True),
        "random_forest": (
            RandomForestClassifier(n_estimators=300, class_weight=weights, random_state=random_state),
            False,
        ),
    }

    results = {}
    for name, (clf, needs_scaling) in models.items():
        X_tr, X_te = (X_train_scaled, X_test_scaled) if needs_scaling else (X_train, X_test)

        clf.fit(X_tr, y_train)
        y_pred = clf.predict(X_te)
        y_proba = clf.predict_proba(X_te)

        report = classification_report(y_test, y_pred)
        auc_scores = roc_auc_score(y_test, y_proba, multi_class="ovr", average=None, labels=clf.classes_)
        auc_per_class = dict(zip(clf.classes_, auc_scores))

        if verbose:
            print(f"\n=== {name} ===")
            print(report)
            print("per-class one-vs-rest ROC-AUC:")
            for cls, auc in auc_per_class.items():
                print(f"  {cls}: {auc:.3f}")
            if name == "logistic_regression":
                print("pairwise (one-vs-one) ROC-AUC:")
                for (a, b), auc in pairwise_roc_auc(clf, X_te, y_test, classes).items():
                    print(f"  {a} vs {b}: {auc:.3f}")
            print(f"majority-class baseline ({majority_class}): {baseline_acc:.3f} accuracy")

        fig = None
        if save_confusion:
            cm = confusion_matrix(y_test, y_pred, labels=classes, normalize="true")
            fig, ax = plt.subplots(figsize=(7, 5.5))
            ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, colorbar=False, values_format=".2f")
            ax.set_title(f"{tag} - {name}\n(sequence-grouped held-out, row-normalized)")
            fig.tight_layout()

            RESULTS_DIR.mkdir(exist_ok=True)
            path = RESULTS_DIR / f"{tag}_confusion_matrix_{name}.png"
            fig.savefig(path, dpi=150)
            if verbose:
                print(f"Saved {path}")

        results[name] = (report, auc_per_class, fig)

    return results
