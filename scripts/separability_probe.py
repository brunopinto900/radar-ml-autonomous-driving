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
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
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
    results_dir=RESULTS_DIR,
) -> dict:
    """Sequence-grouped split (StratifiedGroupKFold on `groups`, e.g. sequence_name, ~1/n_splits
    held out - instances from the same sequence share background/weather/vehicle, so an
    instance-level split would leak sequence-specific signal into "held-out" test data) with
    leakage asserts, then class-count-weighted LogisticRegression + RandomForestClassifier,
    per-class one-vs-rest ROC-AUC. `verbose` gates the classification_report / pairwise-AUC
    prints - off for sweeps over many configurations where per-run detail is just noise.
    `save_confusion` gates the row-normalized confusion matrix plot, saved to
    {results_dir}/{tag}_confusion_matrix_<model>.png (results_dir defaults to the results/ root,
    callers with their own subfolder - e.g. results/taxonomy/ - should pass it explicitly). High
    AUC does not imply good precision/recall - AUC is threshold-independent, but class weighting
    shifts the actual argmax decision boundary, so `metrics_per_class` (precision/recall/f1/
    support at that operating point, from
    precision_recall_fscore_support) is returned alongside auc_per_class rather than assuming
    one implies the other. Returns {model_name: (report, auc_per_class, metrics_per_class,
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

        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, labels=classes, zero_division=0
        )
        metrics_per_class = {
            cls: {"precision": p, "recall": r, "f1": f, "support": int(s)}
            for cls, p, r, f, s in zip(classes, precision, recall, f1, support)
        }

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

            results_dir.mkdir(parents=True, exist_ok=True)
            path = results_dir / f"{tag}_confusion_matrix_{name}.png"
            fig.savefig(path, dpi=150)
            if verbose:
                print(f"Saved {path}")

        results[name] = (report, auc_per_class, metrics_per_class, fig)

    return results


def run_probe_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    classes: list[str],
    n_splits: int = 5,
    random_state: int = 0,
) -> dict:
    """Like run_probe, but actually uses all n_splits folds instead of just the first, training
    fresh on each fold's 4/5 and evaluating on its 1/5, then reporting mean +/- std across folds.
    A single fold's AUC/F1 is noisy - see Design_Decisions.md decision 5's bus example, where a
    small class's F1 swung around between adjacent bin counts on one fold alone - so a number
    that's actually being compared or reported (rather than a quick default pick) should come
    from here, not run_probe. No confusion matrix/report text (those need one fixed split to
    display); prints and returns {model_name: {metric: {class: {"mean":.., "std":..}}}} for
    metric in ("auc", "precision", "recall", "f1")."""
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    model_names = ("logistic_regression", "random_forest")
    fold_scores = {name: {"auc": [], "precision": [], "recall": [], "f1": []} for name in model_names}

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        assert set(y_train) == set(classes) and set(y_test) == set(classes), (
            f"fold {fold}: sequence-grouped split dropped a class from train or test"
        )
        assert not set(groups[train_idx]) & set(groups[test_idx]), f"fold {fold}: train/test share a sequence"

        scaler = StandardScaler().fit(X_train)
        X_train_scaled, X_test_scaled = scaler.transform(X_train), scaler.transform(X_test)
        weights = class_weights(pd.Series(y_train))

        models = {
            "logistic_regression": (LogisticRegression(max_iter=1000, class_weight=weights), True),
            "random_forest": (
                RandomForestClassifier(n_estimators=300, class_weight=weights, random_state=random_state),
                False,
            ),
        }
        for name, (clf, needs_scaling) in models.items():
            X_tr, X_te = (X_train_scaled, X_test_scaled) if needs_scaling else (X_train, X_test)
            clf.fit(X_tr, y_train)
            y_pred = clf.predict(X_te)
            y_proba = clf.predict_proba(X_te)

            auc_scores = roc_auc_score(y_test, y_proba, multi_class="ovr", average=None, labels=clf.classes_)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, labels=classes, zero_division=0
            )
            fold_scores[name]["auc"].append(dict(zip(clf.classes_, auc_scores)))
            fold_scores[name]["precision"].append(dict(zip(classes, precision)))
            fold_scores[name]["recall"].append(dict(zip(classes, recall)))
            fold_scores[name]["f1"].append(dict(zip(classes, f1)))
        print(f"fold {fold + 1}/{n_splits} done")

    summary = {}
    for name, metrics in fold_scores.items():
        summary[name] = {}
        for metric, per_fold_dicts in metrics.items():
            summary[name][metric] = {
                cls: {
                    "mean": float(np.mean([d[cls] for d in per_fold_dicts])),
                    "std": float(np.std([d[cls] for d in per_fold_dicts])),
                }
                for cls in classes
            }

    for name in model_names:
        print(f"\n=== {name} (mean +/- std over {n_splits} folds) ===")
        for metric in ("auc", "precision", "recall", "f1"):
            table = pd.DataFrame(summary[name][metric]).T
            table.columns = [f"{metric}_mean", f"{metric}_std"]
            print(table.round(3).to_string())

    return summary
