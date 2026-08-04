"""Ad-hoc diagnostic: does the min4pts-surviving dense-`car` tail actually
overlap with `large_vehicle` in raw 80-dim feature space, or is that just an
inference from proxies (point count/extent/bbox)? Two direct checks on the
actual feature vectors the MLP consumes (MLP_FINDINGS.md section 6):

1. Centroid distance - is the dense-car tail's mean feature vector closer to
   large_vehicle's than the full car population's mean is?
2. Separability probe - can a linear classifier (logistic regression) tell
   dense-car apart from large_vehicle as well as it can tell full-car apart
   from large_vehicle? If separability drops for the dense-only tail, that's
   direct evidence of feature overlap, not just a proxy correlation.

Both `large_vehicle` populations use the min4pts-filtered set (n=21,660) - the
actual population the min4pts model trained on - held fixed across both
comparisons so only the car-side population varies. Reuses cached features,
no retraining of the actual MLP.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

from dataloader import RESULTS_DIR
from train_mlp import CLASSES

CACHE_DIR = RESULTS_DIR / "mlp_feature_cache"
SEED = 42

CAR_IDX = CLASSES.index("car")
LV_IDX = CLASSES.index("large_vehicle")


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def probe(car_X: np.ndarray, lv_X: np.ndarray, label: str) -> None:
    X = np.concatenate([car_X, lv_X])
    y = np.concatenate([np.zeros(len(car_X)), np.ones(len(lv_X))])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]
    preds = clf.predict(X_test)

    auc = roc_auc_score(y_test, probs)
    bal_acc = balanced_accuracy_score(y_test, preds)
    print(f"{label}: n_car={len(car_X)} n_lv={len(lv_X)} -> held-out AUC={auc:.3f}  "
          f"balanced_acc={bal_acc:.3f}")


if __name__ == "__main__":
    X_train_full = np.load(CACHE_DIR / "train_X.npy")
    y_train_full = np.load(CACHE_DIR / "train_y.npy")
    X_train_min4 = np.load(CACHE_DIR / "train_min4pts_X.npy")
    y_train_min4 = np.load(CACHE_DIR / "train_min4pts_y.npy")

    lv_vec = X_train_min4[y_train_min4 == LV_IDX]          # large_vehicle, min4pts-filtered
    car_full_vec = X_train_full[y_train_full == CAR_IDX]    # car, full/unfiltered
    car_dense_vec = X_train_min4[y_train_min4 == CAR_IDX]   # car, min4pts-surviving tail

    print(f"lv (min4pts): n={len(lv_vec)}")
    print(f"car (full): n={len(car_full_vec)}")
    print(f"car (dense tail): n={len(car_dense_vec)}")

    print("\n--- 1. Centroid distance to large_vehicle ---")
    lv_centroid = lv_vec.mean(axis=0)
    car_full_centroid = car_full_vec.mean(axis=0)
    car_dense_centroid = car_dense_vec.mean(axis=0)

    cos_full = cosine_sim(car_full_centroid, lv_centroid)
    cos_dense = cosine_sim(car_dense_centroid, lv_centroid)
    euc_full = float(np.linalg.norm(car_full_centroid - lv_centroid))
    euc_dense = float(np.linalg.norm(car_dense_centroid - lv_centroid))

    print(f"car (full)  vs large_vehicle: cosine_sim={cos_full:.4f}  euclidean={euc_full:.3f}")
    print(f"car (dense) vs large_vehicle: cosine_sim={cos_dense:.4f}  euclidean={euc_dense:.3f}")
    print(f"cosine_sim increase (dense - full): {cos_dense - cos_full:+.4f}")
    print(f"euclidean decrease (full - dense): {euc_full - euc_dense:+.3f}")

    print("\n--- 2. Separability probe (logistic regression, held-out 30%) ---")
    probe(car_full_vec, lv_vec, "car (full)  vs large_vehicle")
    probe(car_dense_vec, lv_vec, "car (dense) vs large_vehicle")
