"""
Minimal FI-2010 Baseline (SINGLE FILE)

- Loads FI-2010-style inputs from 03b
- Chronological train / val / test split
- Majority baseline
- Logistic Regression baseline
- Macro F1 + confusion matrices

If this fails -> DATA/TARGET issue
If this works -> MODEL was the issue
"""

from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, classification_report


# =============================================================================
# CONFIG
# =============================================================================

TICKER = "CSCO"

TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15

TAU = 0.002          # FIXED threshold like FI-2010
MAX_SAMPLES = 1_000_000   # cap for speed (can increase later)


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "model_inputs_fi2010" / f"inputs_{TICKER}.parquet"

assert DATA_PATH.exists(), f"Missing dataset: {DATA_PATH}"


# =============================================================================
# HELPERS
# =============================================================================

def classify(y_ret: np.ndarray, tau: float) -> np.ndarray:
    y = np.ones_like(y_ret, dtype=np.int64)
    y[y_ret > tau]  = 2
    y[y_ret < -tau] = 0
    return y


# =============================================================================
# LOAD DATA
# =============================================================================

print("=" * 90)
print("MINIMAL FI-2010 BASELINE")
print(f"TICKER : {TICKER}")
print("=" * 90)

pf = pq.ParquetFile(DATA_PATH)

X_list, y_ret_list = [], []

for batch in pf.iter_batches(batch_size=200_000):
    X_list.extend(batch.column("X").to_pylist())
    y_ret_list.extend(batch.column("y").to_pylist())

    if len(X_list) >= MAX_SAMPLES:
        break

X = np.asarray(X_list, dtype=np.float32)
y_ret = np.asarray(y_ret_list, dtype=np.float32)

y = classify(y_ret, TAU)

N = len(X)
print(f"Loaded samples: {N:,}")
print("Class distribution:", dict(zip(*np.unique(y, return_counts=True))))

# =============================================================================
# CHRONOLOGICAL SPLIT
# =============================================================================

n_train = int(TRAIN_FRAC * N)
n_val   = int(VAL_FRAC * N)

X_train, y_train = X[:n_train], y[:n_train]
X_val,   y_val   = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
X_test,  y_test  = X[n_train+n_val:], y[n_train+n_val:]

print("\nSplit sizes:")
print(f"Train: {len(y_train):,}")
print(f"Val  : {len(y_val):,}")
print(f"Test : {len(y_test):,}")


# =============================================================================
# BASELINE 1 — MAJORITY CLASS
# =============================================================================

maj_class = np.bincount(y_train).argmax()
y_pred_maj = np.full_like(y_test, maj_class)

print("\n=== MAJORITY BASELINE ===")
print("Majority class:", maj_class)
print("Macro F1:", f1_score(y_test, y_pred_maj, average="macro"))
print(confusion_matrix(y_test, y_pred_maj))


# =============================================================================
# BASELINE 2 — LOGISTIC REGRESSION
# =============================================================================

print("\n=== LOGISTIC REGRESSION BASELINE ===")

clf = LogisticRegression(
    max_iter=200,
    n_jobs=-1,
    class_weight="balanced",
)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Macro F1:", f1_score(y_test, y_pred, average="macro"))
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=["down", "stay", "up"]))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))


print("\nDONE.")