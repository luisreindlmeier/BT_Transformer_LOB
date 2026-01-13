"""DeepLOB training on prebuilt windows."""

from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, confusion_matrix
from tqdm import tqdm

TICKER = "CSCO"
DATA_ROOT = Path("data/04_windows_NEW") / TICKER

DEVICE = (
    "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

BATCH_SIZE = 128 if DEVICE != "cpu" else 64
EPOCHS = 1
LR = 1e-3
LOG_EVERY = 200

NUM_CLASSES = 3
TAU = None
AUTO_TAU_TARGET_STAT_PCT = 0.6

NUM_WORKERS = 0
PIN_MEMORY = False

USE_CLASS_WEIGHTS = True

def yret_to_class(y_ret: np.ndarray, tau: float) -> np.ndarray:
    y = y_ret.astype(np.float32, copy=False)
    out = np.full_like(y, 1, dtype=np.int64)
    out[y < -tau] = 0
    out[y >  tau] = 2
    return out

def auto_tau_from_train(root: Path, target_stat_pct: float) -> float:
    """Calibrate tau using |y_ret| quantile; ensure >0 by using tiny epsilon fallback."""
    y_path = root / "train_y.npy"
    if not y_path.exists():
        raise FileNotFoundError(y_path)
    y_train_ret = np.load(y_path, mmap_mode="r")
    abs_y = np.abs(y_train_ret.astype(np.float64, copy=False))
    q = float(np.quantile(abs_y, target_stat_pct))
    return max(q, 1e-6)

def macro_f1_true(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0))

def cm_true(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

def print_cm(cm: np.ndarray) -> None:
    print("  Confusion Matrix (rows=true, cols=pred):")
    print("             Pred:   Down      Stat        Up")
    names = ["Down", "Stat", "Up"]
    for i, name in enumerate(names):
        print(f"  True {name:<4}: {cm[i,0]:9d} {cm[i,1]:9d} {cm[i,2]:9d}")

def class_weights_from_train(y_train: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y_train, minlength=3).astype(np.float64)
    w = 1.0 / np.maximum(counts, 1.0)
    w = w / np.mean(w)
    return torch.tensor(w, dtype=torch.float32)

class WindowDataset(Dataset):
    def __init__(self, X_path: Path, y_path: Path, tau: float):
        self.X = np.load(X_path, mmap_mode="r")
        self.y_ret = np.load(y_path, mmap_mode="r")
        self.y = yret_to_class(self.y_ret, tau=tau)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.from_numpy(np.asarray(self.X[idx], dtype=np.float32).copy())
        y = torch.tensor(self.y[idx]).long()
        return x, y

# =============================================================================
# DEEPLOB MODEL
# =============================================================================

class DeepLOB(nn.Module):
    """
    Simplified DeepLOB-style architecture
    """

    def __init__(self, n_features: int):
        super().__init__()

        # Input: (B, 1, T, F)
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 5), padding=(0, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(1, 5), padding=(0, 2)),
            nn.ReLU(),
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
        )

        # After convs: (B, C=64, T, F)
        self.gru = nn.GRU(
            input_size=64 * n_features,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
        )

        self.fc = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        # x: (B, T, F)
        B, T, F = x.shape

        x = x.unsqueeze(1)          # (B, 1, T, F)
        x = self.conv_block_1(x)    # (B, 32, T, F)
        x = self.conv_block_2(x)    # (B, 64, T, F)

        x = x.permute(0, 2, 1, 3)   # (B, T, C, F)
        x = x.reshape(B, T, -1)    # (B, T, C*F)

        _, h = self.gru(x)          # h: (1, B, 128)
        h = h.squeeze(0)

        return self.fc(h)

# =============================================================================
# TRAIN / EVAL
# =============================================================================

def run_epoch(model, loader, optimizer, criterion, train=True):
    model.train() if train else model.eval()

    losses = []
    preds_all, targets_all = [], []

    loop = tqdm(loader, leave=False)
    for i, (X, y) in enumerate(loop, 1):
        X, y = X.to(DEVICE), y.to(DEVICE)

        if train:
            optimizer.zero_grad()

        logits = model(X)
        loss = criterion(logits, y)

        if train:
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        preds = torch.argmax(logits, dim=1)

        preds_all.append(preds.cpu().numpy())
        targets_all.append(y.cpu().numpy())

        if train and i % LOG_EVERY == 0:
            loop.set_postfix(loss=np.mean(losses[-LOG_EVERY:]))

    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)

    f1 = macro_f1_true(targets_all, preds_all)
    cm = cm_true(targets_all, preds_all)
    return float(np.mean(losses)), float(f1), cm

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 90)
    print("DEEPLOB TRAINING — SANITY CHECK")
    print(f"TICKER : {TICKER}")
    print(f"DEVICE : {DEVICE}")
    print("=" * 90)

    # Auto-calibrate TAU
    global TAU
    if TAU is None:
        TAU = auto_tau_from_train(DATA_ROOT, AUTO_TAU_TARGET_STAT_PCT)
    print(f"TAU    : {TAU:.6f}  (auto target stat ~{AUTO_TAU_TARGET_STAT_PCT:.2f})")

    # Load datasets
    train_ds = WindowDataset(
        DATA_ROOT / "train_X.npy",
        DATA_ROOT / "train_y.npy",
        tau=TAU,
    )
    val_ds = WindowDataset(
        DATA_ROOT / "val_X.npy",
        DATA_ROOT / "val_y.npy",
        tau=TAU,
    )

    # Print class distribution
    y_train = train_ds.y
    y_val = val_ds.y
    tr_counts = np.bincount(y_train, minlength=3)
    va_counts = np.bincount(y_val, minlength=3)
    
    print(f"\nClass Distribution:")
    print(f"Train counts (Down/Stat/Up): {tr_counts.tolist()}")
    print(f"Val   counts (Down/Stat/Up): {va_counts.tolist()}")
    print(f"Train pct: {np.round(tr_counts / tr_counts.sum(), 4).tolist()}")
    print(f"Val   pct: {np.round(va_counts / va_counts.sum(), 4).tolist()}")

    # Class weights
    w = None
    if USE_CLASS_WEIGHTS:
        w = class_weights_from_train(y_train).to(DEVICE)
        print(f"\nClass weights (normalized inverse freq): {w.detach().cpu().numpy().round(3).tolist()}")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    T, F = train_ds[0][0].shape
    print(f"\nInput shape : T={T}, F={F}")
    print(f"Train samples: {len(train_ds):,}")
    print(f"Val samples  : {len(val_ds):,}")

    model = DeepLOB(n_features=F).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(weight=w) if w is not None else nn.CrossEntropyLoss()

    best_val_f1 = 0.0

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        train_loss, train_f1, train_cm = run_epoch(
            model, train_loader, optimizer, criterion, train=True
        )
        val_loss, val_f1, val_cm = run_epoch(
            model, val_loader, optimizer, criterion, train=False
        )

        print(f"Train | loss={train_loss:.4e} | macro-F1={train_f1:.4f}")
        print_cm(train_cm)
        print(f"Val   | loss={val_loss:.4e} | macro-F1={val_f1:.4f}")
        print_cm(val_cm)

        best_val_f1 = max(best_val_f1, val_f1)

    print("\n" + "=" * 90)
    print(f"✔ Best Val Macro-F1: {best_val_f1:.4f}")
    print("Expected:")
    print("  - Must beat MLP baseline (~0.37–0.38)")
    print("  - Good sign if ≥ 0.42")
    print("  - Strong signal if ≥ 0.50")
    print("=" * 90)


if __name__ == "__main__":
    main()