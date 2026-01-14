"""T-LOB training (Berti & Kasneci, 2025) on prebuilt windows.

Dual attention (temporal + feature) transformer for 3-class mid-price movement.
Defaults follow paper search ranges: d_model=256, heads=1, layers=3, lr=3e-4, T from data.
"""

from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import f1_score, confusion_matrix
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.tlob import TLOB

TICKER = "CSCO"
DATA_ROOT = Path.home() / "thesis_output" / "04_windows_NEW" / TICKER

DEVICE = (
    "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

BATCH_SIZE = 128 if DEVICE != "cpu" else 64
EPOCHS = 50
EPOCHS_FAST = 3
FAST_MODE = False  # Full training on complete dataset
MAX_TRAIN_SAMPLES = 20000
MAX_VAL_SAMPLES = 5000
LR_MAX = 3e-4
WEIGHT_DECAY = 0.01
LOG_EVERY = 200
PATIENCE = 10
LABEL_SMOOTHING = 0.1
CLIP_GRAD_NORM = 1.0
USE_SCHEDULER = True

D_MODEL = 256
N_HEADS = 1
N_LAYERS = 3
DROPOUT = 0.1

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
    out[y > tau] = 2
    return out


def auto_tau_from_train(root: Path, target_stat_pct: float) -> float:
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
# TRAIN / EVAL
# =============================================================================

def run_epoch(model, loader, optimizer, criterion, train=True, scheduler=None):
    model.train() if train else model.eval()

    losses = []
    preds_all, targets_all = [], []

    loop = tqdm(loader, leave=False)
    for i, (X, y) in enumerate(loop, 1):
        X, y = X.to(DEVICE), y.to(DEVICE)

        if train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(X)
        loss = criterion(logits, y)

        if train:
            loss.backward()
            if CLIP_GRAD_NORM is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

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
    print("T-LOB TRAINING — DUAL ATTENTION")
    print(f"TICKER : {TICKER}")
    print(f"DEVICE : {DEVICE}")
    print("=" * 90)

    global TAU
    if TAU is None:
        TAU = auto_tau_from_train(DATA_ROOT, AUTO_TAU_TARGET_STAT_PCT)
    print(f"TAU    : {TAU:.6f}  (auto target stat ~{AUTO_TAU_TARGET_STAT_PCT:.2f})")

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

    y_train = train_ds.y
    y_val = val_ds.y
    tr_counts = np.bincount(y_train, minlength=3)
    va_counts = np.bincount(y_val, minlength=3)

    print(f"\nClass Distribution:")
    print(f"Train counts (Down/Stat/Up): {tr_counts.tolist()}")
    print(f"Val   counts (Down/Stat/Up): {va_counts.tolist()}")
    print(f"Train pct: {np.round(tr_counts / tr_counts.sum(), 4).tolist()}")
    print(f"Val   pct: {np.round(va_counts / va_counts.sum(), 4).tolist()}")

    w = None
    if USE_CLASS_WEIGHTS:
        w = class_weights_from_train(y_train).to(DEVICE)
        print(f"\nClass weights: {w.detach().cpu().numpy().round(3).tolist()}")

    train_indices = None
    val_indices = None
    if FAST_MODE:
        train_indices = range(min(len(train_ds), MAX_TRAIN_SAMPLES))
        val_indices = range(min(len(val_ds), MAX_VAL_SAMPLES))
        print(f"\nFAST_MODE ON → using {len(train_indices)} train / {len(val_indices)} val samples")

    train_ds_eff = Subset(train_ds, train_indices) if train_indices is not None else train_ds
    val_ds_eff = Subset(val_ds, val_indices) if val_indices is not None else val_ds

    train_loader = DataLoader(
        train_ds_eff,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_ds_eff,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    T, F = train_ds[0][0].shape
    print(f"\nInput shape : T={T}, F={F}")
    print(f"Train samples: {len(train_ds_eff):,} (orig {len(train_ds):,})")
    print(f"Val samples  : {len(val_ds_eff):,} (orig {len(val_ds):,})")

    model = TLOB(
        n_features=F,
        num_classes=NUM_CLASSES,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=WEIGHT_DECAY)
    scheduler = None
    epochs_total = EPOCHS_FAST if FAST_MODE else EPOCHS
    if USE_SCHEDULER:
        scheduler = OneCycleLR(
            optimizer,
            max_lr=LR_MAX,
            epochs=epochs_total,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            div_factor=25.0,
            final_div_factor=1e3,
        )

    criterion = nn.CrossEntropyLoss(
        weight=w if w is not None else None,
        label_smoothing=LABEL_SMOOTHING,
    )

    best_val_f1 = -1.0
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs_total + 1):
        print(f"\nEpoch {epoch}/{epochs_total}")
        tr_loss, tr_f1, tr_cm = run_epoch(
            model, train_loader, optimizer, criterion, train=True, scheduler=scheduler
        )
        va_loss, va_f1, va_cm = run_epoch(
            model, val_loader, optimizer, criterion, train=False, scheduler=None
        )

        print(f"Train | loss={tr_loss:.4e} | macro-F1={tr_f1:.4f}")
        print_cm(tr_cm)
        print(f"Val   | loss={va_loss:.4e} | macro-F1={va_f1:.4f}")
        print_cm(va_cm)

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (no val F1 improve for {PATIENCE} epochs)")
            break

    if best_state is not None:
        torch.save(best_state, "best_tlob.pt")

    print("\n" + "=" * 90)
    print(f"✔ Best Val Macro-F1: {best_val_f1:.4f}")
    print("T-LOB: dual attention (temporal + feature) with MLP head.")
    print("=" * 90)


if __name__ == "__main__":
    main()
