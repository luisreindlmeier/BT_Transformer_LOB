"""Transformer-only LiT (Xiao et al.) on prebuilt windows."""

from __future__ import annotations

from pathlib import Path
import os
import json
import csv
import time
import numpy as np

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_score, recall_score

# --- TRAINING CONFIGURATION (CENTRALIZED) ---
TICKER = os.getenv("TRAIN_TICKER", "CSCO")
DATA_FRACTION = float(os.getenv("TRAIN_DATA_FRACTION", "0.1"))
# Try local data first, fallback to ~/thesis_output
_local_data = Path(__file__).resolve().parent.parent.parent / "data" / "04_windows_NEW" / TICKER
_vm_data = Path.home() / "thesis_output" / "04_windows_NEW" / TICKER
DATA_ROOT = _local_data if _local_data.exists() else _vm_data

DEVICE = (
    "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

BATCH_SIZE = 256 if DEVICE != "cpu" else 64
EPOCHS = 10
LR_MAX = 2e-3
WEIGHT_DECAY = 0.01
LOG_EVERY = 500

PATIENCE = 3
LABEL_SMOOTHING = 0.1
USE_SCHEDULER = True

D_MODEL = 256
N_HEADS = 8
N_LAYERS = 6
D_FF = 4 * D_MODEL
DROPOUT = 0.2

NUM_CLASSES = 3
TAU = None
AUTO_TAU_TARGET_STAT_PCT = 0.6

NUM_WORKERS = 0
PIN_MEMORY = True if DEVICE != "cpu" else False  # MPS/CUDA benefit from pinned memory

USE_CLASS_WEIGHTS = True
CLIP_GRAD_NORM = 1.0

def banner(title: str) -> None:
    print("=" * 90)
    print(title)
    print("=" * 90)


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


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute comprehensive metrics: accuracy, precision, recall, F1 per class + macro."""
    acc = accuracy_score(y_true, y_pred)
    prec_per_class = precision_score(y_true, y_pred, labels=[0, 1, 2], zero_division=0, average=None)
    rec_per_class = recall_score(y_true, y_pred, labels=[0, 1, 2], zero_division=0, average=None)
    f1_per_class = (2 * prec_per_class * rec_per_class) / (prec_per_class + rec_per_class + 1e-12)
    macro_f1 = macro_f1_true(y_true, y_pred)
    cm = cm_true(y_true, y_pred)
    
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "precision_down": float(prec_per_class[0]),
        "precision_stat": float(prec_per_class[1]),
        "precision_up": float(prec_per_class[2]),
        "recall_down": float(rec_per_class[0]),
        "recall_stat": float(rec_per_class[1]),
        "recall_up": float(rec_per_class[2]),
        "f1_down": float(f1_per_class[0]),
        "f1_stat": float(f1_per_class[1]),
        "f1_up": float(f1_per_class[2]),
        "cm_true_down_pred_down": int(cm[0, 0]),
        "cm_true_down_pred_stat": int(cm[0, 1]),
        "cm_true_down_pred_up": int(cm[0, 2]),
        "cm_true_stat_pred_down": int(cm[1, 0]),
        "cm_true_stat_pred_stat": int(cm[1, 1]),
        "cm_true_stat_pred_up": int(cm[1, 2]),
        "cm_true_up_pred_down": int(cm[2, 0]),
        "cm_true_up_pred_stat": int(cm[2, 1]),
        "cm_true_up_pred_up": int(cm[2, 2]),
    }


def print_cm(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    print("  Confusion Matrix (rows=true, cols=pred):")
    print("             Pred:   Down      Stat        Up")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    names = ["Down", "Stat", "Up"]
    for i, name in enumerate(names):
        print(f"  True {name:<4}: {cm[i,0]:9d} {cm[i,1]:9d} {cm[i,2]:9d}")


def print_cm_obj(cm: np.ndarray) -> None:
    """Print confusion matrix from cm array (for backward compatibility)."""
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


# ------------------------------------------------------------
# DATASET
# ------------------------------------------------------------

class WindowDataset(Dataset):
    def __init__(self, X_path: Path, y_path: Path, tau: float):
        self.X = np.load(X_path, mmap_mode="r")  # (N, W, F)
        self.y_ret = np.load(y_path, mmap_mode="r")
        self.y = yret_to_class(self.y_ret, tau=tau)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.from_numpy(np.asarray(self.X[idx], dtype=np.float32).copy())
        y = torch.tensor(self.y[idx]).long()
        return x, y


# ------------------------------------------------------------
# MODEL (LiT-style Transformer only)
# ------------------------------------------------------------

class LiTTransformer(nn.Module):
    def __init__(self, n_features: int, window: int):
        super().__init__()
        self.window = window

        self.input_proj = nn.Linear(n_features, D_MODEL)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=D_FF,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, D_MODEL))
        self.pos_emb = nn.Parameter(torch.zeros(1, window + 1, D_MODEL))

        self.norm = nn.LayerNorm(D_MODEL)
        self.head = nn.Sequential(
            nn.Linear(D_MODEL, 256),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(256, NUM_CLASSES),
        )

        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        B, T, _ = x.shape

        h = self.input_proj(x)

        cls = self.cls_token.expand(B, -1, -1)  # (B,1,D)
        h = torch.cat([cls, h], dim=1)           # (B,1+T,D)

        pos = self.pos_emb[:, : T + 1]
        h = h + pos

        h = self.encoder(h)
        cls_out = h[:, 0]
        cls_out = self.norm(cls_out)
        return self.head(cls_out)


# ------------------------------------------------------------
# TRAIN / EVAL
# ------------------------------------------------------------

def run_epoch(model, loader, optimizer, criterion, train: bool, scheduler=None, should_stop=None):
    model.train() if train else model.eval()

    losses = []
    preds_all, targets_all = [], []

    loop = loader if train else loader
    t0 = time.time()
    for i, (X, y) in enumerate(loop, 1):
        # Check graceful stop signal
        if should_stop is not None and should_stop[0]:
            print("\n[STOP] Graceful shutdown requested.")
            break
        
        # Skip incomplete batches in training (prevents DataLoader deadlock)
        if train and X.shape[0] < BATCH_SIZE:
            continue
            
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

        preds_all.append(preds.detach().cpu().numpy())
        targets_all.append(y.detach().cpu().numpy())

        if train and i % LOG_EVERY == 0:
            dt = time.time() - t0
            print(f"  [{i:>5}] avg_loss={np.mean(losses[-LOG_EVERY:]):.4e} | elapsed={dt:.1f}s")

    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)

    f1 = macro_f1_true(targets_all, preds_all)
    cm = cm_true(targets_all, preds_all)
    metrics = compute_metrics(targets_all, preds_all)
    return float(np.mean(losses)), float(f1), cm, metrics


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():
    # Cap thread usage to avoid oversubscription freezes on CPU VMs
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    # Graceful shutdown flag
    should_stop = [False]

    def signal_handler(sig, frame):
        print("\n[SIGNAL] Ctrl+C received. Graceful shutdown after current batch...")
        should_stop[0] = True

    import signal
    signal.signal(signal.SIGINT, signal_handler)

    banner("LiT Transformer TRAINING — WINDOW DATA")
    print(f"TICKER : {TICKER}")
    print(f"DEVICE : {DEVICE}")

    # Meta (optional window hint)
    meta_path = DATA_ROOT / "meta.json"
    meta = None
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)

    global TAU
    if TAU is None:
        TAU = auto_tau_from_train(DATA_ROOT, AUTO_TAU_TARGET_STAT_PCT)
    print(f"TAU    : {TAU:.6f} (target stat ~{AUTO_TAU_TARGET_STAT_PCT:.2f})")

    train_ds = WindowDataset(DATA_ROOT / "train_X.npy", DATA_ROOT / "train_y.npy", tau=TAU)
    val_ds   = WindowDataset(DATA_ROOT / "val_X.npy",   DATA_ROOT / "val_y.npy",   tau=TAU)

    # Apply data fraction (subset training data)
    if DATA_FRACTION < 1.0:
        n_train = int(len(train_ds) * DATA_FRACTION)
        train_ds = torch.utils.data.Subset(train_ds, range(n_train))
        print(f"[DATA_FRACTION] Using {n_train:,} / {len(WindowDataset(DATA_ROOT / 'train_X.npy', DATA_ROOT / 'train_y.npy', tau=TAU)):,} train samples")

    y_train = train_ds.dataset.y if isinstance(train_ds, torch.utils.data.Subset) else train_ds.y
    y_val   = val_ds.y
    tr_counts = np.bincount(y_train, minlength=3)
    va_counts = np.bincount(y_val, minlength=3)

    print("\nClass Distribution:")
    print(f"Train counts (Down/Stat/Up): {tr_counts.tolist()}")
    print(f"Val   counts (Down/Stat/Up): {va_counts.tolist()}")
    print(f"Train pct: {np.round(tr_counts / tr_counts.sum(), 4).tolist()}")
    print(f"Val   pct: {np.round(va_counts / va_counts.sum(), 4).tolist()}")

    w = None
    if USE_CLASS_WEIGHTS:
        w = class_weights_from_train(y_train).to(DEVICE)
        print(f"\nClass weights: {w.detach().cpu().numpy().round(3).tolist()}")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,  # Don't drop - skip incomplete batches in training loop instead
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )

    T, F = train_ds[0][0].shape
    if meta and meta.get("window") and meta.get("feature_order"):
        print(f"Meta window: {meta['window']} | feature dim: {len(meta['feature_order'])}")
    print(f"Input shape : T={T}, F={F}")
    print(f"Train samples: {len(train_ds):,}")
    print(f"Val samples  : {len(val_ds):,}")

    model = LiTTransformer(n_features=F, window=T).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=WEIGHT_DECAY)
    scheduler = None
    if USE_SCHEDULER:
        scheduler = OneCycleLR(
            optimizer,
            max_lr=LR_MAX,
            epochs=EPOCHS,
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

    # CSV logging setup
    csv_path = Path(f"metrics_lit.csv")
    csv_fieldnames = ["epoch", "train_loss", "train_macro_f1", "val_loss", "val_macro_f1",
                      "val_accuracy", "val_precision_down", "val_precision_stat", "val_precision_up",
                      "val_recall_down", "val_recall_stat", "val_recall_up",
                      "val_f1_down", "val_f1_stat", "val_f1_up",
                      "val_cm_true_down_pred_down", "val_cm_true_down_pred_stat", "val_cm_true_down_pred_up",
                      "val_cm_true_stat_pred_down", "val_cm_true_stat_pred_stat", "val_cm_true_stat_pred_up",
                      "val_cm_true_up_pred_down", "val_cm_true_up_pred_stat", "val_cm_true_up_pred_up"]
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
    csv_writer.writeheader()

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        tr_loss, tr_f1, tr_cm, tr_metrics = run_epoch(
            model, train_loader, optimizer, criterion, train=True, scheduler=scheduler, should_stop=should_stop
        )
        va_loss, va_f1, va_cm, va_metrics = run_epoch(
            model, val_loader, optimizer, criterion, train=False, scheduler=None, should_stop=should_stop
        )

        # Check graceful stop
        if should_stop[0]:
            print(f"Stopping training at epoch {epoch}.")
            break

        print(f"Train | loss={tr_loss:.4e} | macro-F1={tr_f1:.4f}")
        print_cm_obj(tr_cm)
        print(f"Val   | loss={va_loss:.4e} | macro-F1={va_f1:.4f}")
        print_cm_obj(va_cm)

        # Write metrics to CSV
        csv_row = {"epoch": epoch, "train_loss": tr_loss, "train_macro_f1": tr_f1, 
                   "val_loss": va_loss, "val_macro_f1": va_f1}
        csv_row.update({f"val_{k}": v for k, v in va_metrics.items()})
        csv_writer.writerow(csv_row)
        csv_file.flush()

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (no val F1 improve for {PATIENCE} epochs)")
            break

    csv_file.close()
    if best_state is not None:
        torch.save(best_state, "best_lit_transformer.pt")

    print("\n" + "=" * 90)
    print(f"✔ Best Val Macro-F1: {best_val_f1:.4f}")
    print("Targets (Xiao et al.): Transformer should surpass MLP and be competitive with DeepLOB.")
    print("If F1 plateaus, consider tuning: d_model, n_layers, LR, weight_decay, epochs.")
    print("=" * 90)

if __name__ == "__main__":
    main()