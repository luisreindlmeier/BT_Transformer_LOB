"""Transformer-only LiT (Xiao et al.) on prebuilt windows."""

from __future__ import annotations

from pathlib import Path
import json
import time
import numpy as np

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, confusion_matrix

TICKER = "CSCO"
DATA_ROOT = Path.home() / "thesis_output" / "04_windows_NEW" / TICKER

DEVICE = (
    "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

BATCH_SIZE = 128 if DEVICE != "cpu" else 64
EPOCHS = 50
LR_MAX = 2e-3
WEIGHT_DECAY = 0.01
LOG_EVERY = 200

PATIENCE = 10
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
PIN_MEMORY = False

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

def run_epoch(model, loader, optimizer, criterion, train: bool, scheduler=None):
    model.train() if train else model.eval()

    losses = []
    preds_all, targets_all = [], []

    loop = loader if train else loader
    t0 = time.time()
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

        preds_all.append(preds.detach().cpu().numpy())
        targets_all.append(y.detach().cpu().numpy())

        if train and i % LOG_EVERY == 0:
            dt = time.time() - t0
            print(f"  [{i:>5}] avg_loss={np.mean(losses[-LOG_EVERY:]):.4e} | elapsed={dt:.1f}s")

    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)

    f1 = macro_f1_true(targets_all, preds_all)
    cm = cm_true(targets_all, preds_all)
    return float(np.mean(losses)), float(f1), cm


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():
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

    y_train = train_ds.y
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
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
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

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
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
        torch.save(best_state, "best_lit_transformer.pt")

    print("\n" + "=" * 90)
    print(f"✔ Best Val Macro-F1: {best_val_f1:.4f}")
    print("Targets (Xiao et al.): Transformer should surpass MLP and be competitive with DeepLOB.")
    print("If F1 plateaus, consider tuning: d_model, n_layers, LR, weight_decay, epochs.")
    print("=" * 90)

import pytorch_lightning as pl

from src.models.lit_model import LiT
from src.models.datamodule import LOBDataModule


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main_lightning():
    """Optional Lightning entrypoint (kept for compatibility; not run by default)."""
    dm = LOBDataModule(
        dataset_root=PROJECT_ROOT / "data/datasets/lit_eventbased/CSCO",
        batch_size=64,
        num_workers=2,
    )

    dm.setup()

    model = LiT(
        class_weights=dm.class_weights,
    )

    trainer = pl.Trainer(
        max_epochs=3,
        accelerator="mps",
        devices=1,
        precision="32-true",
        enable_progress_bar=True,
        num_sanity_val_steps=0,
        val_check_interval=1.0,
    )

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()