"""Baseline MLP training on prebuilt windows with feature ablation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import json
import time
import numpy as np

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import f1_score, confusion_matrix

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_score, recall_score

import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.simple_mlp import SimpleMLP

DEVICE = (
    "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else
    ("cuda" if torch.cuda.is_available() else "cpu")
)

# --- TRAINING CONFIGURATION (CENTRALIZED) ---
TICKER = os.getenv("TRAIN_TICKER", "CSCO")
DATA_FRACTION = float(os.getenv("TRAIN_DATA_FRACTION", "0.1"))  # Train on 10% of dataset (can be overridden via CLI)
# Try local data first, fallback to ~/thesis_output
_local_data = Path(__file__).resolve().parent.parent.parent / "data" / "04_windows_NEW" / TICKER
_vm_data = Path.home() / "thesis_output" / "04_windows_NEW" / TICKER
DATA_ROOT = _local_data if _local_data.exists() else _vm_data

WINDOW = None
TAU = None
AUTO_TAU_TARGET_STAT_PCT = 0.6
N_EVENT_FEATURES = 9

BATCH_SIZE = 64  # Reduced for CPU stability (was 512 GPU / 256 CPU)
EPOCHS = 10
LR_MAX = 2e-3
WEIGHT_DECAY = 0.01
PATIENCE = 3
LABEL_SMOOTHING = 0.1
USE_SCHEDULER = True
CLIP_GRAD_NORM = 1.0

# Safe DataLoader defaults (avoid deadlocks on CPU VMs)
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "0"))  # 0 == main process loading
PIN_MEMORY = False
DL_TIMEOUT = int(os.getenv("DL_TIMEOUT", "120"))  # seconds for worker recv (if workers>0)
PREFETCH_FACTOR = int(os.getenv("PREFETCH_FACTOR", "2"))
PERSISTENT_WORKERS = False

def make_loader(ds, batch_size, shuffle):
    """Create DataLoader with timeout only if NUM_WORKERS > 0 (required by PyTorch)."""
    kwargs = dict(
        dataset=ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )
    if NUM_WORKERS > 0:
        kwargs["timeout"] = DL_TIMEOUT
        kwargs["prefetch_factor"] = PREFETCH_FACTOR
        kwargs["persistent_workers"] = PERSISTENT_WORKERS
    return DataLoader(**kwargs)

USE_CLASS_WEIGHTS = True

FEATURE_CONFIGS = [
    ("all", "All Features (Events + LOB)"),
    ("events_only", "Events Only"),
    ("lob_only", "LOB Only"),
]

def banner(title: str) -> None:
    print("=" * 90)
    print(title)
    print("=" * 90)

def yret_to_class(y_ret: np.ndarray, tau: float) -> np.ndarray:
    """
    Map log-return -> 3-class:
      0: down  (y < -tau)
      1: stat  (|y| <= tau)
      2: up    (y > tau)
    """
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
    # if data are mostly zeros, quantile will be 0; keep a tiny epsilon to avoid degenerate binning
    return max(q, 1e-6)

def macro_f1_true(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # TRUE macro over all 3 classes, even if a class is missing in y_true
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

def print_cm(cm: np.ndarray) -> None:
    # rows=true, cols=pred
    # classes: 0=Down,1=Stat,2=Up
    print("  Confusion Matrix (rows=true, cols=pred):")
    print("             Pred:   Down      Stat        Up")
    names = ["Down", "Stat", "Up"]
    for i, name in enumerate(names):
        print(f"  True {name:<4}: {cm[i,0]:9d} {cm[i,1]:9d} {cm[i,2]:9d}")

def compute_baselines(y_train: np.ndarray, y_val: np.ndarray) -> dict:
    maj_class = int(np.bincount(y_train, minlength=3).argmax())
    y_pred_maj = np.full_like(y_val, maj_class)
    f1_maj = macro_f1_true(y_val, y_pred_maj)

    rng = np.random.default_rng(42)
    y_pred_rand = rng.integers(0, 3, size=len(y_val), dtype=np.int64)
    f1_rand = macro_f1_true(y_val, y_pred_rand)

    return {
        "majority_class": maj_class,
        "majority_f1": float(f1_maj),
        "random_f1": float(f1_rand),
    }

def class_weights_from_train(y_train: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y_train, minlength=3).astype(np.float64)
    # inverse frequency
    w = 1.0 / np.maximum(counts, 1.0)
    # normalize so mean weight ~ 1.0
    w = w / np.mean(w)
    return torch.tensor(w, dtype=torch.float32)

@dataclass
class SplitFiles:
    X_path: Path
    y_path: Path

def split_files(root: Path, split: str) -> SplitFiles:
    return SplitFiles(
        X_path=root / f"{split}_X.npy",
        y_path=root / f"{split}_y.npy",
    )

class WindowDataset(Dataset):
    def __init__(self, root: Path, split: str, feature_type: str, n_event_features: int, tau: float, window_override: int | None = None):
        sf = split_files(root, split)
        if not sf.X_path.exists() or not sf.y_path.exists():
            raise FileNotFoundError(f"Missing files for split={split}: {sf.X_path} / {sf.y_path}")

        self.X = np.load(sf.X_path, mmap_mode="r")  # (N, W, F)
        self.y_ret = np.load(sf.y_path, mmap_mode="r")  # (N,)
        self.y = yret_to_class(self.y_ret, tau=tau)

        if self.X.ndim != 3:
            raise ValueError(f"Expected X to be 3D (N,W,F). Got shape={self.X.shape}")

        self.feature_type = feature_type
        self.n_event = int(n_event_features)

        _, W, F = self.X.shape
        target_window = window_override or WINDOW
        if target_window is not None and W != target_window:
            print(f"[WARN] WINDOW mismatch: expected {target_window}, got {W}")
        # keep self.W to actual data for downstream reshapes

        if not (0 < self.n_event < F):
            raise ValueError(f"n_event_features={self.n_event} invalid for F={F}")

        self.W = W
        self.F = F

    def __len__(self) -> int:
        return self.X.shape[0]

    def _slice(self, x: np.ndarray) -> np.ndarray:
        # x: (W,F)
        if self.feature_type == "all":
            return x
        if self.feature_type == "events_only":
            return x[:, : self.n_event]
        if self.feature_type == "lob_only":
            return x[:, self.n_event :]
        raise ValueError(f"Unknown feature_type={self.feature_type}")

    def __getitem__(self, idx: int):
        x = self.X[idx]              # (W,F) memmap view
        x = self._slice(x)           # (W,F')
        x = x.reshape(-1)            # flatten (W*F',)
        x = torch.from_numpy(np.asarray(x, dtype=np.float32).copy())  # copy to avoid non-writable warning
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return x, y

def make_loader(root: Path, split: str, feature_type: str, shuffle: bool, sampler=None, window_override: int | None = None) -> DataLoader:
    ds = WindowDataset(
        root=root,
        split=split,
        feature_type=feature_type,
        n_event_features=N_EVENT_FEATURES,
        tau=TAU,
        window_override=window_override,
    )
    kwargs = dict(
        dataset=ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,  # Don't drop - skip incomplete batches in training loop instead
    )
    if NUM_WORKERS > 0:
        kwargs["timeout"] = DL_TIMEOUT
        kwargs["prefetch_factor"] = PREFETCH_FACTOR
        kwargs["persistent_workers"] = PERSISTENT_WORKERS
    return DataLoader(**kwargs)

# -----------------------------------------------------------------------------
# TRAIN / EVAL
# -----------------------------------------------------------------------------

def run_epoch(model, loader, criterion, optimizer, device: str, train: bool, scheduler=None, should_stop=None):
    if train:
        model.train()
    else:
        model.eval()

    all_t = []
    all_p = []
    losses = []

    t0 = time.time()

    for step, (X, y) in enumerate(loader, 1):
        # Check graceful stop signal (set by KeyboardInterrupt handler)
        if should_stop is not None and should_stop[0]:
            print("\n[STOP] Graceful shutdown requested.")
            break
        
        # Skip incomplete batches in training (prevents DataLoader deadlock)
        if train and X.shape[0] < BATCH_SIZE:
            continue
            
        X = X.to(device, non_blocking=False)
        y = y.to(device, non_blocking=False)

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

        losses.append(float(loss.item()))
        preds = torch.argmax(logits, dim=1)

        all_t.append(y.detach().cpu().numpy())
        all_p.append(preds.detach().cpu().numpy())

        if step % 200 == 0:
            dt = time.time() - t0
            phase = "train" if train else "val"
            print(f"  [{phase} {step:>5}/{len(loader)}] avg_loss={np.mean(losses[-200:]):.4e} | elapsed={dt:.1f}s")

    y_true = np.concatenate(all_t).astype(np.int64)
    y_pred = np.concatenate(all_p).astype(np.int64)

    f1 = macro_f1_true(y_true, y_pred)
    cm = cm_true(y_true, y_pred)
    metrics = compute_metrics(y_true, y_pred)
    return float(np.mean(losses)), f1, cm, metrics

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

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

    # Graceful shutdown flag (shared across epochs)
    should_stop = [False]

    def signal_handler(sig, frame):
        print("\n[SIGNAL] Ctrl+C received. Graceful shutdown after current batch...")
        should_stop[0] = True

    import signal
    signal.signal(signal.SIGINT, signal_handler)
    banner("BASELINE TRAINING — HARD SANITY CHECK + FEATURE ABLATION (FIXED)")

    # read meta if present to pick window / feature order
    meta_path = DATA_ROOT / "meta.json"
    meta = None
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)

    global WINDOW
    if WINDOW is None and meta and "window" in meta:
        WINDOW = int(meta["window"])

    print(f"TICKER      : {TICKER}")
    print(f"DEVICE      : {DEVICE}")
    print(f"BATCH_SIZE  : {BATCH_SIZE}")
    # Auto-calibrate TAU if None
    global TAU
    if TAU is None:
        TAU = auto_tau_from_train(DATA_ROOT, AUTO_TAU_TARGET_STAT_PCT)
    print(f"TAU         : {TAU:.6f}  (auto target stat ~{AUTO_TAU_TARGET_STAT_PCT:.2f})")
    if meta and meta.get("feature_order"):
        print(f"WINDOW      : {WINDOW} (from meta)")
        print(f"Feature dim : {len(meta['feature_order'])}")
    print(f"N_EVENT_FEAT: {N_EVENT_FEATURES}  (must match how Step05 concatenated features!)")
    print(f"CLASS_WEIGHTS: {USE_CLASS_WEIGHTS}")
    print("")

    # --- load labels directly via dataset (cheap, memmap) for baselines ---
    ds_train_all = WindowDataset(DATA_ROOT, "train", "all", N_EVENT_FEATURES, TAU, window_override=WINDOW)
    ds_val_all   = WindowDataset(DATA_ROOT, "val", "all", N_EVENT_FEATURES, TAU, window_override=WINDOW)

    # Apply data fraction (subset training data)
    if DATA_FRACTION < 1.0:
        n_train = int(len(ds_train_all) * DATA_FRACTION)
        ds_train_all = torch.utils.data.Subset(ds_train_all, range(n_train))
        print(f"[DATA_FRACTION] Using {n_train:,} / {len(WindowDataset(DATA_ROOT, 'train', 'all', N_EVENT_FEATURES, TAU, window_override=WINDOW)):,} train samples")

    y_train = ds_train_all.dataset.y if isinstance(ds_train_all, torch.utils.data.Subset) else ds_train_all.y
    y_val   = ds_val_all.y

    # distributions
    tr_counts = np.bincount(y_train, minlength=3)
    va_counts = np.bincount(y_val, minlength=3)

    banner("CHECK 0: CLASS DISTRIBUTION")
    print(f"Train counts (Down/Stat/Up): {tr_counts.tolist()}")
    print(f"Val   counts (Down/Stat/Up): {va_counts.tolist()}")
    print(f"Train pct: {np.round(tr_counts / tr_counts.sum(), 4).tolist()}")
    print(f"Val   pct: {np.round(va_counts / va_counts.sum(), 4).tolist()}")

    # baselines
    banner("CHECK 1: MAJORITY & RANDOM BASELINE (TRUE MACRO)")
    base = compute_baselines(y_train, y_val)
    print(f"Majority class : {base['majority_class']}  (0=Down,1=Stat,2=Up)")
    print(f"Majority F1    : {base['majority_f1']:.4f}   (TRUE macro over [0,1,2])")
    print(f"Random F1      : {base['random_f1']:.4f}")
    print("→ A learned model should beat BOTH (especially majority).")

    # class weights
    w = None
    if USE_CLASS_WEIGHTS:
        w = class_weights_from_train(y_train).to(DEVICE)
        print("\nClass weights (normalized inverse freq):", w.detach().cpu().numpy().round(3).tolist())

    # ablation loop
    results = {}

    for feat_type, feat_name in FEATURE_CONFIGS:
        banner(f"TRAINING: {feat_name}")

        train_loader = make_loader(DATA_ROOT, "train", feat_type, shuffle=True, sampler=None, window_override=WINDOW)
        val_loader   = make_loader(DATA_ROOT, "val", feat_type, shuffle=False, sampler=None, window_override=WINDOW)

        # infer input dim
        x0, _ = next(iter(train_loader))
        input_dim = int(x0.shape[1])

        print(f"Input dim      : {input_dim}")
        print(f"Train samples  : {len(train_loader.dataset):,}")
        print(f"Val samples    : {len(val_loader.dataset):,}")

        model = SimpleMLP(input_dim).to(DEVICE)
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

        best_val = -1.0
        best_state = None
        no_improve = 0
        
        # CSV logging
        import csv
        csv_path = f"metrics_baseline_{feat_type}.csv"
        csv_file = open(csv_path, "w", newline="")
        
        for epoch in range(1, EPOCHS + 1):
            print(f"\nEpoch {epoch}/{EPOCHS}")

            tr_loss, tr_f1, tr_cm, tr_metrics = run_epoch(
                model, train_loader, criterion, optimizer, DEVICE, train=True, scheduler=scheduler, should_stop=should_stop
            )
            va_loss, va_f1, va_cm, va_metrics = run_epoch(
                model, val_loader, criterion, optimizer, DEVICE, train=False, scheduler=None, should_stop=should_stop
            )

            # Check graceful stop
            if should_stop[0]:
                print(f"Stopping training at epoch {epoch}.")
                break

            print(f"Train | loss={tr_loss:.4e} | macro-F1={tr_f1:.4f}")
            print(f"Val   | loss={va_loss:.4e} | macro-F1={va_f1:.4f}")
            print_cm(va_cm)

            # Log to CSV (write header on first epoch)
            if epoch == 1:
                fieldnames = ["epoch", "train_loss", "train_macro_f1", "val_loss", "val_macro_f1"] + list(va_metrics.keys())
                csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                csv_writer.writeheader()
            
            row = {"epoch": epoch, "train_loss": tr_loss, "train_macro_f1": tr_f1, "val_loss": va_loss, "val_macro_f1": va_f1}
            row.update(va_metrics)
            csv_writer.writerow(row)
            csv_file.flush()

            if va_f1 > best_val:
                best_val = va_f1
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (no val F1 improve for {PATIENCE} epochs)")
                break

        csv_file.close()
        print(f"✓ Metrics saved to: {csv_path}")

        if best_state is not None:
            torch.save(best_state, f"best_baseline_{feat_type}.pt")

        results[feat_type] = {"name": feat_name, "input_dim": input_dim, "best_val_f1": float(best_val)}
        print(f"\n✔ Best Val macro-F1: {best_val:.4f}")

    # summary
    banner("SUMMARY: FEATURE ABLATION vs BASELINES")
    print(f"Baselines: majority={base['majority_f1']:.4f} | random={base['random_f1']:.4f}")
    print("-" * 80)
    print(f"{'Feature Set':<30} {'InputDim':>10} {'BestF1':>10} {'GainVsMaj':>12}")
    print("-" * 80)

    for feat_type, feat_name in FEATURE_CONFIGS:
        r = results[feat_type]
        gain = r["best_val_f1"] - base["majority_f1"]
        print(f"{r['name']:<30} {r['input_dim']:>10d} {r['best_val_f1']:>10.4f} {gain:>12.4f}")

    print("-" * 80)
    print("\nInterpretation tips:")
    print("- If best_val_f1 <= majority_f1: no signal (or label imbalance dominates).")
    print("- If class weights ON boosts val macro-F1: imbalance was the main blocker.")
    print("- If LOB >> Events: orderbook carries more predictive signal (common).")
    print("- If All > max(Events, LOB): features are complementary (best sign).")

if __name__ == "__main__":
    main()