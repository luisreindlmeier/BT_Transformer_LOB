"""Baseline MLP training on prebuilt windows with feature ablation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import f1_score, confusion_matrix

import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.simple_mlp import SimpleMLP

DEVICE = (
    "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else
    ("cuda" if torch.cuda.is_available() else "cpu")
)

TICKER = "CSCO"
DATA_ROOT = Path("data/04_windows_NEW") / TICKER

WINDOW = None
TAU = None
AUTO_TAU_TARGET_STAT_PCT = 0.6
N_EVENT_FEATURES = 9

BATCH_SIZE = 512 if DEVICE in ("mps", "cuda") else 256
EPOCHS = 1
LR = 1e-3

NUM_WORKERS = 0
PIN_MEMORY = False

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
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )

# -----------------------------------------------------------------------------
# TRAIN / EVAL
# -----------------------------------------------------------------------------

def run_epoch(model, loader, criterion, optimizer, device: str, train: bool):
    if train:
        model.train()
    else:
        model.eval()

    all_t = []
    all_p = []
    losses = []

    t0 = time.time()

    for step, (X, y) in enumerate(loader, 1):
        X = X.to(device, non_blocking=False)
        y = y.to(device, non_blocking=False)

        if train:
            optimizer.zero_grad(set_to_none=True)

        # autocast: CUDA ok; MPS autocast is sometimes flaky — keep it off for stability
        logits = model(X)
        loss = criterion(logits, y)

        if train:
            loss.backward()
            optimizer.step()

        losses.append(float(loss.item()))
        preds = torch.argmax(logits, dim=1)

        all_t.append(y.detach().cpu().numpy())
        all_p.append(preds.detach().cpu().numpy())

        if step % 200 == 0:
            dt = time.time() - t0
            print(f"  [{step:>5}] avg_loss={np.mean(losses[-200:]):.4e} | elapsed={dt:.1f}s")

    y_true = np.concatenate(all_t).astype(np.int64)
    y_pred = np.concatenate(all_p).astype(np.int64)

    f1 = macro_f1_true(y_true, y_pred)
    cm = cm_true(y_true, y_pred)
    return float(np.mean(losses)), f1, cm

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
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

    y_train = ds_train_all.y
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

    # build a sampler to rebalance train split towards ~20/60/20
    target_pct = np.array([0.2, 0.6, 0.2], dtype=np.float64)
    current_pct = tr_counts / tr_counts.sum()
    weights = np.zeros_like(y_train, dtype=np.float64)
    per_class_weight = target_pct / np.maximum(current_pct, 1e-12)
    for cls, w_cls in enumerate(per_class_weight):
        weights[y_train == cls] = w_cls
    weights = weights / weights.mean()  # normalize for stability
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.double),
        num_samples=len(y_train),
        replacement=True,
    )

    for feat_type, feat_name in FEATURE_CONFIGS:
        banner(f"TRAINING: {feat_name}")

        train_loader = make_loader(DATA_ROOT, "train", feat_type, shuffle=True, sampler=sampler, window_override=WINDOW)
        val_loader   = make_loader(DATA_ROOT, "val", feat_type, shuffle=False, sampler=None, window_override=WINDOW)

        # infer input dim
        x0, _ = next(iter(train_loader))
        input_dim = int(x0.shape[1])

        print(f"Input dim      : {input_dim}")
        print(f"Train samples  : {len(train_loader.dataset):,}")
        print(f"Val samples    : {len(val_loader.dataset):,}")

        model = SimpleMLP(input_dim).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        criterion = nn.CrossEntropyLoss(weight=w) if w is not None else nn.CrossEntropyLoss()

        best_val = -1.0
        for epoch in range(1, EPOCHS + 1):
            print(f"\nEpoch {epoch}/{EPOCHS}")

            tr_loss, tr_f1, tr_cm = run_epoch(model, train_loader, criterion, optimizer, DEVICE, train=True)
            va_loss, va_f1, va_cm = run_epoch(model, val_loader, criterion, optimizer, DEVICE, train=False)

            print(f"Train | loss={tr_loss:.4e} | macro-F1={tr_f1:.4f}")
            print(f"Val   | loss={va_loss:.4e} | macro-F1={va_f1:.4f}")
            print_cm(va_cm)

            best_val = max(best_val, va_f1)

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