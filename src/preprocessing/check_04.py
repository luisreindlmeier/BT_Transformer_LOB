"""Step 04 validation: Check data quality, class balance, and detect train/val drift."""

from pathlib import Path
import numpy as np
import pandas as pd
import json

TICKER = "CSCO"
WINDOW = 100
TAU = 0.0003
N_CSV_SAMPLES = 500
N_SAMPLE_STATS = 50_000

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "04_windows_NEW" / TICKER
DEBUG_ROOT = DATA_ROOT / "debug"
DEBUG_ROOT.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {0: "Down", 1: "Stat", 2: "Up"}

def hard_fail(msg):
    raise RuntimeError(f"\n❌ CRITICAL: {msg}\n")

def load_split(split):
    X = np.load(DATA_ROOT / f"{split}_X.npy", mmap_mode="r")
    y = np.load(DATA_ROOT / f"{split}_y.npy", mmap_mode="r")
    return X, y

def check_1_basic_sanity(X, y, split):
    print(f"\n{'='*90}")
    print(f"[{split.upper()}] CHECK 1: BASIC SANITY")
    print(f"{'='*90}")
    
    print(f"X: {X.shape} ({X.dtype})")
    print(f"y: {y.shape} ({y.dtype})")
    
    if X.ndim != 3:
        hard_fail(f"{split}: X must be 3D")
    if X.shape[1] != WINDOW:
        hard_fail(f"{split}: Window size {X.shape[1]} != {WINDOW}")
    if len(y) != X.shape[0]:
        hard_fail(f"{split}: X/y length mismatch")
    
    # NaN/Inf check
    if not np.isfinite(X).all():
        hard_fail(f"{split}: NaN/Inf in X")
    if not np.isfinite(y).all():
        hard_fail(f"{split}: NaN/Inf in y")
    
    print("✔ Shapes OK, no NaN/Inf")


def check_2_label_distribution(y_ret_train, y_ret_val):
    print(f"\n{'='*90}")
    print(f"CHECK 2: LABEL DISTRIBUTION (Train vs Val)")
    print(f"{'='*90}")
    
    # Konvertiere y_ret zu Klassen
    def to_class(y_ret):
        y = np.ones(len(y_ret), dtype=np.int32)
        y[y_ret < -TAU] = 0
        y[y_ret > TAU] = 2
        return y
    
    y_train_class = to_class(y_ret_train)
    y_val_class = to_class(y_ret_val)
    
    print("\nTRAIN:")
    for k in range(3):
        cnt = np.sum(y_train_class == k)
        pct = 100 * cnt / len(y_train_class)
        print(f"  {CLASS_NAMES[k]:<4}: {cnt:>8} ({pct:5.2f}%)")
    
    print("\nVAL:")
    for k in range(3):
        cnt = np.sum(y_val_class == k)
        pct = 100 * cnt / len(y_val_class)
        print(f"  {CLASS_NAMES[k]:<4}: {cnt:>8} ({pct:5.2f}%)")
    
    # Drift check
    train_pcts = np.array([np.sum(y_train_class == k) / len(y_train_class) for k in range(3)])
    val_pcts = np.array([np.sum(y_val_class == k) / len(y_val_class) for k in range(3)])
    
    max_drift = np.max(np.abs(train_pcts - val_pcts))
    print(f"\nMax distribution drift: {max_drift*100:.2f}%")
    
    if max_drift > 0.10:
        print("⚠️  WARNING: >10% drift between train/val — möglicherweise temporal shift!")
    else:
        print("✔ Train/Val distribution consistent")


def check_3_feature_statistics(X_train, X_val):
    """Feature-Statistiken Train vs Val — Distribution Shift Detektion"""
    print(f"\n{'='*90}")
    print(f"CHECK 3: FEATURE STATISTICS (Train vs Val)")
    print(f"{'='*90}")
    
    # Sample für Speed
    n_train = min(N_SAMPLE_STATS, len(X_train))
    n_val = min(N_SAMPLE_STATS, len(X_val))
    
    idx_train = np.random.choice(len(X_train), n_train, replace=False)
    idx_val = np.random.choice(len(X_val), n_val, replace=False)
    
    X_train_sample = X_train[idx_train].reshape(-1, X_train.shape[-1])
    X_val_sample = X_val[idx_val].reshape(-1, X_val.shape[-1])
    
    mean_train = X_train_sample.mean(axis=0)
    mean_val = X_val_sample.mean(axis=0)
    
    std_train = X_train_sample.std(axis=0)
    std_val = X_val_sample.std(axis=0)
    
    # Mean shift
    mean_shift = np.abs(mean_train - mean_val).mean()
    print(f"Mean shift (avg): {mean_shift:.4f}")
    
    # Std shift
    std_ratio = (std_val / (std_train + 1e-8)).mean()
    print(f"Std ratio (val/train): {std_ratio:.4f}")
    
    if mean_shift > 0.5:
        print("⚠️  WARNING: Large mean shift between train/val!")
    if std_ratio < 0.7 or std_ratio > 1.3:
        print("⚠️  WARNING: Large variance shift between train/val!")
    
    print("✔ Feature statistics checked")


def check_4_class_conditional_separation(X, y_ret):
    """Class-Conditional Feature Separation — Können die Klassen getrennt werden?"""
    print(f"\n{'='*90}")
    print(f"CHECK 4: CLASS-CONDITIONAL FEATURE SEPARATION")
    print(f"{'='*90}")
    
    # Konvertiere zu Klassen
    y_class = np.ones(len(y_ret), dtype=np.int32)
    y_class[y_ret < -TAU] = 0
    y_class[y_ret > TAU] = 2
    
    # Sample für Speed
    n = min(N_SAMPLE_STATS, len(X))
    idx = np.random.choice(len(X), n, replace=False)
    
    X_sample = X[idx].reshape(-1, X.shape[-1])
    y_sample = np.repeat(y_class[idx], WINDOW)  # replicate for each timestep
    
    # Feature energy per class
    print("\nMean |feature| per class:")
    for k in range(3):
        mask = y_sample == k
        if mask.sum() == 0:
            print(f"  {CLASS_NAMES[k]:<4}: no samples")
            continue
        energy = np.mean(np.abs(X_sample[mask]))
        print(f"  {CLASS_NAMES[k]:<4}: {energy:.4f}")
    
    # Temporal change per class
    print("\nMean temporal change |X[t] - X[t-1]| per class:")
    for k in range(3):
        mask = y_class[idx] == k
        if mask.sum() == 0:
            continue
        X_diff = np.diff(X[idx][mask], axis=1)
        change = np.mean(np.abs(X_diff))
        print(f"  {CLASS_NAMES[k]:<4}: {change:.4f}")
    
    print("✔ Class-conditional analysis complete")


def check_5_temporal_leakage(X, y_ret):
    """Temporal Leakage Check — Ist die Zukunft im Fenster sichtbar?"""
    print(f"\n{'='*90}")
    print(f"CHECK 5: TEMPORAL LEAKAGE CHECK")
    print(f"{'='*90}")
    
    # Sample
    n = min(N_SAMPLE_STATS, len(X))
    idx = np.random.choice(len(X), n, replace=False)
    
    X_last = X[idx, -1, :]  # letzter Zeitschritt
    y_sample = y_ret[idx]
    
    # Korrelation zwischen letztem Zeitschritt und Label
    corr_matrix = np.corrcoef(X_last.T, y_sample)
    corr_with_label = corr_matrix[-1, :-1]
    
    max_corr = np.nanmax(np.abs(corr_with_label))
    
    print(f"Max |corr(feature[t_end], y_ret)| = {max_corr:.4f}")
    
    if max_corr > 0.8:
        hard_fail("Suspiciously high correlation → LEAKAGE DETECTED!")
    elif max_corr > 0.5:
        print("⚠️  WARNING: High correlation — check feature construction!")
    else:
        print("✔ No obvious leakage")


def check_6_csv_export(X_train, y_train, X_val, y_val):
    """CSV Export für manuelle Inspektion"""
    print(f"\n{'='*90}")
    print(f"CHECK 6: CSV EXPORT (Manual Inspection)")
    print(f"{'='*90}")
    
    def export_split(X, y, split_name):
        n = min(N_CSV_SAMPLES, len(X))
        idx = np.random.choice(len(X), n, replace=False)
        
        rows = []
        for i in idx:
            row = {"y_ret": float(y[i])}
            
            # Klasse
            if y[i] < -TAU:
                row["class"] = "Down"
            elif y[i] <= TAU:
                row["class"] = "Stat"
            else:
                row["class"] = "Up"
            
            # Fenster-Statistiken
            row["x_mean"] = float(X[i].mean())
            row["x_std"] = float(X[i].std())
            row["x_min"] = float(X[i].min())
            row["x_max"] = float(X[i].max())
            
            # Erste + letzte Features (erste 5)
            for j in range(min(5, X.shape[-1])):
                row[f"x_first_feat{j}"] = float(X[i, 0, j])
                row[f"x_last_feat{j}"] = float(X[i, -1, j])
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        out_path = DEBUG_ROOT / f"preview_{split_name}.csv"
        df.to_csv(out_path, index=False)
        print(f"  ✔ {split_name}: {out_path}")
        return out_path
    
    train_csv = export_split(X_train, y_train, "train")
    val_csv = export_split(X_val, y_val, "val")
    
    print(f"\n✔ CSV exports ready for manual inspection!")
    print(f"  Öffne in Excel/Numbers und checke:")
    print(f"  - Sind die y_ret Werte realistisch?")
    print(f"  - Unterscheiden sich die Klassen in den Features?")
    print(f"  - Gibt es auffällige Muster?")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 90)
    print("CHECK 05 — KRITISCHE VALIDIERUNG (Overfitting Prevention)")
    print("=" * 90)
    
    # Load data
    print("\nLoading data...")
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    
    print(f"  Train: {len(X_train):,} samples")
    print(f"  Val:   {len(X_val):,} samples")
    
    # Run checks
    check_1_basic_sanity(X_train, y_train, "train")
    check_1_basic_sanity(X_val, y_val, "val")
    
    check_2_label_distribution(y_train, y_val)
    
    check_3_feature_statistics(X_train, X_val)
    
    check_4_class_conditional_separation(X_train, y_train)
    
    check_5_temporal_leakage(X_train, y_train)
    
    check_6_csv_export(X_train, y_train, X_val, y_val)
    
    # Summary
    print("\n" + "=" * 90)
    print("✔ ALL CHECKS PASSED")
    print("=" * 90)
    print("\nNext steps:")
    print("  1. Check CSV files in", DEBUG_ROOT)
    print("  2. If OK → Start training")
    print("  3. Expected: Val macro-F1 > 0.40")
    print("=" * 90)

if __name__ == "__main__":
    main()
