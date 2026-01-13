"""
Quick check: Class distribution for train/val after tau thresholds
"""
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "04_windows" / "CSCO"

# Same tau as in train_baseline.py
TAU = 0.0003

def check_split(split):
    y_path = DATA_ROOT / f"{split}_y.npy"
    y = np.load(y_path)
    
    # Apply tau thresholds (same logic as dataset.py)
    labels = np.zeros(len(y), dtype=np.int64)
    labels[y > TAU] = 2   # Up
    labels[y < -TAU] = 0  # Down
    labels[np.abs(y) <= TAU] = 1  # Stationary
    
    # Count distribution
    counts = np.bincount(labels, minlength=3)
    total = len(labels)
    
    print(f"\n{split.upper()} ({total:,} samples)")
    print("=" * 60)
    print(f"  Down (0):       {counts[0]:>10,}  ({100*counts[0]/total:>5.2f}%)")
    print(f"  Stationary (1): {counts[1]:>10,}  ({100*counts[1]/total:>5.2f}%)")
    print(f"  Up (2):         {counts[2]:>10,}  ({100*counts[2]/total:>5.2f}%)")
    print("=" * 60)
    
    return counts, total

if __name__ == "__main__":
    print("=" * 60)
    print(f"CLASS DISTRIBUTION CHECK (TAU = {TAU})")
    print("=" * 60)
    
    train_counts, train_total = check_split("train")
    val_counts, val_total = check_split("val")
    
    print("\nSUMMARY")
    print("=" * 60)
    print(f"Train balance: {100*max(train_counts)/train_total:.2f}% majority class")
    print(f"Val   balance: {100*max(val_counts)/val_total:.2f}% majority class")
    print(f"\nIdeal for 3-class: ~33.3% each")
    print(f"Current train: Down={100*train_counts[0]/train_total:.2f}%, Stat={100*train_counts[1]/train_total:.2f}%, Up={100*train_counts[2]/train_total:.2f}%")
