"""
Analyze actual y_ret distribution to understand imbalance root cause
"""
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "04_windows" / "CSCO"

def analyze_returns(split):
    y_path = DATA_ROOT / f"{split}_y.npy"
    y = np.load(y_path)
    
    print(f"\n{split.upper()} - Y_RET STATISTICS")
    print("=" * 70)
    print(f"  Count:      {len(y):>12,}")
    print(f"  Mean:       {np.mean(y):>12.6f}")
    print(f"  Std:        {np.std(y):>12.6f}")
    print(f"  Min:        {np.min(y):>12.6f}")
    print(f"  Max:        {np.max(y):>12.6f}")
    print(f"  Median:     {np.median(y):>12.6f}")
    print()
    print("  Percentiles:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"    {p:>2}%:      {np.percentile(y, p):>12.6f}")
    
    # Count how many exceed various thresholds
    print(f"\n  Counts exceeding thresholds:")
    for tau in [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]:
        up = np.sum(y > tau)
        down = np.sum(y < -tau)
        stat = len(y) - up - down
        print(f"    TAU={tau:.4f}: Down={100*down/len(y):>5.2f}%, Stat={100*stat/len(y):>5.2f}%, Up={100*up/len(y):>5.2f}%")
    
    return y

if __name__ == "__main__":
    print("=" * 70)
    print("Y_RET DISTRIBUTION ANALYSIS")
    print("=" * 70)
    
    train_y = analyze_returns("train")
    val_y = analyze_returns("val")
    
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print("If you want ~20% up/down each (40% total non-stationary):")
    print("  → Look at the 20th and 80th percentiles above")
    print("  → Set TAU to that value")
    print()
    print("OR: Increase HORIZON_EVENTS in STEP 01 preprocessing")
    print("    (currently likely 100 → try 500-2000)")
    print("    This will make returns larger and more meaningful.")
