"""
analyze_event_frequency.py

Calculate average event frequency (events per second/millisecond).
Uses raw LOBSTER data to determine temporal resolution.

Goal: Understand what HORIZON_EVENTS=1000 means in real time.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

TICKER = "CSCO"
PROJECT_ROOT = Path(__file__).resolve().parents[0]
DATA_ROOT = PROJECT_ROOT / "data" / "NASDAQ" / TICKER

print("=" * 90)
print("EVENT FREQUENCY ANALYSIS")
print(f"TICKER: {TICKER}")
print("=" * 90)

# Load message files (raw events)
msg_files = sorted([p for p in DATA_ROOT.iterdir() if "message" in p.name])
if not msg_files:
    raise FileNotFoundError(f"No message files found in {DATA_ROOT}")

print(f"\nFound {len(msg_files)} message files")
print(f"Loading first file for analysis: {msg_files[0].name}")

# Load just first file for speed (representative sample)
raw_cols = ["time", "event_type", "order_id", "size", "price", "direction"]
events = pd.read_csv(msg_files[0], header=None, names=raw_cols)
print(f"  Loaded {len(events):,} events from first file")

times = events["time"].to_numpy(dtype=np.float64)

# Calculate time deltas (in seconds)
dt = np.diff(times)
dt = dt[dt > 0]  # remove day-reset artifacts (negative jumps)

# =========================================================================
# STATISTICS
# =========================================================================

print("\n" + "=" * 90)
print("TIME DELTA STATISTICS (seconds)")
print("=" * 90)

print(f"\nCount:            {len(dt):,} intervals")
print(f"Mean:             {np.mean(dt):.6f} s  = {np.mean(dt)*1000:.3f} ms")
print(f"Median:           {np.median(dt):.6f} s  = {np.median(dt)*1000:.3f} ms")
print(f"Std Dev:          {np.std(dt):.6f} s  = {np.std(dt)*1000:.3f} ms")
print(f"Min:              {np.min(dt):.6f} s  = {np.min(dt)*1000:.3f} ms")
print(f"Max:              {np.max(dt):.6f} s  = {np.max(dt)*1000:.3f} ms")
print(f"95th percentile:  {np.percentile(dt, 95):.6f} s  = {np.percentile(dt, 95)*1000:.3f} ms")

# =========================================================================
# EVENTS PER SECOND
# =========================================================================

events_per_sec = 1.0 / np.mean(dt)
print(f"\nEvents per second: {events_per_sec:.1f} events/sec")
print(f"Events per ms:     {events_per_sec/1000:.6f} events/ms")

# =========================================================================
# HORIZON ANALYSIS
# =========================================================================

HORIZON_EVENTS_LIST = [100, 500, 1000, 2000, 5000]

print("\n" + "=" * 90)
print("HORIZON DURATION (what does N events mean in real time?)")
print("=" * 90)

for h in HORIZON_EVENTS_LIST:
    # Approximate: h events * mean_dt_per_event
    duration_s = h * np.mean(dt)
    duration_ms = duration_s * 1000
    
    print(f"\n{h:,} events:")
    print(f"  ≈ {duration_s:.3f} sec")
    print(f"  ≈ {duration_ms:.1f} ms")

# =========================================================================
# COMPARE TO PAPER
# =========================================================================

print("\n" + "=" * 90)
print("COMPARISON TO PAPER (300ms-1000ms horizons)")
print("=" * 90)

paper_horizons_ms = [300, 500, 1000]
for h_ms in paper_horizons_ms:
    h_events = h_ms / (np.mean(dt) * 1000)
    print(f"\nPaper horizon {h_ms}ms = ~{h_events:.0f} events")

# =========================================================================
# DISTRIBUTION HISTOGRAM
# =========================================================================

print("\n" + "=" * 90)
print("TIME DELTA DISTRIBUTION (histogram)")
print("=" * 90)

hist, bins = np.histogram(dt * 1000, bins=50)
print(f"\nMin dt:  {np.min(dt*1000):.3f} ms")
print(f"Max dt:  {np.max(dt*1000):.3f} ms")
print(f"\nMost common intervals (top 5):")
for i in np.argsort(hist)[-5:][::-1]:
    bin_start = bins[i]
    bin_end = bins[i+1]
    count = hist[i]
    pct = 100 * count / len(dt)
    print(f"  {bin_start:.3f}-{bin_end:.3f} ms: {count:,} events ({pct:.1f}%)")

print("\n" + "=" * 90)
print("✔ ANALYSIS COMPLETE")
print("=" * 90)
