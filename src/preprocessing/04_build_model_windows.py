"""Step 04: Build fixed-length sliding windows for sequence models."""

from pathlib import Path
import os
import json
import time
import gc
import numpy as np
import pandas as pd
import glob
try:
    import pyarrow.parquet as pq
    import pyarrow as pa
except Exception:
    pq = None
    pa = None

TICKER = os.getenv("TICKER", "CSCO")
TIME_GRID_MS = 10
HORIZON_MS = 1000
WINDOW = HORIZON_MS // TIME_GRID_MS
DTYPE = np.float32
CHUNK_SIZE = 25_000

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IN_ROOT = Path.home() / "thesis_output" / "03_normalized_NEW"
OUT_ROOT = Path.home() / "thesis_output" / "04_windows_NEW" / TICKER
OUT_ROOT.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "val"]

def log_progress(i: int, total: int, start_time: float, every: int = 100_000):
    if i % every == 0 and i > 0:
        elapsed = time.time() - start_time
        rate = i / elapsed
        remaining_s = (total - i) / rate
        print(
            f"  [{i:,}/{total:,}] "
            f"({100*i/total:.1f}%) | "
            f"elapsed {elapsed/60:.1f} min | "
            f"ETA {remaining_s/60:.1f} min"
        )

def _build_from_normalized(split: str) -> tuple:
    print(f"\nBuilding windows for {split}...")

    BASE = IN_ROOT / TICKER

    events = pd.read_parquet(BASE / split / "events.parquet")
    lob    = pd.read_parquet(BASE / split / "orderbook.parquet")
    labels = pd.read_parquet(BASE / split / "labels.parquet")

    assert len(events) == len(lob) == len(labels), "Alignment error"

    events = events.reset_index(drop=True)
    lob = lob.reset_index(drop=True)
    labels = labels.reset_index(drop=True)
    print(f"  Using full data: {len(events):,} samples")

    # Combine features; keep time_s separately for resampling
    assert "time_s" in events.columns, "Step 01 must retain 'time_s' for time-grid windows"
    feature_df = pd.concat([events.drop(columns=["time_s"]), lob], axis=1)
    feature_names = list(feature_df.columns)
    X_feat = feature_df.to_numpy(dtype=DTYPE)
    t = events["time_s"].to_numpy(dtype=np.float64)
    y = labels["y_ret"].to_numpy(dtype=DTYPE)

    # Release pandas frames early to save RAM when using float32
    del events, lob, labels
    gc.collect()

    n = len(X_feat)
    n_features = X_feat.shape[1]
    # valid samples determined by label availability (Step 01 already guards tail/day reset)
    valid = np.isfinite(y)
    idxs = np.nonzero(valid)[0]
    n_samples = len(idxs)

    # Pre-allocate output files
    X_path = OUT_ROOT / f"{split}_X.npy"
    y_path = OUT_ROOT / f"{split}_y.npy"
    
    # Create memory-mapped files for streaming writes
    X_out = np.lib.format.open_memmap(
        X_path, mode='w+', dtype=DTYPE,
        shape=(n_samples, WINDOW, n_features)
    )
    y_out = np.lib.format.open_memmap(
        y_path, mode='w+', dtype=DTYPE,
        shape=(n_samples,)
    )

    # -------------------------------------------------------------------------
    # FAST PATH: precompute day boundaries + vectorized search via np.searchsorted
    # -------------------------------------------------------------------------
    # Day ids and boundaries computed once
    day_id = np.zeros(n, dtype=np.int32)
    if n > 1:
        dt = t[1:] - t[:-1]
        resets_mask = (dt < 0)
        day_id[1:] = np.cumsum(resets_mask.astype(np.int32))
        resets_idx = np.flatnonzero(resets_mask) + 1  # start index of new days
        day_starts = np.concatenate(([0], resets_idx))
        day_ends = np.concatenate((resets_idx, [n]))
    else:
        day_starts = np.array([0], dtype=np.int32)
        day_ends = np.array([n], dtype=np.int32)

    # Fixed grid offsets in seconds
    grid_offsets = (np.arange(WINDOW, dtype=np.float64) * (TIME_GRID_MS / 1000.0))

    # Chunked sliding window writing directly into memmaps
    start_time = time.time()

    for chunk_start in range(0, n_samples, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, n_samples)

        # Cache per-day segment slices
        last_day = -1
        seg_start = seg_end = None
        t_seg = None
        X_seg = None

        for local_i, sample_idx in enumerate(idxs[chunk_start:chunk_end]):
            d = day_id[sample_idx]
            if d != last_day:
                seg_start = int(day_starts[d])
                seg_end = int(day_ends[d])
                t_seg = t[seg_start:seg_end]
                X_seg = X_feat[seg_start:seg_end]
                last_day = d

            t0 = t[sample_idx]
            grid = t0 + grid_offsets  # (WINDOW,)

            # positions of last observation <= grid
            pos = np.searchsorted(t_seg, grid, side='right') - 1
            pos = np.clip(pos, 0, len(t_seg) - 1).astype(np.int64)

            # Direct write into memmap
            X_out[chunk_start + local_i] = X_seg[pos]
            y_out[chunk_start + local_i] = y[sample_idx]

        X_out.flush()
        y_out.flush()

        log_progress(chunk_end, n_samples, start_time)

    elapsed = time.time() - start_time
    throughput = n_samples / elapsed

    print(f"✔ {n_samples:,} samples in {elapsed/60:.1f} min ({throughput:.0f} samples/sec)")

    return (n_samples, WINDOW, n_features), (n_samples,), feature_names


def _build_from_step04_chunks(split: str) -> tuple:
    print(f"\nBuilding windows for {split} from STEP 03 chunks...")

    chunks = sorted(glob.glob(str(PROJECT_ROOT / "data" / "03_windows" / f"{split}_{TICKER}_chunk_*.parquet")))
    if not chunks:
        raise FileNotFoundError(f"No STEP 03 chunk files found for split '{split}' at data/03_windows/{split}_{TICKER}_chunk_*.parquet")

    if pq is None:
        raise ImportError("pyarrow is required to read STEP 04 parquet chunks. Please install pyarrow.")

    # Inspect first chunk to infer shapes
    first_tbl = pq.read_table(chunks[0], columns=["X", "y"])
    X0 = first_tbl["X"].to_pylist()[0]
    win = len(X0)
    feat = len(X0[0]) if win > 0 else 0

    if win != WINDOW:
        print(f"  Note: WINDOW constant ({WINDOW}) differs from chunk window ({win}). Using chunk window {win}.")
        win = win

    # Compute total samples
    total = 0
    for ch in chunks:
        total += pq.read_table(ch, columns=["y"]).num_rows

    # Pre-allocate memmaps
    X_path = OUT_ROOT / f"{split}_X.npy"
    y_path = OUT_ROOT / f"{split}_y.npy"
    X_out = np.lib.format.open_memmap(X_path, mode="w+", dtype=DTYPE, shape=(total, win, feat))
    y_out = np.lib.format.open_memmap(y_path, mode="w+", dtype=DTYPE, shape=(total,))

    start_time = time.time()
    idx = 0
    for ch_i, ch in enumerate(chunks):
        tbl = pq.read_table(ch, columns=["X", "y"])
        
        # Use PyArrow directly (faster than pandas for nested lists)
        X_lists = tbl["X"].to_pylist()  # avoid pandas overhead
        y_arr = tbl["y"].to_pylist()
        
        # Vectorized conversion: stack all X lists at once (faster than per-item)
        X_batch = np.array(X_lists, dtype=DTYPE)
        y_batch = np.array(y_arr, dtype=DTYPE)

        X_out[idx:idx+len(X_batch)] = X_batch
        y_out[idx:idx+len(y_batch)] = y_batch
        idx += len(X_batch)

        log_progress(idx, total, start_time, every=max(100_000, CHUNK_SIZE))

    print(f"✔ {total:,} samples written from {len(chunks)} chunks")
    return (total, win, feat), (total,), None


def build_windows(split: str) -> tuple:
    """Dispatch: prefer normalized files if present, else use STEP 04 chunks."""
    base_dir = IN_ROOT / TICKER / split
    if (base_dir / "events.parquet").exists() and (base_dir / "orderbook.parquet").exists() and (base_dir / "labels.parquet").exists():
        return _build_from_normalized(split)
    else:
        return _build_from_step04_chunks(split)

def main():
    print("=" * 90)
    print("STEP 04 — BUILD MODEL WINDOWS (CHUNKED, MEMORY-SAFE)")
    print(f"TICKER : {TICKER}")
    print(f"WINDOW : {WINDOW}")
    print(f"CHUNK_SIZE : {CHUNK_SIZE:,}")
    print("=" * 90)

    meta = {
        "ticker": TICKER,
        "window": WINDOW,
        "chunk_size": CHUNK_SIZE,
        "splits": {},
    }

    feature_names = None
    for split in SPLITS:
        X_shape, y_shape, feature_names = build_windows(split)
        meta["splits"][split] = {
            "X_shape": X_shape,
            "y_shape": y_shape,
        }

    meta["feature_order"] = feature_names

    # Save meta
    with open(OUT_ROOT / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\nMeta written:")
    print(OUT_ROOT / "meta.json")
    print("\nSTEP 04 COMPLETE ✔")

if __name__ == "__main__":
    main()