"""Step 01: Transform raw LOBSTER data to preprocessed features and labels."""

from __future__ import annotations

from pathlib import Path
import os
import json
import numpy as np
import pandas as pd

TICKER = os.getenv("TICKER", "CSCO")

# Event-based horizon (in number of events)
HORIZON_EVENTS = 10
TICK_SIZE = 100.0
DROP_ZERO_RET = False
PRICE_CLIP_TICKS = 300

# Robust day-reset detection: if time gap > 8h, treat as new trading session
MAX_INTRA_SESSION_SEC = 8 * 3600

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "NASDAQ" / TICKER
OUT_ROOT = Path.home() / "thesis_output" / "01_preprocessed"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

EPS = 1e-12

RAW_EVENT_COLS = [
    "time",
    "event_type",
    "order_id",
    "size",
    "price",
    "direction",
]


def build_orderbook_columns(n_levels: int) -> list[str]:
    cols: list[str] = []
    for k in range(1, n_levels + 1):
        cols.extend([
            f"ask_price_{k}",
            f"ask_size_{k}",
            f"bid_price_{k}",
            f"bid_size_{k}",
        ])
    return cols

def safe_log_dt(time_s: pd.Series) -> np.ndarray:
    t = time_s.to_numpy(dtype=np.float64, copy=False)
    dt = np.empty_like(t, dtype=np.float64)
    dt[0] = 0.0
    dt[1:] = t[1:] - t[:-1]
    dt = np.where(dt < 0, 0.0, dt)
    return np.log(dt + 1e-6).astype(np.float32)


def one_hot_event_type(ev_type: pd.Series) -> pd.DataFrame:
    oh = pd.get_dummies(ev_type.astype(np.int32), prefix="ev")
    # stable ev_1..ev_6
    for k in range(1, 7):
        col = f"ev_{k}"
        if col not in oh.columns:
            oh[col] = 0
    return oh[[f"ev_{k}" for k in range(1, 7)]].astype(np.float32)


def compute_mid_from_raw_ob(ob_raw: pd.DataFrame) -> np.ndarray:
    ask1 = ob_raw["ask_price_1"].to_numpy(dtype=np.float64, copy=False)
    bid1 = ob_raw["bid_price_1"].to_numpy(dtype=np.float64, copy=False)
    return 0.5 * (ask1 + bid1)


def signed_log(x: np.ndarray) -> np.ndarray:
    """
    signed log transform: sign(x) * log1p(|x|)
    """
    s = np.sign(x)
    return (s * np.log1p(np.abs(x))).astype(np.float32)

def preprocess_events(events_raw: pd.DataFrame, mid_raw: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame(index=events_raw.index)

    out["log_dt"] = safe_log_dt(events_raw["time"])
    # keep absolute time (seconds since midnight) for time-grid resampling later
    out["time_s"] = events_raw["time"].astype(np.float32)

    size = events_raw["size"].to_numpy(dtype=np.float64, copy=False)
    direction = events_raw["direction"].to_numpy(dtype=np.float64, copy=False)
    signed_size = size * direction
    out["signed_size"] = signed_log(signed_size)

    price = events_raw["price"].to_numpy(dtype=np.float64, copy=False)
    rel_price = (price - mid_raw) / (mid_raw + EPS)
    out["rel_event_price"] = rel_price.astype(np.float32)  # No clipping for consistency

    out = pd.concat([out, one_hot_event_type(events_raw["event_type"])], axis=1)
    return out.astype(np.float32)


def preprocess_orderbook(ob_raw: pd.DataFrame, mid_raw: np.ndarray, n_levels: int) -> pd.DataFrame:
    out = pd.DataFrame(index=ob_raw.index)

    for k in range(1, n_levels + 1):
        ask_p = ob_raw[f"ask_price_{k}"].to_numpy(dtype=np.float64)
        bid_p = ob_raw[f"bid_price_{k}"].to_numpy(dtype=np.float64)

        out[f"ask_dist_{k}"] = np.clip(
            (ask_p - mid_raw) / TICK_SIZE,
            -PRICE_CLIP_TICKS,
            PRICE_CLIP_TICKS
        ).astype(np.float32)

        out[f"bid_dist_{k}"] = np.clip(
            (bid_p - mid_raw) / TICK_SIZE,
            -PRICE_CLIP_TICKS,
            PRICE_CLIP_TICKS
        ).astype(np.float32)

        out[f"ask_log_size_{k}"] = np.log1p(
            ob_raw[f"ask_size_{k}"].to_numpy(dtype=np.float64)
        ).astype(np.float32)

        out[f"bid_log_size_{k}"] = np.log1p(
            ob_raw[f"bid_size_{k}"].to_numpy(dtype=np.float64)
        ).astype(np.float32)

    return out

def build_labels_and_mask(mid_raw: np.ndarray, time_s: pd.Series) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Build y_ret (event-based horizon) with day-reset safety.
    y_ret[t] = log(mid[t+H]) - log(mid[t]) where H = HORIZON_EVENTS
    Avoids cross-day leakage by detecting trading session resets.
    """
    N = len(mid_raw)
    h = int(HORIZON_EVENTS)
    
    # Event-based labels: look H events ahead
    y = np.full(N, np.nan, dtype=np.float32)
    if h > 0 and h < N:
        mid_now = mid_raw[:-h].astype(np.float64, copy=False)
        mid_fut = mid_raw[h:].astype(np.float64, copy=False)
        y_val = (np.log(mid_fut + EPS) - np.log(mid_now + EPS)).astype(np.float32)
        y[:N - h] = y_val
    
    # Detect day resets: time goes backward or jumps > 8h (new session)
    valid = np.isfinite(y).copy()
    if N > 1:
        t = time_s.to_numpy(dtype=np.float64, copy=False)
        dt = t[1:] - t[:-1]
        is_reset = (dt < 0) | (dt > MAX_INTRA_SESSION_SEC)
        reset_indices = np.where(is_reset)[0] + 1  # +1 to align with time array
        
        # Invalidate labels that cross a reset boundary
        for reset_idx in reset_indices:
            # Invalidate [reset_idx - h, reset_idx) to avoid cross-boundary labels
            start_invalid = max(0, reset_idx - h)
            y[start_invalid:reset_idx] = np.nan
            valid[start_invalid:reset_idx] = False
    
    if DROP_ZERO_RET:
        valid &= (y != 0.0)
    
    labels = pd.DataFrame({"y_ret": y})
    return labels, valid

def main() -> None:
    print("=" * 90)
    print("STEP 01 — PREPROCESS RAW LOBSTER (EVENT-BASED, TRAINING-SAFE)")
    print(f"TICKER         : {TICKER}")
    print(f"HORIZON_EVENTS : {HORIZON_EVENTS}")
    print(f"TICK_SIZE      : {TICK_SIZE}")
    print(f"DROP_ZERO_RET  : {DROP_ZERO_RET}")
    print("=" * 90)

    msg_files = sorted(p for p in DATA_ROOT.iterdir() if "message" in p.name and not p.name.startswith("."))
    ob_files = sorted(p for p in DATA_ROOT.iterdir() if "orderbook" in p.name and not p.name.startswith("."))
    if not msg_files or not ob_files:
        raise RuntimeError(f"Missing message/orderbook files in {DATA_ROOT}")

    print("\nLoading and preprocessing files sequentially to save memory...")
    
    # Determine orderbook depth from first file
    sample = pd.read_csv(ob_files[0], nrows=1, header=None, encoding="latin1")
    n_levels = sample.shape[1] // 4
    if n_levels <= 0:
        raise RuntimeError("Invalid orderbook depth detected")
    ob_cols = build_orderbook_columns(n_levels)
    
    # Process files one at a time to avoid loading everything into memory
    processed_chunks = []
    total_rows_before = 0
    total_rows_after = 0
    
    for i, (msg_f, ob_f) in enumerate(zip(msg_files, ob_files), 1):
        print(f"  Processing file {i}/{len(msg_files)}: {msg_f.name}...")
        
        # Load and clean events
        events_chunk = pd.read_csv(msg_f, header=None, names=RAW_EVENT_COLS, encoding="latin1")
        n_before = len(events_chunk)
        events_chunk = events_chunk.dropna(subset=["time", "event_type", "size", "price", "direction"])
        events_chunk = events_chunk.reset_index(drop=True)  # FIX: Reset indices after dropna
        events_chunk = events_chunk.drop(columns=["order_id"])
        events_chunk["time"] = events_chunk["time"].astype(np.float64)
        events_chunk["event_type"] = events_chunk["event_type"].astype(np.int32)
        events_chunk["size"] = events_chunk["size"].astype(np.float64)
        events_chunk["price"] = events_chunk["price"].astype(np.float64)
        events_chunk["direction"] = events_chunk["direction"].astype(np.int32)
        
        # Load and clean orderbook
        ob_chunk = pd.read_csv(ob_f, header=None, names=ob_cols, encoding="latin1")
        ob_chunk = ob_chunk.dropna()
        ob_chunk = ob_chunk.reset_index(drop=True)  # FIX: Reset indices after dropna
        
        # Ensure alignment: take min length
        n_after = min(len(events_chunk), len(ob_chunk))
        if n_after < max(len(events_chunk), len(ob_chunk)):
            events_chunk = events_chunk.iloc[:n_after].reset_index(drop=True)
            ob_chunk = ob_chunk.iloc[:n_after].reset_index(drop=True)
        
        total_rows_before += n_before
        total_rows_after += n_after
        
        # Compute mid and preprocess this chunk
        mid_chunk = compute_mid_from_raw_ob(ob_chunk)
        labels_chunk, mask_chunk = build_labels_and_mask(mid_chunk, events_chunk["time"])
        
        events_processed = preprocess_events(events_chunk, mid_chunk)
        ob_processed = preprocess_orderbook(ob_chunk, mid_chunk, n_levels=n_levels)
        
        # Apply mask and save chunk
        chunk_data = {
            'events': events_processed.loc[mask_chunk].reset_index(drop=True),
            'orderbook': ob_processed.loc[mask_chunk].reset_index(drop=True),
            'labels': labels_chunk.loc[mask_chunk].reset_index(drop=True)
        }
        processed_chunks.append(chunk_data)
        
        # Free memory
        del events_chunk, ob_chunk, mid_chunk, labels_chunk, mask_chunk, events_processed, ob_processed
    
    if total_rows_before > total_rows_after:
        print(f"\n  Total dropped rows: {total_rows_before - total_rows_after:,} ({100*(total_rows_before-total_rows_after)/total_rows_before:.2f}%)")
    
    # Concatenate all processed chunks
    print("\nConcatenating processed chunks...")
    events = pd.concat([c['events'] for c in processed_chunks], ignore_index=True)
    orderbook = pd.concat([c['orderbook'] for c in processed_chunks], ignore_index=True)
    labels = pd.concat([c['labels'] for c in processed_chunks], ignore_index=True)

    # Save to parquet
    events_path = OUT_ROOT / f"events_{TICKER}.parquet"
    ob_path = OUT_ROOT / f"orderbook_{TICKER}.parquet"
    y_path = OUT_ROOT / f"labels_{TICKER}.parquet"
    manifest = OUT_ROOT / f"manifest_{TICKER}.json"

    events.to_parquet(events_path, index=False)
    orderbook.to_parquet(ob_path, index=False)
    labels.to_parquet(y_path, index=False)

    meta = {
        "ticker": TICKER,
        "rows": int(len(events)),
        "horizon_events": (None if HORIZON_MS is not None else int(HORIZON_EVENTS)),
        "horizon_ms": (list(HORIZON_MS) if isinstance(HORIZON_MS, tuple) else (int(HORIZON_MS) if HORIZON_MS is not None else None)),
        "horizon_ms_mode": (HORIZON_MS_MODE if HORIZON_MS is not None else None),
        "tick_size": float(TICK_SIZE),
        "drop_zero_ret": bool(DROP_ZERO_RET),
        "event_features": list(events.columns),
        "orderbook_features": list(orderbook.columns),
        "label_columns": list(labels.columns),
        "column_semantics": {
            "log_dt": "log(time[t]-time[t-1] + 1e-6), day-reset safe",
            "signed_size": "sign(size*direction) * log1p(|size*direction|)",
            "rel_event_price": "(event_price - mid_price)/mid_price, clipped",
            "ev_1..ev_6": "one-hot event type",
            "ask_dist_k/bid_dist_k": "(ask/bid price at level k - mid) / tick_size",
            "ask_log_size_k/bid_log_size_k": "log1p(size) at level k",
            "y_ret": "log(mid[t+H]) - log(mid[t])",
        },
        "notes": [
            "Event-based horizon only (HORIZON_EVENTS).",
            "No z-score normalization (avoid data leakage from future stats).",
            "No clipping of rel_event_price (consistency with LOB features).",
            "Day-reset detection: time jumps > 8h treated as new session.",
            "Indices reset after dropna to prevent drift.",
        ],
    }
    manifest.write_text(json.dumps(meta, indent=2))

    print("\nSaved:")
    print(f"  events    -> {events_path}")
    print(f"  orderbook -> {ob_path}")
    print(f"  labels    -> {y_path}")
    print(f"  manifest  -> {manifest}")
    print(f"\nFinal rows: {len(events):,}")
    print("\nDONE ✔")

if __name__ == "__main__":
    main()