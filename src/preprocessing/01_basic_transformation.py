"""Step 01: Transform raw LOBSTER data to preprocessed features and labels."""

from __future__ import annotations

from pathlib import Path
import os
import json
import numpy as np
import pandas as pd

TICKER = os.getenv("TICKER", "CSCO")

HORIZON_MS: int | tuple[int, int] | None = None
HORIZON_MS_MODE: str = "upper"
HORIZON_EVENTS = 10
TICK_SIZE = 100.0
EVENT_REL_PRICE_CLIP = 0.05
DROP_ZERO_RET = False
PRICE_CLIP_TICKS = 300

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
    rel_price = np.clip(rel_price, -EVENT_REL_PRICE_CLIP, EVENT_REL_PRICE_CLIP).astype(np.float32)
    out["rel_event_price"] = rel_price

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
    Build y_ret safely (event-based or time-based), avoiding day-cross leakage.

    Modes:
      - Event-based (HORIZON_MS is None):
          y_ret[t] = log(mid[t+H_events]) - log(mid[t])
      - Time-based (HORIZON_MS is int ms):
          y_ret[t] = log(mid[t']) - log(mid[t]) where time[t'] >= time[t] + H_ms
          and t' is constrained to remain within the same trading day segment

    Returns:
      labels_full (length N, with NaN where invalid / tail)
      mask (length N) boolean, true where label is valid (+ optional drop_zero)
    """
    N = len(mid_raw)
    y = np.full(N, np.nan, dtype=np.float32)

    if HORIZON_MS is None:
        # Event-based horizon
        if HORIZON_EVENTS is None or HORIZON_EVENTS <= 0:
            y[:] = 0.0
            valid = np.ones(N, dtype=bool)
        else:
            h = int(HORIZON_EVENTS)
            mid_now = mid_raw[:-h].astype(np.float64, copy=False)
            mid_fut = mid_raw[h:].astype(np.float64, copy=False)
            y_val = (np.log(mid_fut + EPS) - np.log(mid_now + EPS)).astype(np.float32)
            y[: N - h] = y_val
            valid = np.isfinite(y)
    else:
        # Time-based horizon (milliseconds)
        if isinstance(HORIZON_MS, tuple):
            L_ms, U_ms = int(HORIZON_MS[0]), int(HORIZON_MS[1])
            if L_ms > U_ms:
                L_ms, U_ms = U_ms, L_ms
            L_sec = float(L_ms) / 1000.0
            U_sec = float(U_ms) / 1000.0
        else:
            L_sec = U_sec = float(int(HORIZON_MS)) / 1000.0

        t = time_s.to_numpy(dtype=np.float64, copy=False)

        # Identify day segments where time is non-decreasing
        day_id = np.zeros(N, dtype=np.int32)
        if N > 1:
            dt = t[1:] - t[:-1]
            resets = (dt < 0).astype(np.int32)
            # cumulative sum of resets gives day index per row (shifted by 1 for alignment)
            day_id[1:] = np.cumsum(resets)

        # Two-pointer search within each day segment
        # For each i, find j_low: t[j_low] >= t[i] + L_sec, j_up: t[j_up] >= t[i] + U_sec
        for d in range(int(day_id.max()) + 1):
            idx = np.nonzero(day_id == d)[0]
            if len(idx) == 0:
                continue
            start = idx[0]
            end = idx[-1] + 1  # exclusive
            j_low = start
            j_up = start
            use_avg = (HORIZON_MS_MODE == "avg")
            use_lower = (HORIZON_MS_MODE == "lower")
            use_upper = (HORIZON_MS_MODE == "upper") or (not (use_avg or use_lower))

            for i in range(start, end):
                target_low = t[i] + L_sec
                target_up = t[i] + U_sec

                if j_low < i:
                    j_low = i
                if j_up < i:
                    j_up = i

                while j_low < end and t[j_low] < target_low:
                    j_low += 1
                while j_up < end and t[j_up] < target_up:
                    j_up += 1

                ret_low = np.nan
                ret_up = np.nan
                if j_low < end:
                    ret_low = (np.log(float(mid_raw[j_low]) + EPS) - np.log(float(mid_raw[i]) + EPS)).astype(np.float32)
                if j_up < end:
                    ret_up = (np.log(float(mid_raw[j_up]) + EPS) - np.log(float(mid_raw[i]) + EPS)).astype(np.float32)

                if use_avg:
                    if np.isfinite(ret_low) and np.isfinite(ret_up):
                        y[i] = np.float32(0.5 * (float(ret_low) + float(ret_up)))
                    else:
                        y[i] = np.nan
                elif use_lower:
                    y[i] = ret_low
                else:  # upper (default)
                    y[i] = ret_up

        valid = np.isfinite(y)

    if DROP_ZERO_RET:
        valid &= (y != 0.0)

    labels = pd.DataFrame({"y_ret": y})
    return labels, valid

def main() -> None:
    print("=" * 90)
    print("STEP 01 — PREPROCESS RAW LOBSTER (TRAINING-SAFE, FI-2010/LiT STYLE)")
    print(f"TICKER         : {TICKER}")
    print(f"HORIZON_EVENTS : {HORIZON_EVENTS}")
    print(f"HORIZON_MS     : {HORIZON_MS}")
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
        
        # Load events
        events_chunk = pd.read_csv(msg_f, header=None, names=RAW_EVENT_COLS, encoding="latin1")
        n_before = len(events_chunk)
        events_chunk = events_chunk.dropna(subset=["time", "event_type", "size", "price", "direction"])
        events_chunk = events_chunk.drop(columns=["order_id"])
        events_chunk["time"] = events_chunk["time"].astype(np.float64)
        events_chunk["event_type"] = events_chunk["event_type"].astype(np.int32)
        events_chunk["size"] = events_chunk["size"].astype(np.float64)
        events_chunk["price"] = events_chunk["price"].astype(np.float64)
        events_chunk["direction"] = events_chunk["direction"].astype(np.int32)
        
        # Load orderbook
        ob_chunk = pd.read_csv(ob_f, header=None, names=ob_cols, encoding="latin1")
        ob_chunk = ob_chunk.dropna()
        
        n_after = min(len(events_chunk), len(ob_chunk))
        total_rows_before += n_before
        total_rows_after += n_after
        
        if len(events_chunk) != len(ob_chunk):
            print(f"    [WARN] Length mismatch: events={len(events_chunk)}, ob={len(ob_chunk)}, using min={n_after}")
            events_chunk = events_chunk.iloc[:n_after]
            ob_chunk = ob_chunk.iloc[:n_after]
        
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
            "No z-score normalization in Step 01 (avoid leakage).",
            "NO global clipping of LOB distances in Step 01 (keeps deep-level info).",
            "One mask is applied to events/orderbook/labels to avoid index drift.",
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