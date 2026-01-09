"""
Build aligned model inputs from LOBSTER data using STREAMING.

- Reads Parquet files in CHUNKS
- Never loads full events / orderbook into memory
- Builds rolling event windows + LOB snapshots
- Writes model inputs incrementally to disk
"""

from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
from collections import deque

# -----------------------------
# Paths & config
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "processed"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "model_inputs"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

TICKERS = ["AAPL", "CSCO", "GOOG", "INTC"]

EVENT_WINDOW = 50          # past events
HORIZON_EVENTS = 10        # prediction horizon
CHUNK_SIZE = 200_000       # safe for your disk situation

# -----------------------------
# Helpers
# -----------------------------

def compute_mid(row):
    return (row["ask_price_1"] + row["bid_price_1"]) / 2


# -----------------------------
# Main loop (STREAMING)
# -----------------------------

for ticker in TICKERS:
    print(f"\nBuilding model inputs (streaming) for {ticker}...")

    events_path = DATA_ROOT / f"events_{ticker}.parquet"
    ob_path = DATA_ROOT / f"orderbook_{ticker}.parquet"

    events_pf = pq.ParquetFile(events_path)
    ob_pf = pq.ParquetFile(ob_path)

    output_file = OUTPUT_ROOT / f"inputs_{ticker}.parquet"
    writer = None

    event_buffer = deque(maxlen=EVENT_WINDOW + HORIZON_EVENTS)
    lob_buffer = deque(maxlen=EVENT_WINDOW + HORIZON_EVENTS)

    total_samples = 0

    for events_batch, ob_batch in zip(
        events_pf.iter_batches(batch_size=CHUNK_SIZE),
        ob_pf.iter_batches(batch_size=CHUNK_SIZE)
    ):
        events_df = events_batch.to_pandas()
        ob_df = ob_batch.to_pandas()

        assert len(events_df) == len(ob_df)

        for i in range(len(events_df)):
            event_buffer.append(events_df.iloc[i])
            lob_buffer.append(ob_df.iloc[i])

            if len(event_buffer) < EVENT_WINDOW + HORIZON_EVENTS:
                continue

            # -----------------------------
            # Build inputs
            # -----------------------------

            ev_window = pd.DataFrame(
                list(event_buffer)[:EVENT_WINDOW]
            ).drop(columns=["ticker"], errors="ignore")

            lob_now = lob_buffer[EVENT_WINDOW - 1]
            lob_future = lob_buffer[EVENT_WINDOW - 1 + HORIZON_EVENTS]

            mid_now = compute_mid(lob_now)
            mid_future = compute_mid(lob_future)
            target = (mid_future - mid_now) / mid_now

            event_array = ev_window.values.astype(np.float32)
            lob_array = lob_now.drop(labels=["ticker"], errors="ignore").values.astype(np.float32)

            row = {
                "event_window": event_array.tobytes(),
                "event_window_shape": event_array.shape,
                "lob_snapshot": lob_array.tobytes(),
                "lob_snapshot_shape": lob_array.shape,
                "target": np.float32(target)
            }

            table = pa.Table.from_pylist([row])

            if writer is None:
                writer = pq.ParquetWriter(output_file, table.schema)

            writer.write_table(table)
            total_samples += 1

        print(f"  processed chunk → total samples: {total_samples}")

    if writer is not None:
        writer.close()

    print(f"  -> finished {ticker}: {total_samples} samples written")

print("\nDONE.")