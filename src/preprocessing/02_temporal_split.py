"""Step 02: Split preprocessed data into train/val/test by time."""

from pathlib import Path
import os
import json
import pandas as pd

TICKER = os.getenv("TICKER", "CSCO")

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IN_ROOT  = Path.home() / "thesis_output" / "01_preprocessed"
OUT_ROOT = Path.home() / "thesis_output" / "02_split" / TICKER

EVENTS_IN = IN_ROOT / f"events_{TICKER}.parquet"
OB_IN     = IN_ROOT / f"orderbook_{TICKER}.parquet"
LABELS_IN = IN_ROOT / f"labels_{TICKER}.parquet"

def main() -> None:
    print("=" * 90)
    print("STEP 02 — TEMPORAL SPLIT (TRAIN / VAL / TEST)")
    print(f"TICKER : {TICKER}")
    print("=" * 90)

    events = pd.read_parquet(EVENTS_IN)
    orderbook = pd.read_parquet(OB_IN)
    labels = pd.read_parquet(LABELS_IN)

    assert len(events) == len(orderbook) == len(labels), "Alignment error in STEP 01 outputs"

    n = len(events)
    n_train = int(TRAIN_RATIO * n)
    n_val   = int(VAL_RATIO * n)
    n_test  = n - n_train - n_val

    print(f"Total rows : {n:,}")
    print(f"Train      : {n_train:,}")
    print(f"Val        : {n_val:,}")
    print(f"Test       : {n_test:,}")

    splits = {
        "train": slice(0, n_train),
        "val":   slice(n_train, n_train + n_val),
        "test":  slice(n_train + n_val, n),
    }

    for split, sl in splits.items():
        out_dir = OUT_ROOT / split
        out_dir.mkdir(parents=True, exist_ok=True)

        events.iloc[sl].to_parquet(out_dir / "events.parquet", index=False)
        orderbook.iloc[sl].to_parquet(out_dir / "orderbook.parquet", index=False)
        labels.iloc[sl].to_parquet(out_dir / "labels.parquet", index=False)

        print(f"Saved {split}: {len(events.iloc[sl]):,} rows")

    # -------------------------------------------------------------------------
    # Manifest (for sanity & reproducibility)
    # -------------------------------------------------------------------------
    manifest = {
        "ticker": TICKER,
        "total_rows": n,
        "splits": {
            "train": n_train,
            "val": n_val,
            "test": n_test,
        },
        "ratios": {
            "train": TRAIN_RATIO,
            "val": VAL_RATIO,
            "test": TEST_RATIO,
        },
        "notes": [
            "Strict temporal split (no shuffle).",
            "All splits derived from STEP 01 outputs.",
            "No normalization or windowing applied.",
        ],
    }

    (OUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("\nDONE — STEP 02 SPLIT COMPLETE ✔")

# =============================================================================
if __name__ == "__main__":
    main()