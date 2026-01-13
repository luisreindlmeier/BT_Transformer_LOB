"""
check_02.py

Minimal sanity check for STEP 02.
- Loads TRAIN split only
- Writes small CSV preview (events + orderbook + label)
"""

from pathlib import Path
import pandas as pd

# =============================================================================
# CONFIG
# =============================================================================

TICKER = "CSCO"
N_ROWS = 50

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLIT_ROOT = PROJECT_ROOT / "data" / "02_split" / TICKER / "train"
DEBUG_ROOT = PROJECT_ROOT / "data" / "02_split" / TICKER / "debug"
DEBUG_ROOT.mkdir(exist_ok=True)

CSV_OUT = DEBUG_ROOT / f"preview_train_{TICKER}.csv"

# =============================================================================
# MAIN
# =============================================================================

def main():
    events = pd.read_parquet(SPLIT_ROOT / "events.parquet")
    orderbook = pd.read_parquet(SPLIT_ROOT / "orderbook.parquet")
    labels = pd.read_parquet(SPLIT_ROOT / "labels.parquet")

    assert len(events) == len(orderbook) == len(labels), "Alignment error after split"

    preview = pd.concat(
        [
            events.head(N_ROWS),
            orderbook.head(N_ROWS),
            labels.head(N_ROWS),
        ],
        axis=1,
    )

    preview.to_csv(CSV_OUT, index=False)
    print(f"CSV preview written to:\n{CSV_OUT}")

# =============================================================================
if __name__ == "__main__":
    main()