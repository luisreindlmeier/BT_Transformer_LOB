"""
check_01.py

Minimal sanity check for STEP 01 preprocessing.
Purpose:
- Load STEP 01 outputs
- Write a small CSV preview for manual inspection (Excel)
"""

from pathlib import Path
import pandas as pd

# =============================================================================
# CONFIG
# =============================================================================

TICKER = "CSCO"
PREVIEW_ROWS = 50

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "01_preprocessed"
DEBUG_ROOT = DATA_ROOT / "debug"
DEBUG_ROOT.mkdir(exist_ok=True)

EVENTS_PATH = DATA_ROOT / f"events_{TICKER}.parquet"
OB_PATH     = DATA_ROOT / f"orderbook_{TICKER}.parquet"
LABELS_PATH = DATA_ROOT / f"labels_{TICKER}.parquet"

OUT_CSV = DEBUG_ROOT / f"preview_{TICKER}.csv"

# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    # --- load ---
    events = pd.read_parquet(EVENTS_PATH)
    orderbook = pd.read_parquet(OB_PATH)
    labels = pd.read_parquet(LABELS_PATH)

    # --- basic alignment sanity ---
    n = min(len(events), len(orderbook), len(labels))

    events = events.iloc[:n].reset_index(drop=True)
    orderbook = orderbook.iloc[:n].reset_index(drop=True)
    labels = labels.iloc[:n].reset_index(drop=True)

    # --- merge preview ---
    preview = pd.concat(
        [
            events.iloc[:PREVIEW_ROWS],
            orderbook.iloc[:PREVIEW_ROWS],
            labels.iloc[:PREVIEW_ROWS],
        ],
        axis=1,
    )

    preview.to_csv(OUT_CSV, index=False)

    # --- minimal terminal output ---
    print("=" * 80)
    print("CHECK 01 — CSV PREVIEW WRITTEN")
    print(f"TICKER : {TICKER}")
    print(f"ROWS   : {PREVIEW_ROWS}")
    print(f"FILE   : {OUT_CSV}")
    print("=" * 80)

# =============================================================================
if __name__ == "__main__":
    main()