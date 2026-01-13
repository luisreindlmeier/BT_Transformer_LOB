"""
check_03.py

Minimal check for STEP 03.
Writes CSV preview of normalized TRAIN data.
"""

from pathlib import Path
import pandas as pd

# =============================================================================
# CONFIG
# =============================================================================

TICKER = "CSCO"
N_ROWS = 50

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "data" / "03_normalized_NEW" / TICKER
DEBUG_ROOT = ROOT / "debug"
DEBUG_ROOT.mkdir(exist_ok=True)

CSV_OUT = DEBUG_ROOT / f"preview_train_norm_{TICKER}.csv"

# =============================================================================
# MAIN
# =============================================================================

def main():
    events = pd.read_parquet(ROOT / "train" / "events.parquet")
    orderbook = pd.read_parquet(ROOT / "train" / "orderbook.parquet")
    labels = pd.read_parquet(ROOT / "train" / "labels.parquet")

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