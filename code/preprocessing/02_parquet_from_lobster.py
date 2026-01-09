from pathlib import Path
import pandas as pd

# =============================
# Paths & configuration
# =============================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "NASDAQ"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

TICKERS = ["AAPL", "CSCO", "GOOG", "INTC"]

# =============================
# LOBSTER column definitions
# =============================

EVENT_COLS = [
    "time",
    "event_type",
    "order_id",
    "size",
    "price",
    "direction",
]

def build_orderbook_columns(n_levels: int) -> list[str]:
    cols = []
    for k in range(1, n_levels + 1):
        cols.extend([
            f"ask_price_{k}",
            f"ask_size_{k}",
            f"bid_price_{k}",
            f"bid_size_{k}",
        ])
    return cols

# =============================
# Main preprocessing loop
# =============================

for ticker in TICKERS:
    print(f"\nProcessing {ticker}...")


    ticker_path = DATA_ROOT / ticker
    msg_files = sorted(p for p in ticker_path.iterdir() if "message" in p.name)
    ob_files  = sorted(p for p in ticker_path.iterdir() if "orderbook" in p.name)

    if not msg_files or not ob_files:
        raise RuntimeError(f"Missing message or orderbook files for {ticker}")

    # -------------------------
    # Message files → Parquet
    # -------------------------
    msg_dfs = []
    for f in msg_files:
        df = pd.read_csv(
            f,
            header=None,
            names=EVENT_COLS,
        )
        msg_dfs.append(df)

    events = pd.concat(msg_dfs, ignore_index=True)
    events["ticker"] = ticker

    events_path = OUTPUT_ROOT / f"events_{ticker}.parquet"
    events.to_parquet(events_path, index=False)
    print(f"  -> events saved: {events.shape}")

    # -------------------------
    # Orderbook files → Parquet
    # -------------------------
    sample_ob = pd.read_csv(ob_files[0], nrows=1, header=None)
    n_levels = sample_ob.shape[1] // 4

    if n_levels <= 0:
        raise RuntimeError(f"Invalid orderbook depth detected for {ticker}")

    ob_cols = build_orderbook_columns(n_levels)

    ob_dfs = []
    for f in ob_files:
        df = pd.read_csv(
            f,
            header=None,
            names=ob_cols,
        )
        ob_dfs.append(df)

    orderbook = pd.concat(ob_dfs, ignore_index=True)
    orderbook["ticker"] = ticker

    ob_path = OUTPUT_ROOT / f"orderbook_{ticker}.parquet"
    orderbook.to_parquet(ob_path, index=False)
    print(f"  -> orderbook saved: {orderbook.shape}")

print("\nDONE.")