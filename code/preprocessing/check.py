from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # thesis/
PROCESSED = PROJECT_ROOT / "data" / "processed"

print(pd.read_parquet(PROCESSED / "events_AAPL.parquet").head())
print(pd.read_parquet(PROCESSED / "orderbook_AAPL.parquet").head())