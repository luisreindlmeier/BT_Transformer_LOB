"""Step 03: Fit z-score normalization on train split and apply to all splits."""

from pathlib import Path
import json
import numpy as np
import pandas as pd

TICKER = "CSCO"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IN_ROOT  = Path.home() / "thesis_output" / "02_split" / TICKER
OUT_ROOT = Path.home() / "thesis_output" / "03_normalized_NEW" / TICKER
OUT_ROOT.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "val", "test"]

def fit_scaler(df: pd.DataFrame, cols: list[str]) -> dict:
    scaler = {}
    for c in cols:
        mu = df[c].mean()
        sigma = df[c].std()
        scaler[c] = {
            "mean": float(mu),
            "std": float(sigma if sigma > 0 else 1.0),
        }
    return scaler

def apply_scaler(df: pd.DataFrame, scaler: dict) -> pd.DataFrame:
    out = df.copy()
    for c, stats in scaler.items():
        out[c] = (out[c] - stats["mean"]) / stats["std"]
    return out

def main():
    print("=" * 90)
    print("STEP 03 — FIT NORMALIZATION ON TRAIN, APPLY TO ALL SPLITS")
    print(f"TICKER : {TICKER}")
    print("=" * 90)

    train_events = pd.read_parquet(IN_ROOT / "train" / "events.parquet")
    train_ob     = pd.read_parquet(IN_ROOT / "train" / "orderbook.parquet")

    event_numeric_cols = [
        c for c in train_events.columns
        if c not in [f"ev_{k}" for k in range(1, 7)] and c != "time_s"
    ]

    ob_numeric_cols = list(train_ob.columns)

    print(f"Event numeric features : {len(event_numeric_cols)}")
    print(f"LOB numeric features   : {len(ob_numeric_cols)}")

    # -------------------------------------------------------------------------
    # Fit scalers on TRAIN only
    # -------------------------------------------------------------------------
    event_scaler = fit_scaler(train_events, event_numeric_cols)
    ob_scaler    = fit_scaler(train_ob, ob_numeric_cols)

    # -------------------------------------------------------------------------
    # Apply to all splits
    # -------------------------------------------------------------------------
    for split in SPLITS:
        print(f"\nProcessing split: {split}")

        events = pd.read_parquet(IN_ROOT / split / "events.parquet")
        orderbook = pd.read_parquet(IN_ROOT / split / "orderbook.parquet")
        labels = pd.read_parquet(IN_ROOT / split / "labels.parquet")

        events_norm = events.copy()
        events_norm[event_numeric_cols] = apply_scaler(
            events[event_numeric_cols], event_scaler
        )

        orderbook_norm = apply_scaler(orderbook, ob_scaler)

        out_dir = OUT_ROOT / split
        out_dir.mkdir(parents=True, exist_ok=True)

        events_norm.to_parquet(out_dir / "events.parquet", index=False)
        orderbook_norm.to_parquet(out_dir / "orderbook.parquet", index=False)
        labels.to_parquet(out_dir / "labels.parquet", index=False)

        print(f"Saved normalized {split}")

    # -------------------------------------------------------------------------
    # Save scaler metadata
    # -------------------------------------------------------------------------
    scaler_meta = {
        "event_scaler": event_scaler,
        "orderbook_scaler": ob_scaler,
        "notes": [
            "Scaler fitted ONLY on TRAIN split.",
            "Z-score normalization.",
            "One-hot event features not normalized.",
            "Labels not normalized.",
        ],
    }

    scaler_path = OUT_ROOT / "scaler.json"
    scaler_path.write_text(json.dumps(scaler_meta, indent=2))

    print("\nSaved scaler:")
    print(scaler_path)
    print("\nDONE — STEP 03 COMPLETE ✔")

# =============================================================================
if __name__ == "__main__":
    main()