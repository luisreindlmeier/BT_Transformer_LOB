from pathlib import Path
import torch
import pandas as pd
import numpy as np

# =============================
# CONFIG
# =============================
TICKER = "CSCO"
SPLIT = "train"
SHARD_IDX = 0
N_SAMPLES = 200      # rows in CSV
EVENT_WINDOW = 50
EVENT_DIM = 6

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "datasets" / "lit_eventbased" / TICKER / SPLIT

# =============================
# LOAD SHARD
# =============================
shard = DATA_ROOT / f"shard_{SHARD_IDX:05d}.pt"
data = torch.load(shard, map_location="cpu")

events = data["events"][:N_SAMPLES]     # (N, 50, 6)
lob = data["lob"][:N_SAMPLES]            # (N, 40)
extras = data["extras"][:N_SAMPLES]
y_ret = data["y_ret"][:N_SAMPLES]
y_cls = data["y_cls"][:N_SAMPLES]

# =============================
# BUILD TABLE
# =============================
rows = []

for i in range(N_SAMPLES):
    row = {}

    # --- flatten events ---
    for t in range(EVENT_WINDOW):
        for d in range(EVENT_DIM):
            row[f"ev_t{t}_f{d}"] = events[i, t, d].item()

    # --- LOB ---
    for j in range(lob.shape[1]):
        row[f"lob_{j}"] = lob[i, j].item()

    # --- extras ---
    row["OFI"] = extras[i, 0].item()
    row["QueueImbalance"] = extras[i, 1].item()
    row["ExecPressure"] = extras[i, 2].item()

    # --- targets ---
    row["y_ret"] = y_ret[i].item()
    row["y_cls"] = int(y_cls[i])

    rows.append(row)

df = pd.DataFrame(rows)

# =============================
# SAVE
# =============================
out = Path("training_sample_debug.csv")
df.to_csv(out, index=False)

print(f"\nSaved {len(df)} rows to {out.resolve()}")