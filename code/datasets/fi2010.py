import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class FI2010Dataset(Dataset):
    """
    FI-2010 Dataset with temporal sliding window.

    Each sample:
        X_t = [x_{t-T+1}, ..., x_t]  ∈ R^{T × 144}
        y_t = label at time t for horizon h
    """

    def __init__(self, csv_path, horizon_idx=0, window_size=50):
        super().__init__()

        self.df = pd.read_csv(csv_path)

        # Drop index column if present
        if "Unnamed: 0" in self.df.columns:
            self.df = self.df.drop(columns=["Unnamed: 0"])

        # Features / labels
        self.X = self.df.iloc[:, :-5].values.astype(np.float32)
        self.y_raw = self.df.iloc[:, -5 + horizon_idx].values.astype(int)

        # Map {1,2,3} → {0,1,2}
        self.y = self.y_raw - 1

        self.window_size = window_size

    def __len__(self):
        return len(self.X) - self.window_size + 1

    def __getitem__(self, idx):
        x_seq = self.X[idx : idx + self.window_size]        # (T, 144)
        y = self.y[idx + self.window_size - 1]              # label at t

        return torch.tensor(x_seq), torch.tensor(y)