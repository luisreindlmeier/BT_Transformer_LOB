# src/dataset.py

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# =============================================================================
# Dataset
# =============================================================================

class WindowNPYDataset(Dataset):
    """
    Dataset for STEP 05 windowed numpy arrays.

    Expected structure:
      root/
        train_X.npy
        train_y.npy
        val_X.npy
        val_y.npy
        test_X.npy
        test_y.npy
    
    Labels (classification):
      - 0: down (y_ret < -tau)
      - 1: stationary (-tau <= y_ret <= +tau)
      - 2: up (y_ret > +tau)
    """

    def __init__(self, root: str | Path, split: str, classification: bool = False, 
                 feature_type: str = "all", n_event_features: int = 10, tau: float = 0.002):
        """
        Args:
            feature_type: "all" | "events_only" | "lob_only"
            n_event_features: number of event feature columns (for slicing)
            tau: threshold for 3-class classification (down < -tau, stationary in [-tau, tau], up > tau)
        """
        self.root = Path(root)
        self.classification = classification
        self.feature_type = feature_type
        self.n_event_features = n_event_features
        self.tau = tau

        x_path = self.root / f"{split}_X.npy"
        y_path = self.root / f"{split}_y.npy"

        if not x_path.exists():
            raise FileNotFoundError(x_path)
        if not y_path.exists():
            raise FileNotFoundError(y_path)

        # memory-mapped loading (CRUCIAL for large datasets)
        self.X = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")

        assert len(self.X) == len(self.y), "X/y length mismatch"

        desc = f"[Dataset] {split} ({feature_type}): {len(self.X):,} samples | X shape = {self.X.shape}"
        print(desc)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Flatten window [W, F] -> [W*F] for simple MLP
        X_window = self.X[idx]  # shape (W, F)
        W, F = X_window.shape
        
        # Feature selection
        if self.feature_type == "events_only":
            X_flat = X_window[:, :self.n_event_features].reshape(-1)
        elif self.feature_type == "lob_only":
            X_flat = X_window[:, self.n_event_features:].reshape(-1)
        else:  # "all"
            X_flat = X_window.reshape(-1)
        
        X = torch.from_numpy(np.ascontiguousarray(X_flat)).float()

        y_val = self.y[idx]
        if self.classification:
            # 3-class label: down (0), stationary (1), up (2)
            if y_val < -self.tau:
                y = torch.tensor(0, dtype=torch.long)  # down
            elif y_val > self.tau:
                y = torch.tensor(2, dtype=torch.long)  # up
            else:
                y = torch.tensor(1, dtype=torch.long)  # stationary
        else:
            y = torch.tensor(y_val, dtype=torch.float32)
        return X, y


# =============================================================================
# DataLoader helper
# =============================================================================

def build_dataloader(
    root: str | Path,
    split: str,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 0,
    classification: bool = False,
    pin_memory: bool | None = None,
    prefetch_factor: int | None = None,
    persistent_workers: bool | None = None,
    feature_type: str = "all",
    n_event_features: int = 10,
    tau: float = 0.002,
):
    dataset = WindowNPYDataset(
        root, split, classification=classification,
        feature_type=feature_type, n_event_features=n_event_features, tau=tau
    )

    if pin_memory is None:
        # Only enable pin_memory on CUDA; on CPU/MPS it's not helpful
        pin_memory = torch.cuda.is_available()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(False if persistent_workers is None else persistent_workers),
        prefetch_factor=(2 if prefetch_factor is None else prefetch_factor) if num_workers > 0 else None,
    )