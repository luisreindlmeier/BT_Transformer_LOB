from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class ShardedEventDataset(Dataset):

    def __init__(self, shard_dir: Path):
        self.shard_paths = sorted(shard_dir.glob("shard_*.pt"))
        if not self.shard_paths:
            raise RuntimeError(f"No shard_*.pt files found in {shard_dir}")

        # read shard sizes only (cheap)
        self.shard_sizes = []
        for p in self.shard_paths:
            data = torch.load(p, map_location="cpu", weights_only=True)
            self.shard_sizes.append(data["y_cls"].shape[0])
            del data

        self.cum_sizes = torch.tensor(self.shard_sizes).cumsum(0)

        # per-worker shard cache
        self._cached_shard_id = None
        self._cached_data = None

    def __len__(self):
        return int(self.cum_sizes[-1])

    def __getitem__(self, idx):
        shard_id = int(torch.searchsorted(self.cum_sizes, idx, right=True))
        local_idx = idx if shard_id == 0 else idx - self.cum_sizes[shard_id - 1]

        # load shard only if necessary
        if shard_id != self._cached_shard_id:
            self._cached_data = torch.load(
                self.shard_paths[shard_id],
                map_location="cpu"
            )
            self._cached_shard_id = shard_id

        d = self._cached_data
        return {
            "events": d["events"][local_idx],
            "lob": d["lob"][local_idx],
            "extras": d["extras"][local_idx],
            "y_cls": d["y_cls"][local_idx],
            "y_ret": d["y_ret"][local_idx],
        }

class LOBDataModule(pl.LightningDataModule):

    def __init__(
        self,
        dataset_root: Path,
        batch_size: int = 32,
        num_workers: int = 2,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory

    # -------------------------------------------------

    def setup(self, stage: str | None = None):
        manifest_path = self.dataset_root / "dataset_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        self.train_dataset = ShardedEventDataset(self.dataset_root / "train")
        self.val_dataset   = ShardedEventDataset(self.dataset_root / "val")
        self.test_dataset  = ShardedEventDataset(self.dataset_root / "test")

        self.class_weights = torch.tensor(
            manifest["class_weights"], dtype=torch.float32
        )

    # -------------------------------------------------

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
        )