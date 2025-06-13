from __future__ import annotations

from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import torch
import pytorch_lightning as pl


class DummyLOBDataset(Dataset):
    def __init__(self, length: int = 100, window: int = 10):
        self.length = length
        self.window = window

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(self.window, 30)
        y = torch.randint(0, 3, (1,))
        return x, y


class LOBDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 4, window: int = 10):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.window = window
        if not self.data_dir.exists():
            raise FileNotFoundError(self.data_dir)

    def setup(self, stage: str | None = None) -> None:
        self.train_ds = DummyLOBDataset(window=self.window)
        self.val_ds = DummyLOBDataset(window=self.window)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=self.batch_size)
