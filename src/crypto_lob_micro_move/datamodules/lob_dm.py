from __future__ import annotations

from pathlib import Path
import csv


class CSVLOBDataset:
    """Dataset reading the small CSV produced by the downloader."""

    def __init__(self, csv_path: Path):
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        rows = []
        with open(csv_path) as f:
            reader = csv.reader(f)
            for row in reader:
                rows.append([float(x) for x in row])
        self.x = [r[:-1] for r in rows]
        self.y = [int(r[-1]) for r in rows]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class SimpleDataLoader:
    def __init__(self, dataset: CSVLOBDataset, batch_size: int = 1):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        for i in range(0, len(self.dataset), self.batch_size):
            x = self.dataset.x[i : i + self.batch_size]
            y = self.dataset.y[i : i + self.batch_size]
            yield x, y


class LOBDataModule:
    def __init__(self, data_dir: str, batch_size: int = 4):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.csv_file = self.data_dir / "sample.csv"
        if not self.csv_file.exists():
            raise FileNotFoundError(self.csv_file)

    def setup(self):
        self.train_ds = CSVLOBDataset(self.csv_file)
        self.val_ds = CSVLOBDataset(self.csv_file)

    def train_dataloader(self) -> SimpleDataLoader:
        return SimpleDataLoader(self.train_ds, batch_size=self.batch_size)

    def val_dataloader(self) -> SimpleDataLoader:
        return SimpleDataLoader(self.val_ds, batch_size=self.batch_size)
