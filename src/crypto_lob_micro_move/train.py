from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal


Dataset = Literal["btc", "eth", "ada"]


@dataclass
class TrainConfig:
    """Configuration for the training routine."""

    data_dir: str = "data/raw"
    dataset: Dataset = "btc"
    epochs: int = 1

    @property
    def dataset_path(self) -> Path:
        return Path(self.data_dir) / self.dataset


def train(config: TrainConfig = TrainConfig()) -> None:
    """A stub training routine."""

    data_path = config.dataset_path
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    print(f"Training on {data_path} for {config.epochs} epochs")


if __name__ == "__main__":
    train()
