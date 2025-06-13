from pathlib import Path


def train(data_dir: str = "data/raw", epochs: int = 1) -> None:
    """A stub training routine."""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    print(f"Training on {data_path} for {epochs} epochs")


if __name__ == "__main__":
    train()
