import os
from pathlib import Path

def download_data(output_dir: str = "data/raw") -> None:
    """Download the dataset to the given directory."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    file_path = out_path / "sample.txt"
    file_path.write_text("sample data")
    print(f"Data saved to {file_path}")

if __name__ == "__main__":
    download_data()
