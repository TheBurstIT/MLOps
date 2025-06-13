import os
from pathlib import Path

def download_data(output_dir: str = "data/raw") -> None:
    """Placeholder function to simulate data download."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    dummy_file = out_path / "sample.txt"
    dummy_file.write_text("sample data")
    print(f"Data saved to {dummy_file}")

if __name__ == "__main__":
    download_data()
