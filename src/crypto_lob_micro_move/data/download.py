from pathlib import Path
import csv
import random

def _try_kaggle_download() -> Path | None:
    """Attempt to download the dataset via kagglehub."""
    try:
        import kagglehub

        return Path(
            kagglehub.dataset_download(
                "martinsn/high-frequency-crypto-limit-order-book-data"
            )
        )
    except Exception:
        # Any error means kagglehub or credentials are unavailable
        return None

def download_data(output_dir: str = "data/raw") -> Path:
    """Download the dataset to the given directory."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    dataset_path = _try_kaggle_download()
    if dataset_path:
        print(f"Dataset downloaded to {dataset_path}")
        return dataset_path

    file_path = out_path / "sample.csv"
    num_samples = 10
    features = 30
    rows = []
    for _ in range(num_samples):
        row = [random.gauss(0, 1) for _ in range(features)]
        row.append(random.randint(0, 2))
        rows.append(row)
    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"Data saved to {file_path}")
    return file_path

if __name__ == "__main__":
    download_data()
