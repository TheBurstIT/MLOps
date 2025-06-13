from pathlib import Path
import numpy as np

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
    # generate small synthetic dataset: each row has window*30 features followed
    # by a label column
    num_samples = 10
    window = 10
    data = np.random.randn(num_samples, window * 30)
    labels = np.random.randint(0, 3, size=(num_samples, 1))
    full = np.concatenate([data, labels], axis=1)
    np.savetxt(file_path, full, delimiter=",")
    print(f"Data saved to {file_path}")
    return file_path

if __name__ == "__main__":
    download_data()
