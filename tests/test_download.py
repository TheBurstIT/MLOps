from crypto_lob_micro_move.data.download import download_data
from pathlib import Path
import shutil

def test_download_creates_file(tmp_path):
    output_dir = tmp_path / "data"
    download_data(str(output_dir))
    assert (output_dir / "sample.txt").exists()
