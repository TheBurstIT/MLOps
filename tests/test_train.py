from crypto_lob_micro_move.train import train
from pathlib import Path


def test_train_raises_for_missing_data(tmp_path):
    missing_dir = tmp_path / "missing"
    try:
        train(str(missing_dir))
    except FileNotFoundError:
        assert True
    else:
        assert False, "Expected FileNotFoundError"
