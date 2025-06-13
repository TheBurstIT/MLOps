from crypto_lob_micro_move.train import train, TrainConfig


def test_train_raises_for_missing_data(tmp_path):
    missing_dir = tmp_path / "missing"
    cfg = TrainConfig(data_dir=str(missing_dir), dataset="btc")
    try:
        train(cfg)
    except FileNotFoundError:
        assert True
    else:
        assert False, "Expected FileNotFoundError"


def test_train_uses_dataset_subdirectory(tmp_path):
    data_dir = tmp_path / "data"
    dataset_dir = data_dir / "eth"
    dataset_dir.mkdir(parents=True)
    cfg = TrainConfig(data_dir=str(data_dir), dataset="eth")
    train(cfg)
