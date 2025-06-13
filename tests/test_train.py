from crypto_lob_micro_move.datamodules.lob_dm import LOBDataModule
from crypto_lob_micro_move.models.tcn_module import SimpleTCN
import torch


def test_datamodule_raises_for_missing_data(tmp_path):
    missing_dir = tmp_path / "missing"
    try:
        LOBDataModule(str(missing_dir))
    except FileNotFoundError:
        assert True
    else:
        assert False, "Expected FileNotFoundError"


def test_lightning_step_no_nan(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    dm = LOBDataModule(str(data_dir))
    dm.setup()
    model = SimpleTCN()
    batch = next(iter(dm.train_dataloader()))
    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss).all()
