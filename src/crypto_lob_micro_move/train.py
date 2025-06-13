from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger

from .datamodules.lob_dm import LOBDataModule
from .models.tcn_module import SimpleTCN


CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs"


def run_train(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed)

    dm = LOBDataModule(data_dir=cfg.data.path, batch_size=cfg.batch_size)

    model = SimpleTCN(
        channels=cfg.model.channels, kernel_size=cfg.model.kernel_size, lr=cfg.lr
    )

    logger = MLFlowLogger(
        experiment_name=cfg.logging.experiment, tracking_uri=cfg.logging.tracking_uri
    )

    trainer = pl.Trainer(max_epochs=cfg.epochs, log_every_n_steps=1, logger=logger)
    trainer.fit(model, datamodule=dm)


@hydra.main(config_path=str(CONFIG_PATH), config_name="train/default.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    run_train(cfg)


if __name__ == "__main__":
    main()
