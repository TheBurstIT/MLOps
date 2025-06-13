from pathlib import Path
import numpy as np
from .datamodules.lob_dm import LOBDataModule
from .models.tcn_module import SimpleTCN

try:  # use hydra if installed
    from hydra import main as hydra_main
    from omegaconf import DictConfig
except Exception:  # pragma: no cover - optional dependency
    import yaml

    def hydra_main(config_path: str | None = None, config_name: str | None = None):
        def wrapper(func):
            def inner(*args, **kwargs):
                cfg_file = Path(config_path or "") / (config_name or "config.yaml")
                with open(cfg_file) as f:
                    cfg = yaml.safe_load(f)
                return func(cfg, *args, **kwargs)

            return inner

        return wrapper



def run_train(cfg) -> None:
    dm = LOBDataModule(data_dir=cfg['data']['path'], batch_size=cfg['batch_size'])
    dm.setup()
    input_size = len(dm.train_ds.x[0])
    model = SimpleTCN(input_size=input_size, lr=cfg['lr'])
    for epoch in range(cfg['epochs']):
        for batch_idx, batch in enumerate(dm.train_dataloader()):
            loss = model.training_step(batch, batch_idx)
        print(f"epoch={epoch} loss={loss:.4f}")
    model.save(cfg.get('checkpoint', 'model.json'))


@hydra_main(config_path="configs", config_name="train/default.yaml")
def main(cfg) -> None:  # type: ignore[override]
    run_train(cfg)


if __name__ == '__main__':
    main()
