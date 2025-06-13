from pathlib import Path
from .datamodules.lob_dm import LOBDataModule
from .models.tcn_module import SimpleTCN

try:  # use hydra if installed
    from hydra import main as hydra_main
    from omegaconf import DictConfig
except Exception:  # pragma: no cover - optional dependency
    def _simple_yaml_load(path: Path) -> dict:
        data = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                key, value = line.split(":", 1)
                value = value.strip()
                if value.isdigit():
                    value = int(value)
                else:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                data[key.strip()] = value
        return data

    def hydra_main(config_path: str | None = None, config_name: str | None = None):
        def wrapper(func):
            def inner(*args, **kwargs):
                cfg_file = Path(config_path or "") / (config_name or "config.yaml")
                cfg = _simple_yaml_load(cfg_file)
                return func(cfg, *args, **kwargs)

            return inner

        return wrapper



def run_train(cfg) -> None:
    data_dir = cfg.get('data', {}).get('path', 'data/raw')
    batch_size = cfg.get('batch_size', 4)
    lr = cfg.get('lr', 1e-3)
    epochs = cfg.get('epochs', 1)
    checkpoint = cfg.get('checkpoint', 'model.json')

    dm = LOBDataModule(data_dir=data_dir, batch_size=batch_size)
    dm.setup()
    input_size = len(dm.train_ds.x[0])
    model = SimpleTCN(input_size=input_size, lr=lr)
    for epoch in range(epochs):
        for batch_idx, batch in enumerate(dm.train_dataloader()):
            loss = model.training_step(batch, batch_idx)
        print(f"epoch={epoch} loss={loss:.4f}")
    model.save(checkpoint)


@hydra_main(config_path="configs", config_name="train/default.yaml")
def main(cfg) -> None:  # type: ignore[override]
    run_train(cfg)


if __name__ == '__main__':
    main()
