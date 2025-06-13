import numpy as np
from .datamodules.lob_dm import LOBDataModule
from .models.tcn_module import SimpleTCN


def run_train(cfg) -> None:
    dm = LOBDataModule(data_dir=cfg['data']['path'], batch_size=cfg['batch_size'])
    dm.setup()
    input_size = dm.train_ds.x.shape[1]
    model = SimpleTCN(input_size=input_size, lr=cfg['lr'])
    for epoch in range(cfg['epochs']):
        for batch_idx, batch in enumerate(dm.train_dataloader()):
            loss = model.training_step(batch, batch_idx)
        print(f"epoch={epoch} loss={loss:.4f}")
    model.save(cfg.get('checkpoint', 'model.npz'))


def main():
    import sys

    cfg = {
        'data': {'path': 'data/raw'},
        'batch_size': 4,
        'lr': 0.001,
        'epochs': 1,
    }
    overrides = dict(arg.split('=') for arg in sys.argv[1:])
    for k, v in overrides.items():
        if k in cfg:
            cfg[k] = type(cfg[k])(v)
        elif k == 'data.path':
            cfg['data']['path'] = v
    run_train(cfg)


if __name__ == '__main__':
    main()
