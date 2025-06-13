from pathlib import Path
from crypto_lob_micro_move.data import download
from crypto_lob_micro_move import train as train_module
from crypto_lob_micro_move.export.onnx import export_ckpt_to_onnx

class Commands:
    def download_data(self, output_dir: str = "data/raw"):
        """Download dataset."""
        download.download_data(output_dir)

    def train(self, **overrides):
        """Run training with optional config overrides."""
        cfg = {
            'data': {'path': 'data/raw'},
            'batch_size': 4,
            'lr': 0.001,
            'epochs': 1,
        }
        for k, v in overrides.items():
            if k in cfg:
                cfg[k] = type(cfg[k])(v)
            elif k == "data.path":
                cfg["data"]["path"] = v
        train_module.run_train(cfg)

    def export(self, ckpt: str, out: str):
        """Export a checkpoint to ONNX."""
        export_ckpt_to_onnx(ckpt, out)


def main():
    import fire

    fire.Fire(Commands)

if __name__ == "__main__":
    main()
