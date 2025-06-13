from crypto_lob_micro_move.data import download
from crypto_lob_micro_move import train as train_module
from crypto_lob_micro_move.export.onnx import export_ckpt_to_onnx
from pathlib import Path
from hydra import compose, initialize_config_dir

class Commands:
    def download_data(self, output_dir: str = "data/raw"):
        """Download dataset."""
        download.download_data(output_dir)

    def train(self, **overrides):
        """Run training with optional config overrides."""
        cfg_dir = Path(__file__).resolve().parents[2] / "configs"
        with initialize_config_dir(str(cfg_dir)):
            cfg = compose(config_name="train/default.yaml", overrides=[f"{k}={v}" for k, v in overrides.items()])
        train_module.run_train(cfg)

    def export(self, ckpt: str, out: str):
        """Export a checkpoint to ONNX."""
        export_ckpt_to_onnx(ckpt, out)


def main():
    import fire

    fire.Fire(Commands)

if __name__ == "__main__":
    main()
