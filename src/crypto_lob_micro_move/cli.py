from pathlib import Path
from crypto_lob_micro_move.data import download
from crypto_lob_micro_move import train as train_module
import yaml
from crypto_lob_micro_move.export.onnx import export_ckpt_to_onnx

class Commands:
    def download_data(self, output_dir: str = "data/raw"):
        """Download dataset."""
        download.download_data(output_dir)

    def train(self, **overrides):
        """Run training with optional config overrides."""
        cfg_file = Path(__file__).resolve().parents[1] / "configs/train/default.yaml"
        with open(cfg_file) as f:
            cfg = yaml.safe_load(f)
        for key, value in overrides.items():
            if key in cfg:
                cfg[key] = type(cfg[key])(value)
            elif "." in key:
                group, subkey = key.split(".", 1)
                if group in cfg and isinstance(cfg[group], dict) and subkey in cfg[group]:
                    cfg[group][subkey] = type(cfg[group][subkey])(value)
        train_module.run_train(cfg)

    def export(self, ckpt: str, out: str):
        """Export a checkpoint to ONNX."""
        export_ckpt_to_onnx(ckpt, out)


def main():
    import fire

    fire.Fire(Commands)

if __name__ == "__main__":
    main()
