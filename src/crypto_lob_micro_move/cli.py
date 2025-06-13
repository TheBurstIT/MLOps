from pathlib import Path
from crypto_lob_micro_move.data import download
from crypto_lob_micro_move import train as train_module
from crypto_lob_micro_move.export.onnx import export_ckpt_to_onnx
from pathlib import Path

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

class Commands:
    def download_data(self, output_dir: str = "data/raw"):
        """Download dataset."""
        download.download_data(output_dir)

    def train(self, **overrides):
        """Run training with optional config overrides."""
        cfg_file = Path(__file__).resolve().parents[2] / "configs/train/default.yaml"
        cfg = _simple_yaml_load(cfg_file)
        for key, value in overrides.items():
            if key in cfg:
                cfg[key] = type(cfg[key])(value)
            elif "." in key:
                group, subkey = key.split(".", 1)
                if group not in cfg or not isinstance(cfg[group], dict):
                    cfg[group] = {}
                if subkey in cfg[group]:
                    cfg[group][subkey] = type(cfg[group][subkey])(value)
                else:
                    cfg[group][subkey] = value
        train_module.run_train(cfg)

    def export(self, ckpt: str, out: str):
        """Export a checkpoint to ONNX."""
        export_ckpt_to_onnx(ckpt, out)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Crypto LOB utility")
    sub = parser.add_subparsers(dest="command")

    dl = sub.add_parser("download-data")
    dl.add_argument("--output_dir", default="data/raw")

    tr = sub.add_parser("train")
    tr.add_argument("overrides", nargs="*")

    ex = sub.add_parser("export")
    ex.add_argument("--ckpt", required=True)
    ex.add_argument("--out", required=True)

    args = parser.parse_args()

    cmd = Commands()
    if args.command == "download-data":
        cmd.download_data(args.output_dir)
    elif args.command == "train":
        overrides = dict(item.split("=") for item in args.overrides)
        cmd.train(**overrides)
    elif args.command == "export":
        cmd.export(args.ckpt, args.out)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
