from __future__ import annotations

import json


def _load_weights(path: str) -> dict:
    """Load weights saved as JSON."""
    with open(path) as f:
        return json.load(f)


def export_ckpt_to_onnx(ckpt_path: str, out_path: str) -> None:
    """Copy weights to ``out_path`` for demonstration purposes."""
    weights = _load_weights(ckpt_path)
    with open(out_path, "w") as f:
        json.dump(weights, f)
    print(f"Model exported to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    export_ckpt_to_onnx(args.ckpt, args.out)
