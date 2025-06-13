from __future__ import annotations

from pathlib import Path
import torch

from ..models.tcn_module import SimpleTCN


def export_ckpt_to_onnx(ckpt_path: str, out_path: str) -> None:
    model = SimpleTCN()
    state = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    dummy = torch.randn(1, 10, 30)
    torch.onnx.export(model, dummy, out_path, opset_version=17)
    print(f"Model exported to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    export_ckpt_to_onnx(args.ckpt, args.out)
