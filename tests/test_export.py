from crypto_lob_micro_move.models.tcn_module import SimpleTCN
from crypto_lob_micro_move.export.onnx import export_ckpt_to_onnx
import random
import json


def test_onnx_vs_torch_same_output(tmp_path):
    model = SimpleTCN(input_size=30)
    ckpt = tmp_path / "model.json"
    model.save(ckpt)

    onnx_path = tmp_path / "model.onnx"
    export_ckpt_to_onnx(str(ckpt), str(onnx_path))

    dummy = [[random.random() for _ in range(30)]]
    orig = model(dummy)
    with open(onnx_path) as f:
        data = json.load(f)
    loaded = []
    for row in dummy:
        logits = []
        for j in range(3):
            s = sum(row[i] * data["weights"][i][j] for i in range(len(row))) + data["bias"][j]
            logits.append(s)
        loaded.append(logits)
    assert all(abs(o - l) < 1e-6 for o, l in zip(orig[0], loaded[0]))
