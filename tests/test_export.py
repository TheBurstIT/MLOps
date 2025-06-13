from crypto_lob_micro_move.models.tcn_module import SimpleTCN
from crypto_lob_micro_move.export.onnx import export_ckpt_to_onnx
import torch
import onnxruntime as ort


def test_onnx_vs_torch_same_output(tmp_path):
    model = SimpleTCN()
    ckpt = tmp_path / "model.ckpt"
    trainer = None
    torch.save(model.state_dict(), ckpt)
    # Save checkpoint using Lightning to be loaded by SimpleTCN
    torch.save(model.state_dict(), ckpt)

    onnx_path = tmp_path / "model.onnx"
    export_ckpt_to_onnx(str(ckpt), str(onnx_path))

    dummy = torch.randn(1, 10, 30)
    torch_out = model(dummy)
    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name
    result = session.run(None, {input_name: dummy.numpy()})[0]
    assert torch.allclose(torch_out.detach(), torch.tensor(result), atol=1e-5)
