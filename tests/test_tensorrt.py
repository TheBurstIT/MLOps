from crypto_lob_micro_move.tensorrt import convert_onnx_to_trt
from unittest import mock


def test_convert_onnx_to_trt_calls_subprocess():
    with mock.patch('subprocess.run') as run:
        convert_onnx_to_trt('model.onnx', 'model.plan')
        run.assert_called_with(['onnx2trt', 'model.onnx', '-o', 'model.plan'], check=True)

