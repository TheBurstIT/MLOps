import subprocess


def convert_onnx_to_trt(input_model: str, output_model: str) -> None:
    """Convert an ONNX model to TensorRT using the ``onnx2trt`` CLI."""
    cmd = ["onnx2trt", input_model, "-o", output_model]
    subprocess.run(cmd, check=True)

