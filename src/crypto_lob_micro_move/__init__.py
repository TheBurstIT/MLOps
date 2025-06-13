try:
    from .cli import main
except Exception:  # pragma: no cover - optional dependency
    main = None

from .train import train
from .tensorrt import convert_onnx_to_trt

__all__ = ["main", "train", "convert_onnx_to_trt"]
