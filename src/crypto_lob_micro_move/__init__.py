try:
    from .cli import main
except Exception:  # pragma: no cover - optional dependency
    main = None

from .train import train, TrainConfig
from .tensorrt import convert_onnx_to_trt

__all__ = ["main", "train", "TrainConfig", "convert_onnx_to_trt"]
