try:
    from .cli import main
except Exception:  # pragma: no cover - optional dependency
    main = None

from .train import train

__all__ = ["main", "train"]
