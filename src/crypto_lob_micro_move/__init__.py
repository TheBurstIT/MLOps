try:
    from .cli import main
except Exception:  # pragma: no cover - optional dependency
    main = None

from .train import run_train

__all__ = ["main", "run_train"]
