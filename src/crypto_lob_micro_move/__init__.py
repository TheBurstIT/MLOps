try:  # pragma: no cover - CLI may rely on optional deps
    from .cli import main
except Exception:
    main = None

__all__ = ["main"]
