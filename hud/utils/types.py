from __future__ import annotations
from typing import Any, Callable, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

def with_signature(
    params_cls: Callable[P, Any],
) -> Callable[[Callable[..., R]], Callable[P, R]]:
    """Decorator that gives a method the signature of a Pydantic model."""
    def decorator(method: Callable[..., R]) -> Callable[P, R]:
        return method  # type: ignore[return-value]
    return decorator