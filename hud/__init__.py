"""hud-python.

tools for building, evaluating, and training AI agents.
"""

from __future__ import annotations

import warnings

from .environment import Environment
from .eval import Taskset as Taskset
from .eval import Variant as Variant
from .eval import launch as launch
from .eval import variant as variant
from .eval.context import EvalContext
from .services import Chat
from .telemetry.instrument import instrument


def eval(*args: object, **kwargs: object) -> object:
    """Deprecated v5 eval entry point."""
    raise RuntimeError(
        "hud.eval() is deprecated and no longer executes evaluations directly; "
        "use hud.eval.Taskset(...).run(...) or the `hud eval` CLI."
    )


def trace(*args: object, **kwargs: object) -> object:
    """Deprecated v5 alias for :func:`hud.eval`."""
    warnings.warn(
        "hud.trace() is deprecated: use hud.eval.Taskset(...).run(...) or the `hud eval` CLI.",
        DeprecationWarning,
        stacklevel=2,
    )
    return eval(*args, **kwargs)


__all__ = [
    "Chat",
    "Environment",
    "EvalContext",
    "eval",
    "instrument",
    "trace",
]

try:
    from .version import __version__
except ImportError:
    __version__ = "unknown"
