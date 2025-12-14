"""hud-python.

tools for building, evaluating, and training AI agents.
"""

from __future__ import annotations

import warnings

# Apply patches to third-party libraries early, before other imports
from . import patches as _patches  # noqa: F401
from .environment import Environment
from .eval import EvalContext
from .eval import run_eval as eval
from .telemetry.instrument import instrument


def trace(*args: object, **kwargs: object) -> EvalContext:
    """Deprecated: Use hud.eval() instead.

    .. deprecated:: 0.5.0
        hud.trace() is deprecated. Use hud.eval() or env.eval() instead.
    """
    warnings.warn(
        "hud.trace() is deprecated. Use hud.eval() or env.eval() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return eval(*args, **kwargs)  # type: ignore[arg-type]


__all__ = [
    "Environment",
    "EvalContext",
    "eval",
    "instrument",
    "trace",  # Deprecated alias for eval
]

try:
    from .version import __version__
except ImportError:
    __version__ = "unknown"

try:
    from .utils.pretty_errors import install_pretty_errors

    install_pretty_errors()
except Exception:  # noqa: S110
    pass
