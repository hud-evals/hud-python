"""hud-python.

tools for building, evaluating, and training AI agents.
"""

from __future__ import annotations

import warnings
from importlib import import_module
from typing import TYPE_CHECKING

# Initialize the foundational types module first. hud.types and hud.eval.task
# form an intentional mutual re-export cycle (hud.types.Trace references Task;
# hud.eval.task references MCPToolCall). That cycle only resolves cleanly when
# hud.types is the entry point, so loading it here -- before any subpackage --
# makes import order irrelevant for downstream code and guarantees Trace's
# forward reference is resolved after `import hud`.
import hud.types  # noqa: F401

# hud.eval() is the primary entry point and is light to import. Binding it
# eagerly keeps `hud.eval(...)` callable even after the `hud.eval` submodule is
# imported internally (a submodule import would otherwise shadow a lazy
# attribute of the same name). Runtime patches are applied lazily inside
# run_eval / the runtime packages, not here -- see hud/_runtime.py.
from hud.eval import run_eval as eval

if TYPE_CHECKING:
    from hud.environment import Environment
    from hud.eval import EvalContext
    from hud.services import Chat
    from hud.telemetry.instrument import instrument


def trace(*args: object, **kwargs: object) -> EvalContext:
    """Deprecated: Use hud.eval() instead.

    .. deprecated:: 0.5.2
        hud.trace() is deprecated. Use hud.eval() or env.eval() instead.
    """
    warnings.warn(
        "hud.trace() is deprecated. Use hud.eval() or env.eval() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return eval(*args, **kwargs)  # type: ignore[arg-type, return-value]


# Heavy runtime symbols are imported lazily so that `import hud` (and importing
# the data-model modules like `hud.types`) stays cheap and side-effect-free.
# Importing the backing package applies the runtime patches via
# activate_runtime() in that package's __init__.
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "Environment": ("hud.environment", "Environment"),
    "EvalContext": ("hud.eval", "EvalContext"),
    "Chat": ("hud.services", "Chat"),
    "instrument": ("hud.telemetry.instrument", "instrument"),
}


def __getattr__(name: str) -> object:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'hud' has no attribute {name!r}")
    module_name, attr = target
    return getattr(import_module(module_name), attr)


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_EXPORTS})


__all__ = [
    "Chat",
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
