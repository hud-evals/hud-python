"""hud-python.

tools for building, evaluating, and training AI agents.
"""

from __future__ import annotations

import warnings

# Apply patches to third-party libraries early, before other imports
from . import patches as _patches  # noqa: F401
from .environment import Environment, ScenarioArg, ScenarioInfo
from .eval import EvalContext
from .eval import run_eval as eval
from .scenario_chat import (
    ChatEvent,
    ScenarioChatResult,
    ScenarioChatSession,
    ScenarioChatTurnResult,
    run_scenario_chat,
    run_scenario_chat_interactive,
)
from .telemetry.instrument import instrument


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
    return eval(*args, **kwargs)  # type: ignore[arg-type]


__all__ = [
    "ChatEvent",
    "Environment",
    "EvalContext",
    "ScenarioArg",
    "ScenarioChatResult",
    "ScenarioChatSession",
    "ScenarioChatTurnResult",
    "ScenarioInfo",
    "eval",
    "instrument",
    "run_scenario_chat",
    "run_scenario_chat_interactive",
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
