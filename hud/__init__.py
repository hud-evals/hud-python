"""hud-python.

tools for building, evaluating, and training AI agents.
"""

from __future__ import annotations

import warnings

# Apply patches to third-party libraries early, before other imports
from . import patches as _patches  # noqa: F401
from .environment import Environment, Scenario, ScenarioTool
from .eval import EvalContext
from .eval import run_eval as eval
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


def scenario(
    env: Environment,
    name: str,
    *,
    description: str | None = None,
) -> Scenario:
    """Load a scenario from an environment (local or remote).

    This is a convenience function for creating Scenario handles,
    especially for remote scenarios accessed via MCP.

    Args:
        env: Environment where the scenario is defined.
        name: Scenario name (with or without env prefix like "env:scenario").
        description: Optional description override.

    Returns:
        Scenario object that can create Tasks or be converted to tools.

    Example:
        ```python
        import hud

        # Connect to remote environment
        env = await hud.Environment.connect_hub("http://hub:8000")

        # Load scenario
        checkout = hud.scenario(env, "checkout")

        # Create task from scenario
        task = checkout(user="alice", product_id="123")

        # Or convert to a tool for sub-agent use
        tool = checkout.as_agent_tool("claude")
        ```
    """
    # Check if scenario is already registered locally
    if hasattr(env, "get_scenario"):
        local_scenario = env.get_scenario(name)
        if local_scenario is not None:
            return local_scenario

    # Otherwise, create a remote scenario handle
    return Scenario.from_remote(env, name, description=description)


__all__ = [
    "Environment",
    "EvalContext",
    "Scenario",
    "ScenarioTool",
    "eval",
    "instrument",
    "scenario",
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
