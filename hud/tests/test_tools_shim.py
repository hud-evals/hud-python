"""The deprecated ``hud.tools`` shim: redirects, computer markers, and no-ops.

Lives outside ``hud.tools`` because the shim's meta-path finder intercepts every
``hud.tools.*`` submodule (so test modules can't live under that package).
"""

from __future__ import annotations

import warnings

import pytest


def test_tool_redirects_to_native_location() -> None:
    # A submodule import only warns once (module caching), so assert the redirect
    # result rather than the one-shot warning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from hud.tools.agent import AgentTool

    assert AgentTool.__module__ == "hud.native.tools.agent"


def test_result_types_redirect_to_agents_types() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from hud.tools.types import EvaluationResult

    # The real type (has the ``from_float`` constructor), not a no-op.
    assert EvaluationResult.from_float(0.5).reward == 0.5


def test_top_level_tool_name_redirects() -> None:
    import hud.tools

    with pytest.warns(DeprecationWarning):
        bash = hud.tools.BashTool

    assert bash.__module__.startswith("hud.native.tools")


def test_computer_tool_resolves_to_capability_marker() -> None:
    import hud.tools

    with pytest.warns(DeprecationWarning):
        computer_cls = hud.tools.HudComputerTool

    instance = computer_cls(width=800, height=600)
    assert getattr(instance, "_legacy_capability_kind", None) == "computer"


def test_removed_name_from_redirected_module_falls_back_to_noop() -> None:
    # ``GeminiEditTool`` was dropped in v6; importing it must not raise ImportError.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from hud.tools.coding import GeminiEditTool

        # No-op stand-in: constructs and calls without error.
        assert GeminiEditTool(anything=1)() is not None


def test_unknown_symbol_is_noop_not_error() -> None:
    import hud.tools

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        noop = hud.tools.SomethingThatNeverExisted
        assert noop() is not None
