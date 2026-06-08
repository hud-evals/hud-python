"""The deprecated ``hud.tools`` shim redirects known moved imports only.

Lives outside ``hud.tools`` because the shim's meta-path finder intercepts every
``hud.tools.*`` submodule (so test modules can't live under that package).
"""

from __future__ import annotations

import importlib
import warnings

import pytest


def test_tool_redirects_to_native_location() -> None:
    # A submodule import only warns once (module caching), so assert the redirect
    # result rather than the one-shot warning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        module = importlib.import_module("hud.tools.agent")
        agent_tool = module.AgentTool

    assert agent_tool.__module__ == "hud.native.tools.agent"


def test_result_types_redirect_to_agents_types() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        module = importlib.import_module("hud.tools.types")
        evaluation_result = module.EvaluationResult

    # The real type (has the ``from_float`` constructor), not a no-op.
    assert evaluation_result.from_float(0.5).reward == 0.5


def test_top_level_tool_name_redirects() -> None:
    import hud.tools

    with pytest.warns(DeprecationWarning):
        bash = hud.tools.BashTool

    assert bash.__module__.startswith("hud.native.tools")


def test_removed_computer_tool_resolves_to_capability_marker() -> None:
    import hud.tools

    with pytest.warns(DeprecationWarning):
        computer_cls = hud.tools.HudComputerTool

    instance = computer_cls(width=800, height=600)
    assert getattr(instance, "_legacy_capability_kind", None) == "computer"


def test_removed_computer_module_resolves_to_capability_marker() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        module = importlib.import_module("hud.tools.computer")
        computer_cls = module.AnthropicComputerTool

    assert getattr(computer_cls(), "_legacy_capability_kind", None) == "computer"


def test_removed_name_from_redirected_module_raises_import_error() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(AttributeError):
            module = importlib.import_module("hud.tools.coding")
            _ = module.GeminiEditTool


def test_unknown_symbol_raises_attribute_error() -> None:
    import hud.tools

    with pytest.raises(AttributeError):
        _ = hud.tools.SomethingThatNeverExisted
