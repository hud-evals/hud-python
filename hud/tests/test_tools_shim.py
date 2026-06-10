"""``hud.tools`` v5 compat: type redirects, computer markers, and no-ops.

``hud.tools`` is the real tools package; only symbols/submodules removed in the
v6 teardown go through the compat fallback (with a ``DeprecationWarning``).
"""

from __future__ import annotations

import warnings

import pytest


def test_real_tools_import_without_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        import hud.tools
        from hud.tools.agent import AgentTool

        bash = hud.tools.BashTool

    assert AgentTool.__module__ == "hud.tools.agent"
    assert bash.__module__.startswith("hud.tools.coding")


def test_result_types_redirect_to_agents_types() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from hud.tools.types import EvaluationResult

    # The real type (has the ``from_float`` constructor), not a no-op.
    assert EvaluationResult.from_float(0.5).reward == 0.5


def test_computer_tool_resolves_to_capability_marker() -> None:
    import hud.tools

    with pytest.warns(DeprecationWarning):
        computer_cls = hud.tools.HudComputerTool

    instance = computer_cls(width=800, height=600)
    assert getattr(instance, "_legacy_capability_kind", None) == "computer"


def test_removed_name_from_real_module_falls_back_to_noop() -> None:
    # ``GeminiEditTool`` was dropped in v6; importing it must not raise ImportError.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from hud.tools.coding import GeminiEditTool

        # No-op stand-in: constructs and calls without error.
        assert GeminiEditTool(anything=1)() is not None


def test_removed_submodule_resolves_names() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from hud.tools.filesystem import ReadTool

        assert ReadTool() is not None


def test_unknown_symbol_is_noop_not_error() -> None:
    import hud.tools

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        noop = hud.tools.SomethingThatNeverExisted
        assert noop() is not None


def test_hud_native_aliases_preserve_module_identity() -> None:
    import hud.native
    import hud.native.tools.base as native_base
    from hud.graders import combine
    from hud.tools.base import BaseTool

    assert native_base.BaseTool is BaseTool
    assert hud.native.combine is combine
