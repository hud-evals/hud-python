"""``hud.tools`` v5 compat: type redirects, computer markers, and no-ops.

``hud.tools`` was removed in v6 (shell/file/computer/browser access is a
capability, not a tool). The whole package now resolves through the compat
fallback, each access emitting a ``DeprecationWarning``.
"""

from __future__ import annotations

import warnings

import pytest

from hud.environment import Answer


def test_basetool_and_agenttool_resolve_to_noops() -> None:
    # ``BaseTool`` / ``AgentTool`` were removed in v6; importing them must not
    # raise, but resolves to a no-op stand-in with a DeprecationWarning.
    import hud.tools

    for name in ("BaseTool", "AgentTool"):
        with pytest.warns(DeprecationWarning):
            cls = getattr(hud.tools, name)
        assert cls.__module__ == "hud._legacy"
        assert cls() is not None


def test_result_types_redirect_to_their_v6_homes() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from hud.tools.types import AgentAnswer, EvaluationResult, ScenarioResult, TextContent

    # The real types (not no-ops): graders for results, mcp.types for blocks.
    assert EvaluationResult.from_float(0.5).reward == 0.5
    assert ScenarioResult is EvaluationResult  # renamed in v6
    assert AgentAnswer is Answer  # renamed in v6
    assert TextContent(text="x", type="text").text == "x"


def test_quarantined_v5_shapes_still_work() -> None:
    # ContentResult and ToolError have no v6 counterpart; they live in
    # hud._legacy and keep their v5 behavior for deployed environments.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from hud.tools.bash import ToolError  # type: ignore[import-not-found]
        from hud.tools.types import ContentResult

    combined = ContentResult(output="a", error="e1") + ContentResult(output="b", error="e2")
    assert combined.output == "ab"
    assert combined.error == "e1e2"

    blocks = ContentResult(output="hi", base64_image="iVBORw0KGgo=").to_content_blocks()
    assert [type(b).__name__ for b in blocks] == ["TextContent", "ImageContent"]

    assert issubclass(ToolError, Exception)
    with pytest.raises(ToolError, match="boom"):
        raise ToolError("boom")


def test_computer_tool_resolves_to_capability_marker() -> None:
    import hud.tools

    with pytest.warns(DeprecationWarning):
        computer_cls = hud.tools.HudComputerTool

    instance = computer_cls(width=800, height=600)
    assert getattr(instance, "_legacy_capability_kind", None) == "computer"


def test_shell_tool_resolves_to_capability_marker() -> None:
    # ``BashTool``/``EditTool`` were dropped in v6; a registered one becomes an
    # ``ssh`` capability at serve time via the shell marker.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from hud.tools import BashTool
        from hud.tools.coding import EditTool

    for tool_cls in (BashTool, EditTool):
        instance = tool_cls(base_path="/tmp")
        assert getattr(instance, "_legacy_capability_kind", None) == "shell"


def test_removed_name_from_real_module_falls_back_to_noop() -> None:
    # ``BaseHub`` was dropped in v6; importing it must not raise ImportError.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from hud.tools.base import BaseHub

        # No-op stand-in: constructs and calls without error.
        assert BaseHub(anything=1)() is not None


def test_removed_submodule_resolves_names() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from hud.tools.filesystem import ReadTool

        assert ReadTool() is not None


def test_jupyter_and_playwright_resolve_to_noops() -> None:
    # Dropped in v6: registering them in a v5 env silently does nothing.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from hud.tools import JupyterTool, PlaywrightTool
        from hud.tools.playwright import PlaywrightTool as deep_playwright

    for tool_cls in (JupyterTool, PlaywrightTool, deep_playwright):
        instance = tool_cls(cdp_url="http://localhost:9222")
        assert instance() is not None


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


def test_hud_services_alias_resolves_chat() -> None:
    from hud.eval.chat import Chat
    from hud.services import Chat as legacy_chat  # type: ignore[import-not-found]

    assert legacy_chat is Chat
