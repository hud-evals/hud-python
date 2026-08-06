"""``hud.tools`` v5 compat: type redirects, computer markers, and no-ops.

``hud.tools`` was removed in v6 (shell/file/computer/browser access is a
capability, not a tool). The whole package now resolves through the compat
fallback, each access emitting a ``DeprecationWarning``.
"""

from __future__ import annotations

import importlib
import warnings
from typing import Any

import pytest

from hud.environment import Answer


def _legacy_attr(module: str, name: str) -> Any:
    return getattr(importlib.import_module(module), name)


def test_basetool_and_agenttool_resolve_to_noops() -> None:
    # ``BaseTool`` / ``AgentTool`` were removed in v6; importing them must not
    # raise, but resolves to a no-op stand-in with a DeprecationWarning.
    tools = importlib.import_module("hud.tools")

    for name in ("BaseTool", "AgentTool"):
        with pytest.warns(DeprecationWarning):
            cls = getattr(tools, name)
        assert cls.__module__ == "hud._legacy"
        assert cls() is not None


def test_result_types_redirect_to_their_v6_homes() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        AgentAnswer = _legacy_attr("hud.tools.types", "AgentAnswer")
        EvaluationResult = _legacy_attr("hud.tools.types", "EvaluationResult")
        ScenarioResult = _legacy_attr("hud.tools.types", "ScenarioResult")
        TextContent = _legacy_attr("hud.tools.types", "TextContent")

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
        ToolError = _legacy_attr("hud.tools.bash", "ToolError")
        ContentResult = _legacy_attr("hud.tools.types", "ContentResult")

    combined = ContentResult(output="a", error="e1") + ContentResult(output="b", error="e2")
    assert combined.output == "ab"
    assert combined.error == "e1e2"

    blocks = ContentResult(output="hi", base64_image="iVBORw0KGgo=").to_content_blocks()
    assert [type(b).__name__ for b in blocks] == ["TextContent", "ImageContent"]

    assert issubclass(ToolError, Exception)
    with pytest.raises(ToolError, match="boom"):
        raise ToolError("boom")


def test_computer_tool_resolves_to_capability_marker() -> None:
    tools = importlib.import_module("hud.tools")

    with pytest.warns(DeprecationWarning):
        computer_cls = getattr(tools, "HudComputerTool")

    instance = computer_cls(width=800, height=600)
    assert getattr(instance, "_legacy_capability_kind", None) == "computer"


def test_shell_tool_resolves_to_capability_marker() -> None:
    # ``BashTool``/``EditTool`` were dropped in v6; a registered one becomes an
    # ``ssh`` capability at serve time via the shell marker.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        BashTool = _legacy_attr("hud.tools", "BashTool")
        EditTool = _legacy_attr("hud.tools.coding", "EditTool")

    for tool_cls in (BashTool, EditTool):
        instance = tool_cls(base_path="/tmp")
        assert getattr(instance, "_legacy_capability_kind", None) == "shell"


def test_removed_name_from_real_module_falls_back_to_noop() -> None:
    # ``BaseHub`` was dropped in v6; importing it must not raise ImportError.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        BaseHub = _legacy_attr("hud.tools.base", "BaseHub")

        # No-op stand-in: constructs and calls without error.
        assert BaseHub(anything=1)() is not None


def test_removed_submodule_resolves_names() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        ReadTool = _legacy_attr("hud.tools.filesystem", "ReadTool")

        assert ReadTool() is not None


def test_jupyter_and_playwright_resolve_to_noops() -> None:
    # Dropped in v6: registering them in a v5 env silently does nothing.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        JupyterTool = _legacy_attr("hud.tools", "JupyterTool")
        PlaywrightTool = _legacy_attr("hud.tools", "PlaywrightTool")
        deep_playwright = _legacy_attr("hud.tools.playwright", "PlaywrightTool")

    for tool_cls in (JupyterTool, PlaywrightTool, deep_playwright):
        instance = tool_cls(cdp_url="http://localhost:9222")
        assert instance() is not None


def test_unknown_symbol_is_noop_not_error() -> None:
    tools = importlib.import_module("hud.tools")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        noop = getattr(tools, "SomethingThatNeverExisted")
        assert noop() is not None


def test_hud_native_aliases_preserve_module_identity() -> None:
    from hud.graders import combine

    native = importlib.import_module("hud.native")
    native_base = importlib.import_module("hud.native.tools.base")
    BaseTool = _legacy_attr("hud.tools.base", "BaseTool")

    assert getattr(native_base, "BaseTool") is BaseTool
    assert getattr(native, "combine") is combine


def test_hud_services_alias_resolves_chat() -> None:
    from hud.eval.chat import Chat

    legacy_chat = _legacy_attr("hud.services", "Chat")

    assert legacy_chat is Chat
