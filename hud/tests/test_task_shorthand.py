from __future__ import annotations

from typing import Any

from hud.types import MCPToolCall, Task


def _mk_task(**kwargs: Any) -> Task:
    base = {
        "prompt": "x",
        "mcp_config": {"browser": {"url": "http://localhost"}},
    }
    base.update(kwargs)
    return Task(**base)


def test_setup_shorthand_name_arguments():
    t = _mk_task(setup_tool={"name": "navigate", "arguments": {"url": "https://a.com"}})
    assert isinstance(t.setup_tool, MCPToolCall)
    assert t.setup_tool.name == "navigate"
    assert t.setup_tool.arguments == {"url": "https://a.com"}


def test_setup_shorthand_single_key():
    t = _mk_task(setup_tool={"navigate": {"url": "https://a.com"}})
    assert isinstance(t.setup_tool, MCPToolCall)
    assert t.setup_tool.name == "navigate"
    assert t.setup_tool.arguments == {"url": "https://a.com"}


def test_setup_recursive_wrap():
    t = _mk_task(setup_tool={"setup": {"navigate": {"url": "https://a.com"}}})
    assert isinstance(t.setup_tool, MCPToolCall)
    assert t.setup_tool.name == "setup"
    assert t.setup_tool.arguments == {
        "name": "navigate",
        "arguments": {"url": "https://a.com"},
    }


def test_evaluate_shorthand_name_arguments():
    t = _mk_task(evaluate_tool={"name": "page_contains", "arguments": {"q": "hud.so"}})
    assert isinstance(t.evaluate_tool, MCPToolCall)
    assert t.evaluate_tool.name == "page_contains"
    assert t.evaluate_tool.arguments == {"q": "hud.so"}


def test_evaluate_shorthand_single_key():
    t = _mk_task(evaluate_tool={"page_contains": {"q": "hud.so"}})
    assert isinstance(t.evaluate_tool, MCPToolCall)
    assert t.evaluate_tool.name == "page_contains"
    assert t.evaluate_tool.arguments == {"q": "hud.so"}


def test_evaluate_recursive_wrap():
    t = _mk_task(evaluate_tool={"evaluate": {"page_contains": {"q": "hud.so"}}})
    assert isinstance(t.evaluate_tool, MCPToolCall)
    assert t.evaluate_tool.name == "evaluate"
    assert t.evaluate_tool.arguments == {
        "name": "page_contains",
        "arguments": {"q": "hud.so"},
    }


def test_list_forms_supported():
    t = _mk_task(
        setup_tool=[{"navigate": {"url": "https://a.com"}}],
        evaluate_tool=[{"page_contains": {"q": "hud.so"}}],
    )
    assert isinstance(t.setup_tool, list)
    assert t.setup_tool[0].name == "navigate"
    assert t.setup_tool[0].arguments == {"url": "https://a.com"}
    assert isinstance(t.evaluate_tool, list)
    assert t.evaluate_tool[0].name == "page_contains"
    assert t.evaluate_tool[0].arguments == {"q": "hud.so"}
