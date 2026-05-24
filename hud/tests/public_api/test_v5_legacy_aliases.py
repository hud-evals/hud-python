"""Current v5 legacy alias contracts.

Keeping these checks separate makes intentional v6 cleanup straightforward:
the cleanup can edit or remove this file without touching the normal public
surface tests.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

import pytest


def test_trace_warns_and_delegates_to_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    import hud

    sentinel = object()
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def fake_eval(*args: Any, **kwargs: Any) -> object:
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(hud, "eval", fake_eval)

    with pytest.warns(DeprecationWarning, match=r"hud\.trace\(\) is deprecated"):
        result = hud.trace("task", variants={"model": ["test"]}, group=2)

    assert result is sentinel
    assert calls == [(("task",), {"variants": {"model": ["test"]}, "group": 2})]


def test_load_dataset_warns_and_delegates_to_load_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hud.datasets as datasets
    import hud.datasets.loader as loader

    sentinel = [{"slug": "task"}]
    calls: list[tuple[str, bool]] = []

    def fake_load_tasks(source: str, *, raw: bool = False) -> list[dict[str, str]]:
        calls.append((source, raw))
        return sentinel

    monkeypatch.setattr(loader, "load_tasks", fake_load_tasks)

    with pytest.warns(DeprecationWarning, match=r"load_dataset\(\) is deprecated"):
        result = datasets.load_dataset("local-or-remote-source", raw=True)

    assert result is sentinel
    assert calls == [("local-or-remote-source", True)]


def test_tool_router_aliases_environment_mcp_router() -> None:
    import hud.environment as environment

    assert environment.ToolRouter is environment.MCPRouter


def test_task_reexport_paths_share_the_same_task_model() -> None:
    import hud.types as types

    eval_module = import_module("hud.eval")
    task_module = import_module("hud.eval.task")

    assert types.Task is eval_module.Task is task_module.Task


def test_server_mcp_server_public_and_deep_paths_match() -> None:
    import hud.server as server

    server_module = import_module("hud.server.server")

    assert server.MCPServer is server_module.MCPServer


def test_router_public_paths_are_importable_without_identity_constraint() -> None:
    import hud.environment as environment
    import hud.server as server

    assert environment.MCPRouter is not None
    assert server.MCPRouter is not None
