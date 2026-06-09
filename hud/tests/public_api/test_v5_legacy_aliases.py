"""Current v5 legacy alias contracts.

Keeping these checks separate makes intentional v6 cleanup straightforward:
the cleanup can edit or remove this file without touching the normal public
surface tests.
"""

from __future__ import annotations

from importlib import import_module


def test_tool_router_aliases_environment_mcp_router() -> None:
    import hud.environment as environment

    assert environment.ToolRouter is environment.MCPRouter


def test_task_reexport_paths_share_the_same_task_model() -> None:
    eval_module = import_module("hud.eval")
    task_module = import_module("hud.eval.task")

    assert eval_module.Task is task_module.Task


def test_server_mcp_server_public_and_deep_paths_match() -> None:
    import hud.server as server

    server_module = import_module("hud.server.server")

    assert server.MCPServer is server_module.MCPServer


def test_router_public_paths_are_importable_without_identity_constraint() -> None:
    import hud.environment as environment
    import hud.server as server

    assert environment.MCPRouter is not None
    assert server.MCPRouter is not None
