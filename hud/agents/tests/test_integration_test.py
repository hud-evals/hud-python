"""The local authoring agent (integration_test)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from hud.agents.integration_test import IntegrationTestAgent
from hud.agents.types import IntegrationTestConfig, ToolStep
from hud.types import MCPToolCall


def _fake_run(*, validation: list[Any] | None, tools: dict[str, str]) -> Any:
    """Minimal Run stand-in: manifest + open() returning per-capability MCP clients."""

    class FakeClient:
        async def call_tool(self, name: str, args: dict[str, Any]):
            served = tools.get(name)
            if served is None:
                from mcp.types import TextContent

                from hud.types import MCPToolResult

                return MCPToolResult(
                    content=[TextContent(type="text", text=f"unknown tool: {name!r}")],
                    isError=True,
                )
            from mcp.types import TextContent

            from hud.types import MCPToolResult

            return MCPToolResult(content=[TextContent(type="text", text=served)])

    run = SimpleNamespace()
    run.validation = validation
    run.recorded = []  # type: list[Any]
    run.trace = SimpleNamespace(status="running", stop_reason=None)

    def record(step):
        run.recorded.append(step)

    run.record = record
    run.client = SimpleNamespace()
    run.client.manifest = SimpleNamespace(
        bindings=[
            SimpleNamespace(name="bash", protocol="mcp"),
        ]
    )
    run.client.open = AsyncMock(return_value=FakeClient())
    return run


@pytest.mark.asyncio
async def test_dispatches_validation_calls_and_records_tool_steps() -> None:
    agent = IntegrationTestAgent(IntegrationTestConfig())
    run = _fake_run(
        validation=[
            MCPToolCall(name="bash", arguments={"command": "echo 'golden' > answer.txt"}),
            {"name": "bash", "arguments": {"command": "chmod +x answer.txt"}},
        ],
        tools={"bash": "ok"},
    )

    await agent(run)

    assert len(run.recorded) == 2
    assert all(isinstance(step, ToolStep) for step in run.recorded)
    assert run.recorded[0].call.name == "bash"


@pytest.mark.asyncio
async def test_empty_validation_is_a_noop() -> None:
    agent = IntegrationTestAgent(IntegrationTestConfig())
    run = _fake_run(validation=[], tools={})

    await agent(run)

    assert run.recorded == []


@pytest.mark.asyncio
async def test_invalid_entries_are_skipped_without_crashing() -> None:
    agent = IntegrationTestAgent(IntegrationTestConfig())
    run = _fake_run(
        validation=[{"not": "a tool call"}, 42, MCPToolCall(name="bash", arguments={})],
        tools={"bash": "ok"},
    )

    await agent(run)

    assert len(run.recorded) == 1


def _fake_run_ssh(*, validation: list[Any]) -> Any:
    """Run stand-in whose workspace is published over SSH (ssh/2)."""

    from hud.capabilities import SSHClient

    class FakeSSHClient(SSHClient):
        def __init__(self) -> None:
            pass

        async def run(self, *args: object, **kwargs: Any) -> Any:
            from types import SimpleNamespace

            assert args[:3] == ("bash", "-lc", "echo 'golden' > answer.txt")
            return SimpleNamespace(stdout=b"staged\n", stderr=b"", returncode=0)

    run = _fake_run(validation=validation, tools={})
    run.client.manifest = SimpleNamespace(
        bindings=[SimpleNamespace(name="workspace", protocol="ssh/2")]
    )
    run.client.open = AsyncMock(return_value=FakeSSHClient())
    return run


@pytest.mark.asyncio
async def test_dispatches_validation_over_ssh_workspace() -> None:
    from hud.agents.types import ToolStep

    agent = IntegrationTestAgent(IntegrationTestConfig())
    run = _fake_run_ssh(
        validation=[MCPToolCall(name="bash", arguments={"command": "echo 'golden' > answer.txt"})]
    )

    await agent(run)

    assert len(run.recorded) == 1
    step = run.recorded[0]
    assert isinstance(step, ToolStep)
    assert step.result.isError is False
    assert step.result.content[0].text == "staged"


@pytest.mark.asyncio
async def test_ssh_failure_surfaces_as_error_result() -> None:
    from types import SimpleNamespace

    from hud.capabilities import SSHClient

    class FailingSSHClient(SSHClient):
        def __init__(self) -> None:
            pass

        async def run(self, *args: object, **kwargs: Any) -> Any:
            return SimpleNamespace(stdout=b"", stderr=b"no such file", returncode=127)

    run = _fake_run(validation=[MCPToolCall(name="bash", arguments={"command": "nope"})], tools={})
    run.client.manifest = SimpleNamespace(
        bindings=[SimpleNamespace(name="workspace", protocol="ssh/2")]
    )
    run.client.open = AsyncMock(return_value=FailingSSHClient())

    await IntegrationTestAgent(IntegrationTestConfig())(run)

    assert run.recorded[0].result.isError is True
