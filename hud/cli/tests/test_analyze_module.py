from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hud.cli.analyze import (
    analyze_environment,
    analyze_environment_from_config,
    analyze_environment_from_mcp_config,
    display_interactive,
    display_markdown,
    parse_docker_command,
)

if TYPE_CHECKING:
    from pathlib import Path


# Mark entire module as asyncio to ensure async tests run with pytest-asyncio
pytestmark = pytest.mark.asyncio


def test_parse_docker_command():
    cmd = ["docker", "run", "--rm", "-i", "img"]
    cfg = parse_docker_command(cmd)
    assert cfg == {"local": {"command": "docker", "args": ["run", "--rm", "-i", "img"]}}


@pytest.mark.asyncio
@patch("hud.cli.analyze.MCPClient")
@patch("hud.cli.analyze.console")
async def test_analyze_environment_success_json(mock_console, MockClient):
    client = AsyncMock()
    client.initialize.return_value = None
    client.analyze_environment.return_value = {"tools": [], "resources": []}
    client.shutdown.return_value = None
    MockClient.return_value = client

    await analyze_environment(["docker", "run", "img"], output_format="json", verbose=False)
    assert client.initialize.awaited
    assert client.analyze_environment.awaited
    assert client.shutdown.awaited
    assert mock_console.print_json.called


@pytest.mark.asyncio
@patch("hud.cli.analyze.MCPClient")
@patch("hud.cli.analyze.console")
async def test_analyze_environment_failure(mock_console, MockClient):
    client = AsyncMock()
    client.initialize.side_effect = RuntimeError("boom")
    client.shutdown.return_value = None
    MockClient.return_value = client

    # Should swallow exception and return without raising
    await analyze_environment(["docker", "run", "img"], output_format="json", verbose=True)
    assert client.shutdown.awaited
    assert mock_console.print_json.called is False


def test_display_interactive_metadata_only(monkeypatch):
    import hud.cli.analyze as mod

    monkeypatch.setattr(mod, "console", MagicMock(), raising=False)
    monkeypatch.setattr(mod, "hud_console", MagicMock(), raising=False)

    analysis = {
        "image": "img:latest",
        "status": "cached",
        "tool_count": 2,
        "tools": [
            {"name": "t1", "description": "d1", "inputSchema": {"type": "object"}},
            {"name": "t2", "description": "d2"},
        ],
        "resources": [],
    }
    display_interactive(analysis)


def test_display_markdown_both_paths(capsys):
    # metadata-only
    md_only = {"image": "img:latest", "tool_count": 0, "tools": [], "resources": []}
    display_markdown(md_only)

    # live metadata
    live = {"metadata": {"servers": ["s1"], "initialized": True}, "tools": [], "resources": []}
    display_markdown(live)

    # Check that output was generated
    captured = capsys.readouterr()
    assert "MCP Environment Analysis" in captured.out


@patch("hud.cli.analyze.MCPClient")
async def test_analyze_environment_from_config(MockClient, tmp_path: Path):
    client = AsyncMock()
    client.initialize.return_value = None
    client.analyze_environment.return_value = {"tools": [], "resources": []}
    client.shutdown.return_value = None
    MockClient.return_value = client

    cfg = tmp_path / "mcp.json"
    cfg.write_text('{"local": {"command": "docker", "args": ["run", "img"]}}')
    await analyze_environment_from_config(cfg, output_format="json", verbose=False)
    assert client.initialize.awaited and client.shutdown.awaited


@patch("hud.cli.analyze.MCPClient")
async def test_analyze_environment_from_mcp_config(MockClient):
    client = AsyncMock()
    client.initialize.return_value = None
    client.analyze_environment.return_value = {"tools": [], "resources": []}
    client.shutdown.return_value = None
    MockClient.return_value = client

    mcp_config = {"local": {"command": "docker", "args": ["run", "img"]}}
    await analyze_environment_from_mcp_config(mcp_config, output_format="json", verbose=False)
    assert client.initialize.awaited and client.shutdown.awaited
