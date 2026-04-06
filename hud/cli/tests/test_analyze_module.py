from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hud.cli.analyze import (
    _prepare_mcp_config,
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
@patch("hud.cli.utils.analysis.analyze_environment")
@patch("fastmcp.Client")
@patch("hud.cli.analyze.console")
async def test_analyze_environment_success_json(mock_console, MockClient, mock_mcp_analyze):
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.is_connected = MagicMock(return_value=True)
    client.close = AsyncMock()
    MockClient.return_value = client
    mock_mcp_analyze.return_value = {"tools": [], "resources": []}

    await analyze_environment(["docker", "run", "img"], output_format="json", verbose=False)
    assert client.__aenter__.awaited
    assert mock_mcp_analyze.awaited
    assert client.close.awaited
    assert mock_console.print_json.called


@pytest.mark.asyncio
@patch("fastmcp.Client")
@patch("hud.cli.analyze.console")
async def test_analyze_environment_failure(mock_console, MockClient):
    client = MagicMock()
    client.__aenter__ = AsyncMock(side_effect=RuntimeError("boom"))
    client.is_connected = MagicMock(return_value=True)
    client.close = AsyncMock()
    MockClient.return_value = client

    # Should swallow exception and return without raising
    await analyze_environment(["docker", "run", "img"], output_format="json", verbose=True)
    assert client.close.awaited
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


@patch("hud.cli.utils.analysis.analyze_environment")
@patch("fastmcp.Client")
async def test_analyze_environment_from_config(MockClient, mock_mcp_analyze, tmp_path: Path):
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.is_connected = MagicMock(return_value=True)
    client.close = AsyncMock()
    MockClient.return_value = client
    mock_mcp_analyze.return_value = {"tools": [], "resources": []}

    cfg = tmp_path / "mcp.json"
    cfg.write_text('{"local": {"command": "docker", "args": ["run", "img"]}}')
    await analyze_environment_from_config(cfg, output_format="json", verbose=False)
    assert client.__aenter__.awaited and client.close.awaited


@patch("hud.cli.utils.analysis.analyze_environment")
@patch("fastmcp.Client")
async def test_analyze_environment_from_mcp_config(MockClient, mock_mcp_analyze):
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.is_connected = MagicMock(return_value=True)
    client.close = AsyncMock()
    MockClient.return_value = client
    mock_mcp_analyze.return_value = {"tools": [], "resources": []}

    mcp_config = {"local": {"command": "docker", "args": ["run", "img"]}}
    await analyze_environment_from_mcp_config(mcp_config, output_format="json", verbose=False)
    assert client.__aenter__.awaited and client.close.awaited


@patch("hud.cli.utils.analysis.analyze_environment")
@patch("fastmcp.Client")
async def test_analyze_environment_from_mcp_config_http(MockClient, mock_mcp_analyze):
    """HTTP transport (hud dev) should inject auth=None to skip OAuth discovery."""
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.is_connected = MagicMock(return_value=True)
    client.close = AsyncMock()
    MockClient.return_value = client
    mock_mcp_analyze.return_value = {"tools": [], "resources": []}

    mcp_config = {"hud": {"url": "http://localhost:8000/mcp"}}
    await analyze_environment_from_mcp_config(mcp_config, output_format="json", verbose=False)
    assert client.__aenter__.awaited and client.close.awaited
    # Verify that _prepare_mcp_config injected auth=None
    call_kwargs = MockClient.call_args
    transport_arg = call_kwargs.kwargs.get("transport") or call_kwargs.args[0]
    assert transport_arg["hud"]["auth"] is None


def test_prepare_mcp_config_injects_auth_for_url():
    """URL-based entries get auth=None; stdio entries are left alone."""
    cfg = {
        "hud": {"url": "http://localhost:8000/mcp"},
        "local": {"command": "docker", "args": ["run", "img"]},
    }
    result = _prepare_mcp_config(cfg)
    assert result["hud"]["auth"] is None
    assert result["hud"]["url"] == "http://localhost:8000/mcp"
    assert "auth" not in result["local"]


def test_prepare_mcp_config_preserves_explicit_auth():
    """If auth is already set, don't overwrite it."""
    cfg = {"hud": {"url": "http://localhost:8000/mcp", "auth": "bearer-token"}}
    result = _prepare_mcp_config(cfg)
    assert result["hud"]["auth"] == "bearer-token"
