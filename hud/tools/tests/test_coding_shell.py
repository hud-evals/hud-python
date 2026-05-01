"""Tests for shell compatibility tool."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.types import TextContent

from hud.tools._legacy import ShellTool
from hud.tools.coding import BashTool, ShellCallOutcome, ShellCommandOutput


class TestShellTool:
    """Tests for ShellTool compatibility wrapper."""

    def test_shell_tool_is_bash_tool(self):
        tool = ShellTool()
        assert isinstance(tool, BashTool)
        assert tool.name == "bash"
        assert "native_tools" not in tool.meta

    @pytest.mark.asyncio
    async def test_call_with_commands_uses_bash_behavior(self):
        tool = ShellTool()

        mock_session = MagicMock()
        mock_session._started = False
        mock_session.run = AsyncMock(
            return_value=ShellCommandOutput(
                stdout="test output",
                stderr="",
                outcome=ShellCallOutcome(type="exit", exit_code=0),
            )
        )
        mock_session.start = AsyncMock()

        with patch("hud.tools.coding.bash.ClaudeBashSession", return_value=mock_session):
            result = await tool(command="echo test")

        assert isinstance(result[0], TextContent)
        assert result[0].text == "test output"
        mock_session.run.assert_called_once_with("echo test", timeout_ms=120000)
