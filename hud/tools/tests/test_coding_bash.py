"""Tests for bash tool."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hud.tools.coding import BashTool, ShellCallOutcome, ShellCommandOutput, _BashSession
from hud.tools.types import TextContent, ToolError


class TestBashSession:
    """Tests for _BashSession."""

    @pytest.mark.asyncio
    async def test_session_start(self):
        """Test starting a bash session."""
        session = _BashSession()
        assert session._started is False

        with patch("asyncio.create_subprocess_shell") as mock_create:
            mock_process = MagicMock()
            mock_create.return_value = mock_process

            await session.start()

            assert session._started is True
            assert session._process == mock_process
            mock_create.assert_called_once()

    def test_session_stop_not_started(self):
        """Stopping a session that has not started is a no-op."""
        session = _BashSession()

        session.stop()

    @pytest.mark.asyncio
    async def test_session_run_not_started(self):
        """Test running command on a session that hasn't started."""
        session = _BashSession()

        with pytest.raises(ToolError) as exc_info:
            await session.run("echo test")

        assert "Session has not started" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_session_run_success(self):
        """Test successful command execution."""
        session = _BashSession()
        session._started = True

        # Mock process
        mock_process = MagicMock()
        mock_process.returncode = None
        mock_process.stdin = MagicMock()
        mock_process.stdin.write = MagicMock()
        mock_process.stdin.drain = AsyncMock()
        stdout_buffer = MagicMock()
        stdout_buffer.decode.return_value = "Hello World\n<<exit>>0\n"
        stdout_buffer.clear = MagicMock()
        stderr_buffer = MagicMock()
        stderr_buffer.decode.return_value = ""
        stderr_buffer.clear = MagicMock()
        mock_process.stdout = MagicMock()
        mock_process.stdout._buffer = stdout_buffer
        mock_process.stderr = MagicMock()
        mock_process.stderr._buffer = stderr_buffer

        session._process = mock_process

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await session.run("echo Hello World")

        assert result.stdout == "Hello World"
        assert result.stderr == ""
        assert result.outcome.type == "exit"
        assert result.outcome.exit_code == 0


class TestBashSessionHeredoc:
    """Tests for heredoc handling in ClaudeBashSession."""

    @pytest.mark.asyncio
    async def test_sentinel_on_own_line_after_heredoc(self):
        """Sentinel echo must be on its own line so heredoc terminators aren't corrupted."""
        session = _BashSession()
        session._started = True

        mock_process = MagicMock()
        mock_process.returncode = None
        mock_process.stdin = MagicMock()
        mock_process.stdin.write = MagicMock()
        mock_process.stdin.drain = AsyncMock()
        stdout_buffer = MagicMock()
        stdout_buffer.decode.return_value = "hello\n<<exit>>\n"
        stdout_buffer.clear = MagicMock()
        stderr_buffer = MagicMock()
        stderr_buffer.decode.return_value = ""
        stderr_buffer.clear = MagicMock()
        mock_process.stdout = MagicMock()
        mock_process.stdout._buffer = stdout_buffer
        mock_process.stderr = MagicMock()
        mock_process.stderr._buffer = stderr_buffer

        session._process = mock_process

        heredoc_cmd = "python3 << 'EOF'\nprint('hello')\nEOF"
        with patch("asyncio.sleep", new_callable=AsyncMock):
            await session.run(heredoc_cmd, capture_exit_code=False)

        written = mock_process.stdin.write.call_args[0][0].decode()

        # EOF must be followed by newline, then the echo — never "EOF;" or "EOF echo"
        assert "EOF\necho '<<exit>>'\n" in written
        assert "EOF;" not in written
        assert "EOF echo" not in written

    @pytest.mark.asyncio
    async def test_heredoc_integration(self):
        """Integration test: a real heredoc command completes without hanging."""
        from hud.tools.coding import ClaudeBashSession

        session = ClaudeBashSession()
        session._timeout = 5.0  # fail fast if sentinel is broken
        await session.start()
        try:
            result = await session.run("cat << 'EOF'\nhello from heredoc\nEOF")
            assert "hello from heredoc" in result.stdout
        finally:
            session.stop()

    @pytest.mark.asyncio
    async def test_heredoc_with_python_integration(self):
        """Integration test: python heredoc executes and returns output."""
        from hud.tools.coding import ClaudeBashSession

        session = ClaudeBashSession()
        session._timeout = 5.0
        await session.start()
        try:
            result = await session.run("python3 << 'PYEOF'\nprint('result:', 2 + 2)\nPYEOF")
            assert "result: 4" in result.stdout
        finally:
            session.stop()

    @pytest.mark.asyncio
    async def test_command_after_heredoc_still_works(self):
        """Integration test: session is usable for further commands after a heredoc."""
        from hud.tools.coding import ClaudeBashSession

        session = ClaudeBashSession()
        session._timeout = 5.0
        await session.start()
        try:
            r1 = await session.run("cat << 'EOF'\nfirst\nEOF")
            assert "first" in r1.stdout

            r2 = await session.run("echo second")
            assert "second" in r2.stdout
        finally:
            session.stop()


class TestBashTool:
    """Tests for BashTool."""

    def test_bash_tool_init(self):
        """Test BashTool initialization."""
        tool = BashTool()
        assert tool.session is None

    @pytest.mark.asyncio
    async def test_bash_tool_contract_matches_anthropic_docs(self):
        """BashTool accepts command or restart, with restart not requiring command."""
        tool = BashTool()

        with pytest.raises(ToolError, match="No command provided"):
            await tool()

        new_session = MagicMock()
        new_session.start = AsyncMock()
        with patch("hud.tools.coding.bash.ClaudeBashSession", return_value=new_session):
            result = await tool(restart=True)

        assert isinstance(result[0], TextContent)
        assert result[0].text == "Bash session restarted."
        new_session.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_with_command(self):
        """Test calling tool with a command."""
        tool = BashTool()

        # Mock session - must set _started=False so start() gets called
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

        # Mock _BashSession creation
        with patch("hud.tools.coding.bash.ClaudeBashSession") as mock_session_class:
            mock_session_class.return_value = mock_session

            result = await tool(command="echo test")

            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], TextContent)
            assert result[0].text == "test output"
            mock_session.start.assert_called_once()
            mock_session.run.assert_called_once_with("echo test", timeout_ms=120000)

    @pytest.mark.asyncio
    async def test_call_restart(self):
        """Test restarting the tool."""
        tool = BashTool()

        # Mock new session - start must be AsyncMock for await
        new_session = MagicMock()
        new_session.start = AsyncMock()

        # When session is None, restart uses _BashSession class directly
        with patch("hud.tools.coding.bash.ClaudeBashSession", return_value=new_session):
            result = await tool(restart=True)

            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], TextContent)
            assert result[0].text == "Bash session restarted."
            new_session.start.assert_called_once()
            assert tool.session == new_session

    @pytest.mark.asyncio
    async def test_call_restart_with_existing_session(self):
        """Test restarting the tool when there's an existing session calls stop()."""
        tool = BashTool()

        # Set up existing session with a mock
        old_session = MagicMock()
        old_session.stop = MagicMock()
        tool.session = old_session  # type: ignore[assignment]

        # Mock the new session that will be created
        new_session = MagicMock()
        new_session.start = AsyncMock()

        with patch("hud.tools.coding.bash.ClaudeBashSession", return_value=new_session):
            result = await tool(restart=True)

        # Verify old session was stopped
        old_session.stop.assert_called_once()

        # Verify new session was started
        new_session.start.assert_called_once()

        # Verify result
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert result[0].text == "Bash session restarted."

        # Verify new session replaced the old one
        assert tool.session is not old_session
        assert tool.session is new_session

    @pytest.mark.asyncio
    async def test_call_no_command_error(self):
        """Test calling without command raises error."""
        tool = BashTool()

        with pytest.raises(ToolError) as exc_info:
            await tool()

        assert str(exc_info.value) == "No command provided."

    @pytest.mark.asyncio
    async def test_call_with_existing_session(self):
        """Test calling with an existing session."""
        tool = BashTool()

        # Set up existing session
        existing_session = MagicMock()
        existing_session._started = True
        existing_session.run = AsyncMock(
            return_value=ShellCommandOutput(
                stdout="result",
                stderr="",
                outcome=ShellCallOutcome(type="exit", exit_code=0),
            )
        )
        tool.session = existing_session

        result = await tool(command="ls")

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert result[0].text == "result"
        existing_session.run.assert_called_once_with("ls", timeout_ms=120000)
