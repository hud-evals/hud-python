"""Tests for bash tool."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hud.tools.coding import BashTool, _BashSession
from hud.tools.types import ContentResult, TextContent, ToolError


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
        """Test stopping a session that hasn't started."""
        session = _BashSession()

        with pytest.raises(ToolError) as exc_info:
            session.stop()

        assert "Session has not started" in str(exc_info.value)

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
        mock_process.stdout = MagicMock()
        mock_process.stdout.readuntil = AsyncMock(return_value=b"Hello World\n<<exit>>\n")
        mock_process.stderr = MagicMock()
        mock_process.stderr.read = AsyncMock(return_value=b"")

        session._process = mock_process

        result = await session.run("echo Hello World")

        assert result.output == "Hello World\n"
        assert result.error == ""


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
        mock_process.stdout = MagicMock()
        mock_process.stdout.readuntil = AsyncMock(return_value=b"hello\n<<exit>>\n")
        mock_process.stderr = MagicMock()
        mock_process.stderr.read = AsyncMock(return_value=b"")

        session._process = mock_process

        heredoc_cmd = "python3 << 'EOF'\nprint('hello')\nEOF"
        await session.run(heredoc_cmd)

        written = mock_process.stdin.write.call_args[0][0].decode()

        # EOF must be followed by newline, then the echo — never "EOF;" or "EOF echo"
        assert "EOF\necho '<<exit>>'\n" in written
        assert "EOF;" not in written
        assert "EOF echo" not in written

    @pytest.mark.asyncio
    async def test_heredoc_integration(self):
        """Integration test: a real heredoc command completes without hanging."""
        from hud.tools.coding.bash import ClaudeBashSession

        session = ClaudeBashSession()
        session._timeout = 5.0  # fail fast if sentinel is broken
        await session.start()
        try:
            result = await session.run("cat << 'EOF'\nhello from heredoc\nEOF")
            assert result.output is not None
            assert "hello from heredoc" in result.output
        finally:
            session.stop()

    @pytest.mark.asyncio
    async def test_heredoc_with_python_integration(self):
        """Integration test: python heredoc executes and returns output."""
        from hud.tools.coding.bash import ClaudeBashSession

        session = ClaudeBashSession()
        session._timeout = 5.0
        await session.start()
        try:
            result = await session.run("python3 << 'PYEOF'\nprint('result:', 2 + 2)\nPYEOF")
            assert result.output is not None
            assert "result: 4" in result.output
        finally:
            session.stop()

    @pytest.mark.asyncio
    async def test_command_after_heredoc_still_works(self):
        """Integration test: session is usable for further commands after a heredoc."""
        from hud.tools.coding.bash import ClaudeBashSession

        session = ClaudeBashSession()
        session._timeout = 5.0
        await session.start()
        try:
            r1 = await session.run("cat << 'EOF'\nfirst\nEOF")
            assert r1.output is not None
            assert "first" in r1.output

            r2 = await session.run("echo second")
            assert r2.output is not None
            assert "second" in r2.output
        finally:
            session.stop()


class TestBashTool:
    """Tests for BashTool."""

    def test_bash_tool_init(self):
        """Test BashTool initialization."""
        tool = BashTool()
        assert tool.session is None

    @pytest.mark.asyncio
    async def test_call_with_command(self):
        """Test calling tool with a command."""
        tool = BashTool()

        # Mock session - must set _started=False so start() gets called
        mock_session = MagicMock()
        mock_session._started = False
        mock_session.run = AsyncMock(return_value=ContentResult(output="test output"))
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
            mock_session.run.assert_called_once_with("echo test")

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

        # Track calls across instances of our fake session class
        stop_called = []
        start_called = []

        class FakeSession:
            """Fake session that tracks stop/start calls."""

            async def start(self) -> None:
                start_called.append(True)

            def stop(self) -> None:
                stop_called.append(True)

        # Set up existing session
        old_session = FakeSession()
        tool.session = old_session  # type: ignore[assignment]

        result = await tool(restart=True)

        # Verify old session was stopped
        assert len(stop_called) == 1, "stop() should be called on old session"

        # Verify new session was started
        assert len(start_called) == 1, "start() should be called on new session"

        # Verify result
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert result[0].text == "Bash session restarted."

        # Verify new session is a FakeSession (type preserved)
        assert isinstance(tool.session, FakeSession)
        assert tool.session is not old_session

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
        existing_session.run = AsyncMock(return_value=ContentResult(output="result"))
        tool.session = existing_session

        result = await tool(command="ls")

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert result[0].text == "result"
        existing_session.run.assert_called_once_with("ls")
