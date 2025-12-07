"""Tests for shell tool."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hud.tools.shell import (
    ShellCallOutcome,
    ShellCommandOutput,
    ShellResult,
    ShellTool,
    _BashSession,
)
from hud.tools.types import ToolError


class TestShellCallOutcome:
    """Tests for ShellCallOutcome dataclass."""

    def test_to_dict_exit(self):
        """Test to_dict for exit outcome."""
        outcome = ShellCallOutcome(type="exit", exit_code=0)
        assert outcome.to_dict() == {"type": "exit", "exit_code": 0}

    def test_to_dict_exit_with_error_code(self):
        """Test to_dict for exit outcome with non-zero exit code."""
        outcome = ShellCallOutcome(type="exit", exit_code=1)
        assert outcome.to_dict() == {"type": "exit", "exit_code": 1}

    def test_to_dict_timeout(self):
        """Test to_dict for timeout outcome."""
        outcome = ShellCallOutcome(type="timeout")
        assert outcome.to_dict() == {"type": "timeout"}


class TestShellCommandOutput:
    """Tests for ShellCommandOutput dataclass."""

    def test_to_dict(self):
        """Test to_dict method."""
        output = ShellCommandOutput(
            stdout="hello",
            stderr="",
            outcome=ShellCallOutcome(type="exit", exit_code=0),
        )
        result = output.to_dict()
        assert result["stdout"] == "hello"
        assert result["stderr"] == ""
        assert result["outcome"] == {"type": "exit", "exit_code": 0}


class TestShellResult:
    """Tests for ShellResult dataclass."""

    def test_to_dict_without_max_output_length(self):
        """Test to_dict without max_output_length."""
        result = ShellResult(
            output=[
                ShellCommandOutput(
                    stdout="test",
                    stderr="",
                    outcome=ShellCallOutcome(type="exit", exit_code=0),
                )
            ]
        )
        d = result.to_dict()
        assert "output" in d
        assert len(d["output"]) == 1
        assert "max_output_length" not in d

    def test_to_dict_with_max_output_length(self):
        """Test to_dict with max_output_length."""
        result = ShellResult(
            output=[
                ShellCommandOutput(
                    stdout="test",
                    stderr="",
                    outcome=ShellCallOutcome(type="exit", exit_code=0),
                )
            ],
            max_output_length=1024,
        )
        d = result.to_dict()
        assert d["max_output_length"] == 1024


class TestBashSession:
    """Tests for _BashSession."""

    def test_init(self):
        """Test session initialization."""
        session = _BashSession()
        assert session._started is False
        assert session._timed_out is False

    @pytest.mark.asyncio
    async def test_start(self):
        """Test starting a bash session."""
        session = _BashSession()

        with patch("asyncio.create_subprocess_shell") as mock_create:
            mock_process = MagicMock()
            mock_create.return_value = mock_process

            await session.start()

            assert session._started is True
            assert session._process == mock_process
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_already_started(self):
        """Test starting a session that's already started."""
        session = _BashSession()
        session._started = True

        with patch("asyncio.create_subprocess_shell") as mock_create:
            await session.start()
            mock_create.assert_not_called()

    def test_stop_not_started(self):
        """Test stopping a session that hasn't started."""
        session = _BashSession()
        # Should not raise
        session.stop()

    def test_stop_already_exited(self):
        """Test stopping a session that already exited."""
        session = _BashSession()
        session._started = True
        mock_process = MagicMock()
        mock_process.returncode = 0  # Already exited
        session._process = mock_process

        session.stop()
        mock_process.terminate.assert_not_called()

    def test_stop_running(self):
        """Test stopping a running session."""
        session = _BashSession()
        session._started = True
        mock_process = MagicMock()
        mock_process.returncode = None  # Still running
        session._process = mock_process

        session.stop()
        mock_process.terminate.assert_called_once()

    def test_is_alive_not_started(self):
        """Test is_alive when not started."""
        session = _BashSession()
        assert session.is_alive() is False

    def test_is_alive_running(self):
        """Test is_alive when running."""
        session = _BashSession()
        session._started = True
        session._timed_out = False
        mock_process = MagicMock()
        mock_process.returncode = None
        session._process = mock_process

        assert session.is_alive() is True

    def test_is_alive_timed_out(self):
        """Test is_alive when timed out."""
        session = _BashSession()
        session._started = True
        session._timed_out = True
        mock_process = MagicMock()
        mock_process.returncode = None
        session._process = mock_process

        assert session.is_alive() is False

    def test_is_alive_process_exited(self):
        """Test is_alive when process exited."""
        session = _BashSession()
        session._started = True
        session._timed_out = False
        mock_process = MagicMock()
        mock_process.returncode = 0
        session._process = mock_process

        assert session.is_alive() is False

    @pytest.mark.asyncio
    async def test_run_not_started(self):
        """Test running command on a session that hasn't started."""
        session = _BashSession()

        with pytest.raises(ToolError) as exc_info:
            await session.run("echo test")

        assert "Session has not started" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_run_success(self):
        """Test successful command execution."""
        session = _BashSession()
        session._started = True

        # Mock process
        mock_process = MagicMock()
        mock_process.returncode = None
        mock_process.stdin = MagicMock()
        mock_process.stdin.write = MagicMock()
        mock_process.stdin.drain = AsyncMock()

        # Create mock buffers
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

        # Patch asyncio.sleep to avoid actual delay
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await session.run("echo Hello World")

        assert result.stdout == "Hello World"
        assert result.stderr == ""
        assert result.outcome.type == "exit"
        assert result.outcome.exit_code == 0

    @pytest.mark.asyncio
    async def test_run_with_exit_code(self):
        """Test command execution with non-zero exit code."""
        session = _BashSession()
        session._started = True

        mock_process = MagicMock()
        mock_process.returncode = None
        mock_process.stdin = MagicMock()
        mock_process.stdin.write = MagicMock()
        mock_process.stdin.drain = AsyncMock()

        stdout_buffer = MagicMock()
        stdout_buffer.decode.return_value = "<<exit>>127\n"
        stdout_buffer.clear = MagicMock()

        stderr_buffer = MagicMock()
        stderr_buffer.decode.return_value = "command not found"
        stderr_buffer.clear = MagicMock()

        mock_process.stdout = MagicMock()
        mock_process.stdout._buffer = stdout_buffer
        mock_process.stderr = MagicMock()
        mock_process.stderr._buffer = stderr_buffer

        session._process = mock_process

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await session.run("nonexistent_command")

        assert result.outcome.type == "exit"
        assert result.outcome.exit_code == 127


class TestShellTool:
    """Tests for ShellTool."""

    def test_init(self):
        """Test ShellTool initialization."""
        tool = ShellTool()
        assert tool._session is None

    @pytest.mark.asyncio
    async def test_call_no_commands(self):
        """Test calling without commands raises error."""
        tool = ShellTool()

        with pytest.raises(ToolError) as exc_info:
            await tool()

        assert "No commands provided" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_call_empty_commands(self):
        """Test calling with empty commands list raises error."""
        tool = ShellTool()

        with pytest.raises(ToolError) as exc_info:
            await tool(commands=[])

        assert "No commands provided" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_call_with_command(self):
        """Test calling tool with a command."""
        tool = ShellTool()

        # Mock session
        mock_session = MagicMock()
        mock_session.is_alive.return_value = True
        mock_session.run = AsyncMock(
            return_value=ShellCommandOutput(
                stdout="test output",
                stderr="",
                outcome=ShellCallOutcome(type="exit", exit_code=0),
            )
        )
        mock_session.start = AsyncMock()

        with patch("hud.tools.shell._BashSession") as mock_session_class:
            mock_session_class.return_value = mock_session

            result = await tool(commands=["echo test"])

            assert isinstance(result, ShellResult)
            assert len(result.output) == 1
            assert result.output[0].stdout == "test output"
            mock_session.start.assert_called_once()
            mock_session.run.assert_called_once_with("echo test", None)

    @pytest.mark.asyncio
    async def test_call_with_timeout(self):
        """Test calling tool with timeout_ms."""
        tool = ShellTool()

        mock_session = MagicMock()
        mock_session.is_alive.return_value = True
        mock_session.run = AsyncMock(
            return_value=ShellCommandOutput(
                stdout="output",
                stderr="",
                outcome=ShellCallOutcome(type="exit", exit_code=0),
            )
        )
        mock_session.start = AsyncMock()

        with patch("hud.tools.shell._BashSession") as mock_session_class:
            mock_session_class.return_value = mock_session

            result = await tool(commands=["sleep 1"], timeout_ms=5000)

            mock_session.run.assert_called_once_with("sleep 1", 5000)
            assert result.max_output_length is None

    @pytest.mark.asyncio
    async def test_call_with_max_output_length(self):
        """Test calling tool with max_output_length."""
        tool = ShellTool()

        mock_session = MagicMock()
        mock_session.is_alive.return_value = True
        mock_session.run = AsyncMock(
            return_value=ShellCommandOutput(
                stdout="output",
                stderr="",
                outcome=ShellCallOutcome(type="exit", exit_code=0),
            )
        )
        mock_session.start = AsyncMock()

        with patch("hud.tools.shell._BashSession") as mock_session_class:
            mock_session_class.return_value = mock_session

            result = await tool(commands=["echo test"], max_output_length=2048)

            assert result.max_output_length == 2048

    @pytest.mark.asyncio
    async def test_call_multiple_commands(self):
        """Test calling tool with multiple commands."""
        tool = ShellTool()

        mock_session = MagicMock()
        mock_session.is_alive.return_value = True
        mock_session.run = AsyncMock(
            side_effect=[
                ShellCommandOutput(
                    stdout="first",
                    stderr="",
                    outcome=ShellCallOutcome(type="exit", exit_code=0),
                ),
                ShellCommandOutput(
                    stdout="second",
                    stderr="",
                    outcome=ShellCallOutcome(type="exit", exit_code=0),
                ),
            ]
        )
        mock_session.start = AsyncMock()

        with patch("hud.tools.shell._BashSession") as mock_session_class:
            mock_session_class.return_value = mock_session

            result = await tool(commands=["echo first", "echo second"])

            assert len(result.output) == 2
            assert result.output[0].stdout == "first"
            assert result.output[1].stdout == "second"

    @pytest.mark.asyncio
    async def test_call_reuses_session(self):
        """Test that existing session is reused."""
        tool = ShellTool()

        mock_session = MagicMock()
        mock_session.is_alive.return_value = True
        mock_session.run = AsyncMock(
            return_value=ShellCommandOutput(
                stdout="output",
                stderr="",
                outcome=ShellCallOutcome(type="exit", exit_code=0),
            )
        )
        mock_session.start = AsyncMock()

        with patch("hud.tools.shell._BashSession") as mock_session_class:
            mock_session_class.return_value = mock_session

            # First call
            await tool(commands=["echo first"])
            # Second call
            await tool(commands=["echo second"])

            # Session should only be created once
            assert mock_session_class.call_count == 1

    @pytest.mark.asyncio
    async def test_auto_restart_on_timeout(self):
        """Test auto-restart after timeout."""
        tool = ShellTool()

        # Create a timed-out session
        old_session = MagicMock()
        old_session._timed_out = True
        old_session._process = MagicMock()
        old_session._process.returncode = None
        old_session.is_alive.return_value = False
        old_session.stop = MagicMock()

        tool._session = old_session

        # New session
        new_session = MagicMock()
        new_session.is_alive.return_value = True
        new_session.run = AsyncMock(
            return_value=ShellCommandOutput(
                stdout="output",
                stderr="",
                outcome=ShellCallOutcome(type="exit", exit_code=0),
            )
        )
        new_session.start = AsyncMock()

        with patch("hud.tools.shell._BashSession") as mock_session_class:
            mock_session_class.return_value = new_session

            result = await tool(commands=["echo test"])

            # Old session should be stopped
            old_session.stop.assert_called_once()
            # New session should be created and started
            new_session.start.assert_called_once()
            # Result should include restart message
            assert "timed out" in result.output[0].stderr
            assert "auto-restarted" in result.output[0].stderr

    @pytest.mark.asyncio
    async def test_auto_restart_on_exit(self):
        """Test auto-restart after session exit."""
        tool = ShellTool()

        # Create an exited session
        old_session = MagicMock()
        old_session._timed_out = False
        old_session._process = MagicMock()
        old_session._process.returncode = 1
        old_session.is_alive.return_value = False
        old_session.stop = MagicMock()

        tool._session = old_session

        # New session
        new_session = MagicMock()
        new_session.is_alive.return_value = True
        new_session.run = AsyncMock(
            return_value=ShellCommandOutput(
                stdout="output",
                stderr="",
                outcome=ShellCallOutcome(type="exit", exit_code=0),
            )
        )
        new_session.start = AsyncMock()

        with patch("hud.tools.shell._BashSession") as mock_session_class:
            mock_session_class.return_value = new_session

            result = await tool(commands=["echo test"])

            # Result should include restart message with exit code
            assert "exited with code 1" in result.output[0].stderr

    @pytest.mark.asyncio
    async def test_command_execution_error(self):
        """Test handling of command execution error."""
        tool = ShellTool()

        mock_session = MagicMock()
        mock_session.is_alive.return_value = True
        mock_session.run = AsyncMock(side_effect=Exception("Test error"))
        mock_session.start = AsyncMock()

        with patch("hud.tools.shell._BashSession") as mock_session_class:
            mock_session_class.return_value = mock_session

            result = await tool(commands=["failing command"])

            assert len(result.output) == 1
            assert "Test error" in result.output[0].stderr
            assert result.output[0].outcome.exit_code == 1

    @pytest.mark.asyncio
    async def test_restart_message_added_to_existing_stderr(self):
        """Test that restart message is prepended to existing stderr."""
        tool = ShellTool()

        # Create a timed-out session
        old_session = MagicMock()
        old_session._timed_out = True
        old_session._process = MagicMock()
        old_session._process.returncode = None
        old_session.is_alive.return_value = False
        old_session.stop = MagicMock()

        tool._session = old_session

        # New session
        new_session = MagicMock()
        new_session.is_alive.return_value = True
        new_session.run = AsyncMock(
            return_value=ShellCommandOutput(
                stdout="output",
                stderr="original error",
                outcome=ShellCallOutcome(type="exit", exit_code=1),
            )
        )
        new_session.start = AsyncMock()

        with patch("hud.tools.shell._BashSession") as mock_session_class:
            mock_session_class.return_value = new_session

            result = await tool(commands=["echo test"])

            # Both restart message and original error should be in stderr
            assert "timed out" in result.output[0].stderr
            assert "original error" in result.output[0].stderr

    @pytest.mark.asyncio
    async def test_session_dies_mid_execution(self):
        """Test that session is restarted if it dies mid-execution."""
        tool = ShellTool()

        mock_session = MagicMock()
        # First command succeeds, then session dies, then restarts
        mock_session.is_alive.side_effect = [True, False, True]
        mock_session.run = AsyncMock(
            side_effect=[
                ShellCommandOutput(
                    stdout="first",
                    stderr="",
                    outcome=ShellCallOutcome(type="exit", exit_code=0),
                ),
                ShellCommandOutput(
                    stdout="second",
                    stderr="",
                    outcome=ShellCallOutcome(type="exit", exit_code=0),
                ),
            ]
        )
        mock_session.start = AsyncMock()
        mock_session._timed_out = True
        mock_session._process = MagicMock()
        mock_session._process.returncode = None
        mock_session.stop = MagicMock()

        with patch("hud.tools.shell._BashSession") as mock_session_class:
            mock_session_class.return_value = mock_session

            result = await tool(commands=["echo first", "echo second"])

            assert len(result.output) == 2
