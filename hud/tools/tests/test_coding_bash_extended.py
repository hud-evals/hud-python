"""Extended tests for bash tool to improve coverage."""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hud.tools.coding import _BashSession


class TestBashSessionExtended:
    """Extended tests for _BashSession to improve coverage."""

    @pytest.mark.asyncio
    async def test_session_start_already_started(self):
        """Test starting a session that's already started."""
        session = _BashSession()
        session._started = True

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await session.start()

            # Should call sleep and return early
            mock_sleep.assert_called_once_with(0)

    @pytest.mark.asyncio
    @pytest.mark.skipif(sys.platform == "win32", reason="Unix-specific test")
    async def test_session_start_unix_preexec(self):
        """Test session start on Unix systems uses preexec_fn."""
        session = _BashSession()

        with patch("asyncio.create_subprocess_shell") as mock_create:
            mock_process = MagicMock()
            mock_create.return_value = mock_process

            await session.start()

            # Check that preexec_fn was passed
            call_kwargs = mock_create.call_args[1]
            assert "preexec_fn" in call_kwargs
            assert call_kwargs["preexec_fn"] is not None

    def test_session_stop_with_terminated_process(self):
        """Test stopping a session with already terminated process."""
        session = _BashSession()
        session._started = True

        # Mock process that's already terminated
        mock_process = MagicMock()
        mock_process.returncode = 0  # Process already exited
        session._process = mock_process

        # Should not raise error and not call terminate
        session.stop()
        mock_process.terminate.assert_not_called()

    def test_session_stop_with_running_process(self):
        """Test stopping a session with running process."""
        session = _BashSession()
        session._started = True

        # Mock process that's still running
        mock_process = MagicMock()
        mock_process.returncode = None
        session._process = mock_process

        session.stop()
        mock_process.terminate.assert_called_once()

    @pytest.mark.asyncio
    async def test_session_run_with_exited_process(self):
        """Test running command when process has already exited."""
        session = _BashSession()
        session._started = True

        # Mock process that has exited
        mock_process = MagicMock()
        mock_process.returncode = 1
        session._process = mock_process

        result = await session.run("echo test")

        assert result.stdout == ""
        assert result.stderr == "bash has exited with returncode 1"
        assert result.outcome.type == "exit"
        assert result.outcome.exit_code == 1

    @pytest.mark.asyncio
    async def test_session_run_with_stderr_output(self):
        """Test command execution with stderr output."""
        session = _BashSession()
        session._started = True

        # Mock process
        mock_process = MagicMock()
        mock_process.returncode = None
        mock_process.stdin = MagicMock()
        mock_process.stdin.write = MagicMock()
        mock_process.stdin.drain = AsyncMock()
        stdout_buffer = MagicMock()
        stdout_buffer.decode.return_value = "stdout output\n<<exit>>0\n"
        stdout_buffer.clear = MagicMock()
        stderr_buffer = MagicMock()
        stderr_buffer.decode.return_value = "stderr output\n"
        stderr_buffer.clear = MagicMock()
        mock_process.stdout = MagicMock()
        mock_process.stdout._buffer = stdout_buffer
        mock_process.stderr = MagicMock()
        mock_process.stderr._buffer = stderr_buffer

        session._process = mock_process

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await session.run("command")

        assert result.stdout == "stdout output"
        assert result.stderr == "stderr output"

    @pytest.mark.asyncio
    async def test_session_run_with_asyncio_timeout(self):
        """Test command execution timing out."""
        session = _BashSession()
        session._started = True

        # Mock process
        mock_process = MagicMock()
        mock_process.returncode = None
        mock_process.stdin = MagicMock()
        mock_process.stdin.write = MagicMock()
        mock_process.stdin.drain = AsyncMock()
        stdout_buffer = MagicMock()
        stdout_buffer.decode.return_value = "partial output"
        stdout_buffer.clear = MagicMock()
        stderr_buffer = MagicMock()
        stderr_buffer.decode.return_value = "partial error"
        stderr_buffer.clear = MagicMock()
        mock_process.stdout = MagicMock()
        mock_process.stdout._buffer = stdout_buffer
        mock_process.stderr = MagicMock()
        mock_process.stderr._buffer = stderr_buffer

        session._process = mock_process

        result = await session.run("slow command", timeout_ms=1)

        assert result.outcome.type == "timeout"
        assert result.stdout == ""
        assert result.stderr == ""

    @pytest.mark.asyncio
    async def test_session_run_with_custom_timeout(self):
        """Test that a custom timeout value is used and reported in the error."""
        session = _BashSession(timeout=1.0)
        assert session._timeout == 1.0

        session._started = True

        mock_process = MagicMock()
        mock_process.returncode = None
        mock_process.stdin = MagicMock()
        mock_process.stdin.write = MagicMock()
        mock_process.stdin.drain = AsyncMock()
        stdout_buffer = MagicMock()
        stdout_buffer.decode.return_value = ""
        stdout_buffer.clear = MagicMock()
        stderr_buffer = MagicMock()
        stderr_buffer.decode.return_value = ""
        stderr_buffer.clear = MagicMock()
        mock_process.stdout = MagicMock()
        mock_process.stdout._buffer = stdout_buffer
        mock_process.stderr = MagicMock()
        mock_process.stderr._buffer = stderr_buffer

        session._process = mock_process

        result = await session.run("sleep 5")

        assert result.outcome.type == "timeout"

    @pytest.mark.asyncio
    async def test_session_run_with_stdout_exception(self):
        """Test command execution with exception reading stdout."""
        session = _BashSession()
        session._started = True

        # Mock process
        mock_process = MagicMock()
        mock_process.returncode = None
        mock_process.stdin = MagicMock()
        mock_process.stdin.write = MagicMock()
        mock_process.stdin.drain = AsyncMock()
        stdout_buffer = MagicMock()
        stdout_buffer.decode.side_effect = Exception("Read error")
        mock_process.stdout = MagicMock()
        mock_process.stdout._buffer = stdout_buffer
        mock_process.stderr = MagicMock()
        mock_process.stderr._buffer = MagicMock()

        session._process = mock_process

        with pytest.raises(Exception) as exc_info:
            await session.run("bad command")

        assert "Read error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_session_run_with_stderr_exception(self):
        """Test command execution with exception reading stderr."""
        session = _BashSession()
        session._started = True

        # Mock process
        mock_process = MagicMock()
        mock_process.returncode = None
        mock_process.stdin = MagicMock()
        mock_process.stdin.write = MagicMock()
        mock_process.stdin.drain = AsyncMock()
        stdout_buffer = MagicMock()
        stdout_buffer.decode.return_value = "output\n<<exit>>0\n"
        stdout_buffer.clear = MagicMock()
        stderr_buffer = MagicMock()
        stderr_buffer.decode.side_effect = Exception("Stderr read error")
        mock_process.stdout = MagicMock()
        mock_process.stdout._buffer = stdout_buffer
        mock_process.stderr = MagicMock()
        mock_process.stderr._buffer = stderr_buffer

        session._process = mock_process

        with pytest.raises(Exception) as exc_info:
            await session.run("command")

        assert "Stderr read error" in str(exc_info.value)

    def test_bash_session_different_shells(self):
        """Test that different shells are used on different platforms."""
        session = _BashSession()

        expected = "cmd.exe" if sys.platform == "win32" else "/bin/bash"
        assert session.command == expected
