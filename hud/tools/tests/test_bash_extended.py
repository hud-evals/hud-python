"""Extended tests for bash tool to improve coverage."""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hud.tools.bash import ToolError, _BashSession


class TestBashSessionExtended:
    """Extended tests for _BashSession to improve coverage."""

    @pytest.mark.asyncio
    async def test_session_start_already_started(self):
        """Test starting a session that's already started."""
        session = _BashSession()
        session._started = True

        with patch("asyncio.sleep") as mock_sleep:
            mock_sleep.return_value = None
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

        with patch("asyncio.sleep") as mock_sleep:
            mock_sleep.return_value = None
            result = await session.run("echo test")

            assert result.system == "tool must be restarted"
            assert result.error == "bash has exited with returncode 1"
            mock_sleep.assert_called_once_with(0)

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
        mock_process.stdout = MagicMock()
        mock_process.stdout.readuntil = AsyncMock(return_value=b"stdout output\n<<exit>>\n")
        mock_process.stderr = MagicMock()
        mock_process.stderr.read = AsyncMock(return_value=b"stderr output\n")

        session._process = mock_process

        result = await session.run("command")

        assert result.output == "stdout output\n"
        assert result.error == "stderr output"  # .strip() is called on stderr

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
        mock_process.stdout = MagicMock()
        # Simulate timeout
        mock_process.stdout.readuntil = AsyncMock(side_effect=TimeoutError())

        session._process = mock_process

        # Should raise ToolError on timeout
        with pytest.raises(ToolError) as exc_info:
            await session.run("slow command")

        assert "timed out" in str(exc_info.value)
        assert "120.0 seconds" in str(exc_info.value)

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
        mock_process.stdout = MagicMock()
        # Simulate other exception
        mock_process.stdout.readuntil = AsyncMock(side_effect=Exception("Read error"))

        session._process = mock_process

        # The exception should bubble up
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
        mock_process.stdout = MagicMock()
        mock_process.stdout.readuntil = AsyncMock(return_value=b"output\n<<exit>>\n")
        mock_process.stderr = MagicMock()
        # Simulate stderr read error
        mock_process.stderr.read = AsyncMock(side_effect=Exception("Stderr read error"))

        session._process = mock_process

        # stderr exceptions should also bubble up
        with pytest.raises(Exception) as exc_info:
            await session.run("command")

        assert "Stderr read error" in str(exc_info.value)

    def test_bash_session_different_shells(self):
        """Test that different shells are used on different platforms."""
        session = _BashSession()

        # Currently, _BashSession always uses /bin/bash regardless of platform
        # This test should verify the actual implementation
        assert session.command == "/bin/bash"
