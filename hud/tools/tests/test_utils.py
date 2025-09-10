"""Tests for tools utils."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from hud.tools.utils import maybe_truncate, run


class TestRun:
    """Tests for the run function."""

    @pytest.mark.asyncio
    async def test_run_string_command_success(self):
        """Test running a string command successfully."""
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"output", b""))

        with patch("asyncio.create_subprocess_shell", return_value=mock_proc) as mock_shell:
            return_code, stdout, stderr = await run("echo test")

            assert return_code == 0
            assert stdout == "output"
            assert stderr == ""
            mock_shell.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_list_command_success(self):
        """Test running a list command successfully."""
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"hello world", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            return_code, stdout, stderr = await run(["echo", "hello", "world"])

            assert return_code == 0
            assert stdout == "hello world"
            assert stderr == ""
            mock_exec.assert_called_once_with(
                "echo",
                "hello",
                "world",
                stdin=None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

    @pytest.mark.asyncio
    async def test_run_with_input(self):
        """Test running a command with input."""
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"processed", b""))

        with patch("asyncio.create_subprocess_shell", return_value=mock_proc):
            return_code, stdout, _stderr = await run("cat", input="test input")

            assert return_code == 0
            assert stdout == "processed"
            mock_proc.communicate.assert_called_once_with(input=b"test input")

    @pytest.mark.asyncio
    async def test_run_with_error(self):
        """Test running a command that returns an error."""
        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"error message"))

        with patch("asyncio.create_subprocess_shell", return_value=mock_proc):
            return_code, stdout, stderr = await run("false")

            assert return_code == 1
            assert stdout == ""
            assert stderr == "error message"

    @pytest.mark.asyncio
    async def test_run_with_timeout(self):
        """Test running a command with custom timeout."""
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"done", b""))

        with (
            patch("asyncio.create_subprocess_shell", return_value=mock_proc),
            patch("asyncio.wait_for") as mock_wait_for,
        ):
            mock_wait_for.return_value = (b"done", b"")

            _return_code, _stdout, _stderr = await run("sleep 1", timeout=5.0)

            # Check that wait_for was called with the correct timeout
            mock_wait_for.assert_called_once()
            assert mock_wait_for.call_args[1]["timeout"] == 5.0

    @pytest.mark.asyncio
    async def test_run_timeout_exception(self):
        """Test running a command that times out."""
        mock_proc = AsyncMock()

        with (
            patch("asyncio.create_subprocess_shell", return_value=mock_proc),
            patch("asyncio.wait_for", side_effect=TimeoutError()),
            pytest.raises(asyncio.TimeoutError),
        ):
            await run("sleep infinity", timeout=0.1)


class TestMaybeTruncate:
    """Tests for the maybe_truncate function."""

    def test_maybe_truncate_short_text(self):
        """Test that short text is not truncated."""
        text = "This is a short text"
        result = maybe_truncate(text)
        assert result == text

    def test_maybe_truncate_long_text_default(self):
        """Test that long text is truncated with default limit."""
        text = "x" * 30000  # Much longer than default limit
        result = maybe_truncate(text)

        assert len(result) < len(text)
        assert result.endswith("... (truncated)")
        assert len(result) == 20480 + len("... (truncated)")

    def test_maybe_truncate_custom_limit(self):
        """Test truncation with custom limit."""
        text = "abcdefghijklmnopqrstuvwxyz"
        result = maybe_truncate(text, max_length=10)

        assert result == "abcdefghij... (truncated)"

    def test_maybe_truncate_exact_limit(self):
        """Test text exactly at limit is not truncated."""
        text = "x" * 100
        result = maybe_truncate(text, max_length=100)

        assert result == text

    def test_maybe_truncate_empty_string(self):
        """Test empty string handling."""
        result = maybe_truncate("")
        assert result == ""

    def test_maybe_truncate_unicode(self):
        """Test truncation with unicode characters."""
        text = "🎉" * 5000
        result = maybe_truncate(text, max_length=10)

        assert len(result) > 10  # Because of "... (truncated)" suffix
        assert result.endswith("... (truncated)")
