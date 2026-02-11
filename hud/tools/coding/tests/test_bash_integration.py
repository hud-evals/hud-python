"""Integration tests for bash tool against a real bash process.

These tests verify that command framing (sentinel injection) works
correctly with heredocs, multi-line commands, and edge cases that
cannot be caught by mocking.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile

import pytest

from hud.tools.coding import _BashSession


async def _cleanup(session: _BashSession) -> None:
    """Cleanly shut down a bash session to avoid asyncio transport warnings."""
    proc = session._process
    session.stop()
    # Close stdin so bash sees EOF and exits, then wait briefly for cleanup.
    if proc.stdin:
        proc.stdin.close()
    try:
        await asyncio.wait_for(proc.wait(), timeout=2.0)
    except TimeoutError:
        proc.kill()


@pytest.mark.skipif(sys.platform == "win32", reason="Requires /bin/bash")
class TestBashSessionHeredoc:
    """Integration tests for heredoc commands."""

    @pytest.mark.asyncio
    async def test_heredoc_no_trailing_newline(self):
        """Heredoc without trailing newline should not hang."""
        session = _BashSession()
        session._timeout = 5.0
        await session.start()
        try:
            result = await session.run(
                "cat << 'EOF'\nhello world\nEOF"
            )
            assert result.output is not None
            assert "hello world" in result.output
        finally:
            await _cleanup(session)

    @pytest.mark.asyncio
    async def test_heredoc_with_trailing_newline(self):
        """Heredoc with trailing newline should not hang."""
        session = _BashSession()
        session._timeout = 5.0
        await session.start()
        try:
            result = await session.run(
                "cat << 'EOF'\nhello world\nEOF\n"
            )
            assert result.output is not None
            assert "hello world" in result.output
        finally:
            await _cleanup(session)

    @pytest.mark.asyncio
    async def test_heredoc_write_and_read_file(self):
        """Heredoc that writes a file then cats it back."""
        fd, tmp_path = tempfile.mkstemp(prefix="_bash_test_heredoc_", suffix=".txt")
        os.close(fd)
        session = _BashSession()
        session._timeout = 5.0
        await session.start()
        try:
            result = await session.run(
                f"cat > {tmp_path} << 'EOF'\n"
                "line one\nline two\nEOF\n"
                f"cat {tmp_path}"
            )
            assert result.output is not None
            assert "line one" in result.output
            assert "line two" in result.output
        finally:
            await _cleanup(session)
            os.unlink(tmp_path)
