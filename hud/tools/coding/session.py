"""Shared bash session for shell/bash tools.

This module provides a unified BashSession that can be used by both
BashTool (Claude) and ShellTool (OpenAI) with different output formats.
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from typing import Literal

from hud.tools.types import ContentResult, ToolError


@dataclass
class ShellCallOutcome:
    """Outcome of a shell command execution (OpenAI format)."""

    type: Literal["exit", "timeout"]
    exit_code: int | None = None

    def to_dict(self) -> dict[str, object]:
        if self.type == "timeout":
            return {"type": "timeout"}
        return {"type": "exit", "exit_code": self.exit_code}


@dataclass
class ShellCommandOutput:
    """Output of a single shell command execution (OpenAI format)."""

    stdout: str
    stderr: str
    outcome: ShellCallOutcome

    def to_dict(self) -> dict[str, object]:
        return {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "outcome": self.outcome.to_dict(),
        }

    def to_content_result(self) -> ContentResult:
        """Convert to ContentResult format (Claude/MCP)."""
        if self.outcome.type == "timeout":
            return ContentResult(
                output=self.stdout,
                error=self.stderr or "Command timed out",
                system="timeout",
            )

        error_msg = self.stderr
        if self.outcome.exit_code and self.outcome.exit_code != 0:
            if error_msg:
                error_msg = f"Exit code {self.outcome.exit_code}: {error_msg}"
            else:
                error_msg = f"Exit code {self.outcome.exit_code}"

        return ContentResult(output=self.stdout, error=error_msg if error_msg else None)


class BashSession:
    """A persistent bash shell session.

    This session can be used by both BashTool (Claude) and ShellTool (OpenAI).
    The main differences are in the output format, not the session logic.
    """

    _started: bool
    _process: asyncio.subprocess.Process
    _timed_out: bool

    # Platform-specific shell command
    command: str = "cmd.exe" if sys.platform == "win32" else "/bin/bash"
    _output_delay: float = 0.2  # seconds for polling mode
    _sentinel: str = "<<exit>>"
    _default_timeout: float = 120.0  # seconds

    def __init__(self, cwd: str | None = None) -> None:
        self._started = False
        self._timed_out = False
        self._cwd = cwd

    async def start(self) -> None:
        """Start the bash session."""
        if self._started:
            await asyncio.sleep(0)
            return

        # Platform-specific process creation
        preexec_fn = None
        if sys.platform != "win32":
            # On Unix, use setsid for process group isolation
            # If running as root (e.g., Docker), also demote to uid 1000
            if os.getuid() == 0:

                def demote() -> None:
                    os.setsid()  # type: ignore[attr-defined]
                    os.setgid(1000)  # type: ignore[attr-defined]
                    os.setuid(1000)  # type: ignore[attr-defined]

                preexec_fn = demote
            else:
                preexec_fn = os.setsid  # type: ignore[attr-defined]

        if sys.platform != "win32":
            self._process = await asyncio.create_subprocess_shell(
                self.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                preexec_fn=preexec_fn,
                cwd=self._cwd,
            )
        else:
            self._process = await asyncio.create_subprocess_shell(
                self.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._cwd,
            )

        self._started = True
        self._timed_out = False

    def stop(self) -> None:
        """Terminate the bash shell."""
        if not self._started:
            return
        if self._process.returncode is not None:
            return
        self._process.terminate()

    def is_alive(self) -> bool:
        """Check if the session is alive and usable."""
        return self._started and self._process.returncode is None and not self._timed_out

    async def run(
        self,
        command: str,
        timeout_ms: int | None = None,
        capture_exit_code: bool = True,
    ) -> ShellCommandOutput:
        """Execute a command in the bash shell.

        Args:
            command: Shell command to execute
            timeout_ms: Timeout in milliseconds (default: 120000ms)
            capture_exit_code: Whether to capture exit code via sentinel

        Returns:
            ShellCommandOutput with stdout, stderr, and outcome
        """
        if not self._started:
            raise ToolError("Session has not started.")

        if self._process.returncode is not None:
            return ShellCommandOutput(
                stdout="",
                stderr=f"bash has exited with returncode {self._process.returncode}",
                outcome=ShellCallOutcome(type="exit", exit_code=self._process.returncode),
            )

        if self._timed_out:
            raise ToolError(
                f"timed out: bash did not return in {self._default_timeout} seconds "
                "and must be restarted"
            )

        timeout_sec = (timeout_ms / 1000.0) if timeout_ms else self._default_timeout

        assert self._process.stdin
        assert self._process.stdout
        assert self._process.stderr

        # Send command with sentinel for exit code capture
        # Platform-specific syntax for command chaining and exit code
        if sys.platform == "win32":
            if capture_exit_code:
                cmd_line = f"{command} & echo {self._sentinel}%errorlevel%\n"
            else:
                cmd_line = f"{command} & echo {self._sentinel}\n"
        else:
            if capture_exit_code:
                cmd_line = f"{command}; echo '{self._sentinel}'$?\n"
            else:
                cmd_line = f"{command}; echo '{self._sentinel}'\n"

        self._process.stdin.write(cmd_line.encode())
        await self._process.stdin.drain()

        output = ""
        error = ""
        exit_code: int | None = None

        try:
            async with asyncio.timeout(timeout_sec):
                while True:
                    await asyncio.sleep(self._output_delay)
                    # Read from buffer without blocking
                    output = self._process.stdout._buffer.decode()  # pyright: ignore[reportAttributeAccessIssue]
                    error = self._process.stderr._buffer.decode()  # pyright: ignore[reportAttributeAccessIssue]

                    if self._sentinel in output:
                        sentinel_idx = output.index(self._sentinel)
                        after_sentinel = output[sentinel_idx + len(self._sentinel) :]
                        newline_idx = after_sentinel.find("\n")

                        if capture_exit_code:
                            if newline_idx != -1:
                                exit_code_str = after_sentinel[:newline_idx].strip()
                            else:
                                exit_code_str = after_sentinel.strip()
                            try:
                                exit_code = int(exit_code_str)
                            except ValueError:
                                exit_code = 0

                        output = output[:sentinel_idx]
                        break

        except TimeoutError:
            self._timed_out = True
            self._process.stdout._buffer.clear()  # pyright: ignore[reportAttributeAccessIssue]
            self._process.stderr._buffer.clear()  # pyright: ignore[reportAttributeAccessIssue]
            return ShellCommandOutput(
                stdout=output,
                stderr=error,
                outcome=ShellCallOutcome(type="timeout"),
            )

        # Clean up output
        if output.endswith("\n"):
            output = output[:-1]
        if error.endswith("\n"):
            error = error[:-1]

        # Clear buffers for next command
        self._process.stdout._buffer.clear()  # pyright: ignore[reportAttributeAccessIssue]
        self._process.stderr._buffer.clear()  # pyright: ignore[reportAttributeAccessIssue]

        return ShellCommandOutput(
            stdout=output,
            stderr=error,
            outcome=ShellCallOutcome(type="exit", exit_code=exit_code),
        )


__all__ = ["BashSession", "ShellCallOutcome", "ShellCommandOutput"]
