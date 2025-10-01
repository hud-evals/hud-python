from __future__ import annotations

import asyncio
import os
import sys
from typing import TYPE_CHECKING

from .base import BaseTool
from .types import ContentResult, ToolError

if TYPE_CHECKING:
    from mcp.types import ContentBlock


class _BashSession:
    """A session of a bash shell."""

    _started: bool
    _process: asyncio.subprocess.Process

    command: str = "/bin/bash"
    _output_delay: float = 0.2  # seconds
    _timeout: float = 120.0  # seconds
    _sentinel: str = "<<exit>>"

    def __init__(self) -> None:
        self._started = False
        self._timed_out = False

    async def start(self) -> None:
        if self._started:
            await asyncio.sleep(0)
            return

        # Platform-specific subprocess creation
        kwargs = {
            "shell": True,
            "bufsize": 0,
            "stdin": asyncio.subprocess.PIPE,
            "stdout": asyncio.subprocess.PIPE,
            "stderr": asyncio.subprocess.PIPE,
        }

        # Only use setsid on Unix-like systems
        if sys.platform != "win32":
            kwargs["preexec_fn"] = os.setsid

        self._process = await asyncio.create_subprocess_shell(self.command, **kwargs)

        self._started = True

    def stop(self) -> None:
        """Terminate the bash shell."""
        if not self._started:
            raise ToolError("Session has not started.")
        if self._process.returncode is not None:
            return
        self._process.terminate()

    async def run(self, command: str) -> ContentResult:
        """Execute a command in the bash shell."""
        if not self._started:
            raise ToolError("Session has not started.")
        if self._process.returncode is not None:
            await asyncio.sleep(0)
            return ContentResult(
                system="tool must be restarted",
                error=f"bash has exited with returncode {self._process.returncode}",
            )
        if self._timed_out:
            raise ToolError(
                f"timed out: bash did not return in {self._timeout} seconds and must be restarted",
            ) from None

        if self._process.stdin is None:
            raise ToolError("stdin is None")
        if self._process.stdout is None:
            raise ToolError("stdout is None")
        if self._process.stderr is None:
            raise ToolError("stderr is None")

        # Send command to the process
        self._process.stdin.write(command.encode() + f"; echo '{self._sentinel}'\n".encode())
        await self._process.stdin.drain()

        # Read output from the process, until the sentinel is found
        sentinel_line = f"{self._sentinel}\n"
        sentinel_bytes = sentinel_line.encode()

        try:
            raw_out: bytes = await asyncio.wait_for(
                self._process.stdout.readuntil(sentinel_bytes),
                timeout=self._timeout,
            )
            output = raw_out.decode()[: -len(sentinel_line)]
        except (TimeoutError, asyncio.LimitOverrunError):
            self._timed_out = True
            raise ToolError(
                f"timed out: bash did not return in {self._timeout} seconds and must be restarted",
            ) from None

        # Attempt non-blocking stderr fetch (may return empty)
        try:
            error_bytes = await asyncio.wait_for(self._process.stderr.read(), timeout=0.01)
            error = error_bytes.decode().rstrip("\n")
        except TimeoutError:
            error = ""

        return ContentResult(output=output, error=error)


class BashTool(BaseTool):
    """
    A tool that allows the agent to run bash commands.
    The tool maintains a persistent bash session that can be restarted.
    """

    def __init__(self, session: _BashSession | None = None) -> None:
        """Initialize BashTool with an optional session.

        Args:
            session: Optional pre-configured bash session. If not provided,
                     a new session will be created on first use.
        """
        super().__init__(
            env=session,
            name="bash",
            title="Bash Shell",
            description="Execute bash commands in a persistent shell session",
        )

    @property
    def session(self) -> _BashSession | None:
        """Get the current bash session (alias for context)."""
        return self.env

    @session.setter
    def session(self, value: _BashSession | None) -> None:
        """Set the bash session (alias for context)."""
        self.env = value

    async def __call__(
        self, command: str | None = None, restart: bool = False
    ) -> list[ContentBlock]:
        if restart:
            if self.session:
                self.session.stop()
            self.session = _BashSession()
            await self.session.start()

            return ContentResult(output="Bash session restarted.").to_content_blocks()

        if self.session is None:
            self.session = _BashSession()
            await self.session.start()

        if command is not None:
            result = await self.session.run(command)
            return result.to_content_blocks()

        raise ToolError("No command provided.")
