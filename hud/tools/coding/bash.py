"""Bash tool for Claude agents.

This tool conforms to Anthropic's bash tool specification and is used
when running with Claude models that support native bash.

Note: This uses a simpler readuntil-based session compared to ShellTool's
polling-based session, as Claude's bash API has different timeout handling.
"""

from __future__ import annotations

import asyncio
from typing import ClassVar

from mcp.types import ContentBlock  # noqa: TC002

from hud.tools.base import BaseTool
from hud.tools.native_types import NativeToolSpec, NativeToolSpecs
from hud.tools.types import ContentResult, ToolError
from hud.types import AgentType

from .utils import get_demote_preexec_fn


class ClaudeBashSession:
    """A persistent bash shell session for Claude's bash tool.

    Uses readuntil-based output capture, which is simpler than ShellTool's
    polling approach but doesn't support dynamic timeouts.
    """

    _started: bool
    _process: asyncio.subprocess.Process
    _timed_out: bool

    command: str = "/bin/bash"
    _timeout: float = 120.0  # seconds
    _sentinel: str = "<<exit>>"

    def __init__(self) -> None:
        self._started = False
        self._timed_out = False

    async def start(self) -> None:
        """Start the bash session."""
        if self._started:
            await asyncio.sleep(0)
            return

        self._process = await asyncio.create_subprocess_shell(
            self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            preexec_fn=get_demote_preexec_fn(),
        )

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


# Alias for backward compatibility
_BashSession = ClaudeBashSession


class BashTool(BaseTool):
    """A tool that allows the agent to run bash commands.

    The tool maintains a persistent bash session that can be restarted.
    This is the Claude-native version that returns ContentResult format
    and supports manual restart via the `restart` parameter.

    Native specs: Claude (bash_20250124)
    Role: "shell" (mutually exclusive with ShellTool)
    Supported models: Claude 3.5 Sonnet, 3.7 Sonnet, Sonnet 4, Opus 4
    """

    native_specs: ClassVar[NativeToolSpecs] = {
        AgentType.CLAUDE: NativeToolSpec(
            api_type="bash_20250124",
            api_name="bash",
            beta="computer-use-2025-01-24",
            role="shell",
            # Claude models that support computer use / bash tool
            # https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/bash-tool
            supported_models=(
                "claude-3-5-sonnet-*",
                "claude-3-7-sonnet-*",
                "claude-sonnet-4-*",
                "claude-opus-4-*",
                "claude-4-5-sonnet-*",
                "claude-4-5-opus-*",
            ),
        ),
    }

    def __init__(self, session: ClaudeBashSession | None = None) -> None:
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
    def session(self) -> ClaudeBashSession | None:
        """Get the current bash session."""
        return self.env

    @session.setter
    def session(self, value: ClaudeBashSession | None) -> None:
        """Set the bash session."""
        self.env = value

    async def __call__(
        self, command: str | None = None, restart: bool = False
    ) -> list[ContentBlock]:
        """Execute a bash command or restart the session.

        Args:
            command: Shell command to execute
            restart: If True, restart the bash session

        Returns:
            List of MCP ContentBlocks with the result
        """
        if restart:
            session_cls = type(self.session) if self.session else ClaudeBashSession
            if self.session:
                self.session.stop()
            self.session = session_cls()
            await self.session.start()
            return ContentResult(output="Bash session restarted.").to_content_blocks()

        if self.session is None:
            self.session = ClaudeBashSession()

        if not self.session._started:
            await self.session.start()

        if command is not None:
            result = await self.session.run(command)
            return result.to_content_blocks()

        raise ToolError("No command provided.")


__all__ = ["BashTool", "ClaudeBashSession", "_BashSession"]
