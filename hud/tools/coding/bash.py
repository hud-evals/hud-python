"""Environment bash tool."""

from __future__ import annotations

from mcp.types import ContentBlock  # noqa: TC002

from hud.tools.base import BaseTool
from hud.tools.types import ContentResult, ToolError

from .session import BashSession

ClaudeBashSession = BashSession
_BashSession = BashSession


class BashTool(BaseTool):
    """Environment tool for running commands in a persistent bash shell.

    The tool maintains a persistent bash session that can be restarted.
    """

    def __init__(
        self,
        session: BashSession | None = None,
        timeout: float = BashSession.DEFAULT_TIMEOUT,
        name: str = "bash",
        title: str = "Bash Shell",
        description: str = "Execute bash commands in a persistent shell session",
    ) -> None:
        """Initialize BashTool with an optional session.

        Args:
            session: Optional pre-configured bash session. If not provided,
                     a new session will be created on first use.
            timeout: Timeout in seconds for command execution. Defaults to 120s.
                     If a pre-configured session is provided, the timeout is
                     derived from that session instead.
        """
        super().__init__(
            env=session,
            name=name,
            title=title,
            description=description,
            meta={"capability": "shell"},
        )
        self._timeout = session._timeout if session is not None else timeout

    @property
    def session(self) -> BashSession | None:
        """Get the current bash session."""
        return self.env

    @session.setter
    def session(self, value: BashSession | None) -> None:
        """Set the bash session."""
        self.env = value

    def _create_session(self) -> BashSession:
        return ClaudeBashSession(timeout=self._timeout)

    async def __call__(
        self,
        command: str | None = None,
        restart: bool = False,
        timeout_seconds: float | None = None,
    ) -> list[ContentBlock]:
        """Execute a bash command or restart the session.

        Args:
            command: Shell command to execute
            restart: If True, restart the bash session
            timeout_seconds: Optional per-command timeout in seconds

        Returns:
            List of MCP ContentBlocks with the result
        """
        if restart:
            if self.session:
                self.session.stop()
            self.session = self._create_session()
            await self.session.start()
            return ContentResult(output="Bash session restarted.").to_content_blocks()

        if self.session is None:
            self.session = self._create_session()

        if not self.session._started:
            await self.session.start()

        if command is not None:
            timeout = timeout_seconds if timeout_seconds is not None else self._timeout
            timeout_ms = int(timeout * 1000)
            result = await self.session.run(command, timeout_ms=timeout_ms)
            return result.to_content_result().to_content_blocks()

        raise ToolError("No command provided.")


BashToolSession = BashSession


__all__ = ["BashTool", "BashToolSession", "ClaudeBashSession", "_BashSession"]
