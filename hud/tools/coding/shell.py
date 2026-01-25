"""Shell tool for OpenAI agents.

This tool conforms to OpenAI's shell tool specification:
https://platform.openai.com/docs/guides/tools-shell

Key features:
- Auto-restart on error (no manual restart command)
- Dynamic timeout via timeout_ms from agent
- Dynamic max_output_length from agent (passed back, not truncated locally)
- Output conforms to shell_call_output format
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from hud.tools.base import BaseTool
from hud.tools.native_types import NativeToolSpec, NativeToolSpecs
from hud.tools.types import ToolError
from hud.types import AgentType

from .session import BashSession, ShellCallOutcome, ShellCommandOutput


@dataclass
class ShellResult:
    """Result of shell tool execution, conforming to shell_call_output format."""

    output: list[ShellCommandOutput]
    max_output_length: int | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "output": [o.to_dict() for o in self.output],
        }
        if self.max_output_length is not None:
            result["max_output_length"] = self.max_output_length
        return result


class ShellTool(BaseTool):
    """A tool that allows the agent to run shell commands.

    Conforms to OpenAI's shell tool specification with:
    - Auto-restart on error (session automatically restarts if needed)
    - Dynamic timeout via timeout_ms parameter
    - Dynamic max_output_length (passed back to API, no local truncation)
    - Supports concurrent command execution

    Native specs: OpenAI (shell)
    """

    native_specs: ClassVar[NativeToolSpecs] = {
        AgentType.OPENAI: NativeToolSpec(
            api_type="shell",
            api_name="shell",
            role="shell",
        ),
    }

    _session: BashSession | None

    def __init__(self, session: BashSession | None = None) -> None:
        """Initialize ShellTool with an optional session.

        Args:
            session: Optional pre-configured bash session. If not provided,
                     a new session will be created on first use.
        """
        super().__init__(
            env=session,
            name="shell",
            title="Shell",
            description="Execute shell commands in a persistent bash session",
        )
        self._session = session

    async def _ensure_session(self) -> tuple[BashSession, str | None]:
        """Ensure a working session exists, auto-restarting if needed.

        Returns:
            Tuple of (session, restart_message) where restart_message is set
            if the session was restarted due to an error.
        """
        restart_message = None

        if self._session is not None and not self._session.is_alive():
            old_session = self._session
            if old_session._timed_out:
                restart_message = "Previous session timed out. Session auto-restarted."
            elif old_session._process.returncode is not None:
                restart_message = (
                    f"Previous session exited with code {old_session._process.returncode}. "
                    "Session auto-restarted."
                )
            else:
                restart_message = "Previous session was not usable. Session auto-restarted."
            old_session.stop()
            self._session = None

        if self._session is None:
            self._session = BashSession()
            await self._session.start()

        return self._session, restart_message

    async def __call__(
        self,
        commands: list[str] | None = None,
        timeout_ms: int | None = None,
        max_output_length: int | None = None,
        **kwargs: object,
    ) -> ShellResult:
        """Execute shell commands.

        Args:
            commands: List of shell commands to execute
            timeout_ms: Optional timeout in milliseconds for each command
            max_output_length: Optional max output length (passed back to API)

        Returns:
            ShellResult conforming to shell_call_output format
        """
        if not commands:
            raise ToolError("No commands provided.")

        session, restart_message = await self._ensure_session()
        outputs: list[ShellCommandOutput] = []

        for command in commands:
            if not session.is_alive():
                session, new_restart_msg = await self._ensure_session()
                if new_restart_msg:
                    restart_message = new_restart_msg

            try:
                result = await session.run(command, timeout_ms)

                if restart_message:
                    if result.stderr:
                        result.stderr = f"[SYSTEM: {restart_message}]\n{result.stderr}"
                    else:
                        result.stderr = f"[SYSTEM: {restart_message}]"
                    restart_message = None

                outputs.append(result)
            except Exception as e:
                outputs.append(
                    ShellCommandOutput(
                        stdout="",
                        stderr=str(e),
                        outcome=ShellCallOutcome(type="exit", exit_code=1),
                    )
                )

        return ShellResult(
            output=outputs,
            max_output_length=max_output_length,
        )


__all__ = ["ShellTool", "ShellResult"]
