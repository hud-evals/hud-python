"""
Shell tool implementation conforming to OpenAI's shell tool specification.
https://platform.openai.com/docs/guides/tools-shell

Key features:
- Auto-restart on error (no manual restart command)
- Dynamic timeout via timeout_ms from agent
- Dynamic max_output_length from agent (passed back, not truncated locally)
- Output conforms to shell_call_output format
"""

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Literal

from .types import ToolError


@dataclass
class ShellCallOutcome:
    """Outcome of a shell command execution."""

    type: Literal["exit", "timeout"]
    exit_code: int | None = None

    def to_dict(self) -> dict:
        if self.type == "timeout":
            return {"type": "timeout"}
        return {"type": "exit", "exit_code": self.exit_code}


@dataclass
class ShellCommandOutput:
    """Output of a single shell command execution."""

    stdout: str
    stderr: str
    outcome: ShellCallOutcome

    def to_dict(self) -> dict:
        return {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "outcome": self.outcome.to_dict(),
        }


@dataclass
class ShellResult:
    """Result of shell tool execution, conforming to shell_call_output format."""

    output: list[ShellCommandOutput]
    max_output_length: int | None = None

    def to_dict(self) -> dict:
        result: dict[str, Any] = {
            "output": [o.to_dict() for o in self.output],
        }
        if self.max_output_length is not None:
            result["max_output_length"] = self.max_output_length
        return result


class _BashSession:
    """A session of a bash shell."""

    _started: bool
    _process: asyncio.subprocess.Process

    command: str = "/bin/bash"
    _output_delay: float = 0.2  # seconds
    _sentinel: str = "<<exit>>"

    def __init__(self) -> None:
        self._started = False
        self._timed_out = False

    async def start(self) -> None:
        if self._started:
            await asyncio.sleep(0)
            return

        def demote() -> None:
            # This only runs in the child process
            os.setsid()
            os.setgid(1000)
            os.setuid(1000)

        self._process = await asyncio.create_subprocess_shell(  # noqa: S604
            self.command,
            preexec_fn=demote,
            shell=True,
            bufsize=0,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
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

    async def run(self, command: str, timeout_ms: int | None = None) -> ShellCommandOutput:
        """Execute a command in the bash shell."""
        if not self._started:
            raise ToolError("Session has not started.")

        # Convert timeout from ms to seconds, default to 120 seconds
        timeout_sec = (timeout_ms / 1000.0) if timeout_ms else 120.0

        # we know these are not None because we created the process with PIPEs
        assert self._process.stdin
        assert self._process.stdout
        assert self._process.stderr

        # send command to the process
        self._process.stdin.write(command.encode() + f"; echo '{self._sentinel}'$?\n".encode())
        await self._process.stdin.drain()

        output = ""
        error = ""
        exit_code = None

        # read output from the process, until the sentinel is found
        try:
            async with asyncio.timeout(timeout_sec):
                while True:
                    await asyncio.sleep(self._output_delay)
                    # if we read directly from stdout/stderr, it will wait forever for
                    # EOF. use the StreamReader buffer directly instead.
                    output = self._process.stdout._buffer.decode()  # pyright: ignore[reportAttributeAccessIssue]
                    error = self._process.stderr._buffer.decode()  # pyright: ignore[reportAttributeAccessIssue]
                    if self._sentinel in output:
                        # Extract exit code from sentinel line
                        sentinel_idx = output.index(self._sentinel)
                        # Find the exit code after the sentinel
                        after_sentinel = output[sentinel_idx + len(self._sentinel) :]
                        newline_idx = after_sentinel.find("\n")
                        if newline_idx != -1:
                            exit_code_str = after_sentinel[:newline_idx].strip()
                        else:
                            exit_code_str = after_sentinel.strip()
                        try:
                            exit_code = int(exit_code_str)
                        except ValueError:
                            exit_code = 0
                        # strip the sentinel and exit code from output
                        output = output[:sentinel_idx]
                        break
        except TimeoutError:
            self._timed_out = True
            # clear the buffers
            self._process.stdout._buffer.clear()  # pyright: ignore[reportAttributeAccessIssue]
            self._process.stderr._buffer.clear()  # pyright: ignore[reportAttributeAccessIssue]

            return ShellCommandOutput(
                stdout=output,
                stderr=error,
                outcome=ShellCallOutcome(type="timeout"),
            )

        if output.endswith("\n"):
            output = output[:-1]

        if error.endswith("\n"):
            error = error[:-1]

        # clear the buffers so that the next output can be read correctly
        self._process.stdout._buffer.clear()  # pyright: ignore[reportAttributeAccessIssue]
        self._process.stderr._buffer.clear()  # pyright: ignore[reportAttributeAccessIssue]

        return ShellCommandOutput(
            stdout=output,
            stderr=error,
            outcome=ShellCallOutcome(type="exit", exit_code=exit_code),
        )


class ShellTool:
    """
    A tool that allows the agent to run shell commands.
    Conforms to OpenAI's shell tool specification.

    Features:
    - Auto-restart on error (session automatically restarts if needed)
    - Dynamic timeout via timeout_ms parameter
    - Dynamic max_output_length (passed back to API, no local truncation)
    - Supports concurrent command execution
    """

    _session: _BashSession | None

    def __init__(self) -> None:
        self._session = None

    async def _ensure_session(self) -> tuple[_BashSession, str | None]:
        """Ensure a working session exists, auto-restarting if needed.

        Returns:
            Tuple of (session, restart_message) where restart_message is set
            if the session was restarted due to an error.
        """
        restart_message = None

        if self._session is not None and not self._session.is_alive():
            # Session exists but is dead - auto-restart
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
            self._session = _BashSession()
            await self._session.start()
            if restart_message is None:
                # First start, no message needed
                pass

        return self._session, restart_message

    async def __call__(
        self,
        commands: list[str] | None = None,
        timeout_ms: int | None = None,
        max_output_length: int | None = None,
        **kwargs: object,
    ) -> ShellResult:
        """
        Execute shell commands.

        Args:
            commands: List of shell commands to execute (can run concurrently).
            timeout_ms: Optional timeout in milliseconds for each command.
            max_output_length: Optional max output length (passed back to API).

        Returns:
            ShellResult conforming to shell_call_output format.
        """
        if not commands:
            raise ToolError("No commands provided.")

        session, restart_message = await self._ensure_session()
        outputs: list[ShellCommandOutput] = []

        # Execute commands - can be done concurrently
        # Note: OpenAI docs say commands can be executed concurrently,
        # but for a single bash session, we run them sequentially.
        # For true concurrency, you'd need multiple sessions or subprocess per command.
        for command in commands:
            # Check if session is still alive before each command
            if not session.is_alive():
                session, new_restart_msg = await self._ensure_session()
                if new_restart_msg:
                    restart_message = new_restart_msg

            try:
                result = await session.run(command, timeout_ms)

                # If we had a restart message, prepend it to the first output's stderr
                if restart_message:
                    if result.stderr:
                        result.stderr = f"[SYSTEM: {restart_message}]\n{result.stderr}"
                    else:
                        result.stderr = f"[SYSTEM: {restart_message}]"
                    restart_message = None  # Only add once

                outputs.append(result)
            except Exception as e:
                # Command execution failed, add error output
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
