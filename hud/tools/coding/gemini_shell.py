"""Gemini-style shell tool implementation.

Based on Gemini CLI's run_shell_command tool:
https://github.com/google-gemini/gemini-cli

This is a simpler shell interface compared to OpenAI's ShellTool,
designed for single command execution with optional working directory.
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass, field
from typing import ClassVar

from mcp.types import ContentBlock  # noqa: TC002

from hud.tools.base import BaseTool
from hud.tools.native_types import NativeToolSpec, NativeToolSpecs
from hud.tools.types import ContentResult, ToolError
from hud.types import AgentType


@dataclass
class GeminiShellOutput:
    """Output from a shell command execution in Gemini CLI format."""

    command: str
    directory: str
    stdout: str
    stderr: str
    exit_code: int | None
    signal: str | None = None
    pid: int | None = None
    background_pids: list[int] = field(default_factory=list)

    def to_llm_content(self) -> str:
        """Format output for LLM consumption (Gemini CLI format)."""
        # Gemini CLI uses this exact format for LLM context
        parts = [
            f"Command: {self.command}",
            f"Directory: {self.directory or '(root)'}",
            f"Output: {self.stdout or '(empty)'}",
            f"Error: {self.stderr or '(none)'}",
            f"Exit Code: {self.exit_code if self.exit_code is not None else '(none)'}",
            f"Signal: {self.signal or '(none)'}",
            f"Background PIDs: {', '.join(map(str, self.background_pids)) or '(none)'}",
            f"Process Group PGID: {self.pid or '(none)'}",
        ]
        return "\n".join(parts)

    def to_content_result(self) -> ContentResult:
        """Convert to ContentResult with Gemini CLI format."""
        llm_content = self.to_llm_content()

        # For display, show just the output if successful, otherwise show error info
        if self.exit_code == 0 and self.stdout:
            display = self.stdout
        elif self.stderr:
            display = f"Error: {self.stderr}"
            if self.exit_code and self.exit_code != 0:
                display += f"\nExit code: {self.exit_code}"
        elif self.exit_code and self.exit_code != 0:
            display = f"Command exited with code: {self.exit_code}"
        else:
            display = "(no output)"

        return ContentResult(output=llm_content, system=display if display != llm_content else None)


class GeminiShellTool(BaseTool):
    """Gemini CLI-style shell command execution.

    A simpler shell interface that executes a single command with optional
    working directory. Unlike ShellTool (OpenAI), this doesn't maintain
    persistent sessions - each command runs in a fresh subprocess.

    Parameters (matching Gemini CLI exactly):
        command: The exact shell command to execute (required)
        description: Brief description of the command for the user (optional)
        dir_path: Path of directory to run command in (optional)

    Output format matches Gemini CLI:
        Command: <command>
        Directory: <directory>
        Output: <stdout>
        Error: <stderr>
        Exit Code: <code>
        Signal: (none)
        Background PIDs: (none)
        Process Group PGID: <pid>

    Native specs: Uses function calling (no native API), but has role="shell"
                  for mutual exclusion with BashTool/ShellTool.
    """

    native_specs: ClassVar[NativeToolSpecs] = {
        # No api_type - uses standard function calling
        # Role ensures mutual exclusion with other shell tools
        AgentType.GEMINI: NativeToolSpec(role="shell"),
    }

    _base_directory: str

    def __init__(self, base_directory: str = ".") -> None:
        """Initialize GeminiShellTool.

        Args:
            base_directory: Base directory for relative paths (project root)
        """
        # Platform-specific shell description
        if sys.platform == "win32":
            shell_desc = (
                "Execute a shell command as `powershell.exe -NoProfile -Command <command>`. "
                "Command can start background processes using Start-Process -NoNewWindow or Start-Job."
            )
        else:
            shell_desc = (
                "Execute a shell command as `bash -c <command>`. "
                "Command can start background processes using &. "
                "Command process group can be terminated as `kill -- -PGID`."
            )

        super().__init__(
            env=None,
            name="run_shell_command",
            title="Shell",
            description=shell_desc,
        )
        self._base_directory = os.path.abspath(base_directory)

    def _resolve_directory(self, dir_path: str | None) -> str:
        """Resolve directory relative to base directory."""
        if dir_path is None:
            return self._base_directory
        if os.path.isabs(dir_path):
            return dir_path
        return os.path.normpath(os.path.join(self._base_directory, dir_path))

    async def __call__(
        self,
        command: str,
        description: str | None = None,
        dir_path: str | None = None,
        timeout_ms: int | None = None,
    ) -> list[ContentBlock]:
        """Execute a shell command.

        Args:
            command: Exact shell command to execute
            description: Brief description of the command for the user
            dir_path: Path of directory to run the command in (optional,
                     defaults to project root). Must be within workspace.
            timeout_ms: Timeout in milliseconds (default: 120000)

        Returns:
            List of ContentBlocks with Gemini CLI formatted output
        """
        if not command or not command.strip():
            raise ToolError("Command cannot be empty.")

        work_dir = self._resolve_directory(dir_path)
        if not os.path.isdir(work_dir):
            raise ToolError(f"Directory does not exist: {work_dir}")

        timeout_sec = (timeout_ms / 1000.0) if timeout_ms else 120.0

        # Choose shell based on platform (matching Gemini CLI behavior)
        if sys.platform == "win32":
            shell_cmd = ["powershell.exe", "-NoProfile", "-Command", command]
        else:
            shell_cmd = ["bash", "-c", command]

        try:
            process = await asyncio.create_subprocess_exec(
                *shell_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=work_dir,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout_sec,
                )
                timed_out = False
            except TimeoutError:
                process.kill()
                await process.wait()
                timed_out = True
                stdout_bytes = b""
                stderr_bytes = b""

            stdout = stdout_bytes.decode("utf-8", errors="replace").rstrip("\n")
            stderr = stderr_bytes.decode("utf-8", errors="replace").rstrip("\n")

            if timed_out:
                # Match Gemini CLI timeout message format
                output = GeminiShellOutput(
                    command=command,
                    directory=dir_path or "(root)",
                    stdout="",
                    stderr=f"Command timed out after {timeout_sec:.1f} seconds",
                    exit_code=None,
                    signal=None,
                    pid=process.pid,
                )
            else:
                output = GeminiShellOutput(
                    command=command,
                    directory=dir_path or "(root)",
                    stdout=stdout,
                    stderr=stderr,
                    exit_code=process.returncode,
                    signal=None,
                    pid=process.pid,
                )

            return output.to_content_result().to_content_blocks()

        except Exception as e:
            raise ToolError(f"Failed to execute command: {e}") from e


__all__ = ["GeminiShellOutput", "GeminiShellTool"]
