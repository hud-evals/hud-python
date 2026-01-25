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
from dataclasses import dataclass
from typing import ClassVar

from hud.tools.base import BaseTool
from hud.tools.native_types import NativeToolSpec, NativeToolSpecs
from hud.tools.types import ContentResult, ToolError
from hud.types import AgentType


@dataclass
class ShellOutput:
    """Output from a shell command execution."""

    command: str
    directory: str
    stdout: str
    stderr: str
    exit_code: int | None

    def to_content_result(self) -> ContentResult:
        """Convert to ContentResult format."""
        output_parts = []
        if self.stdout:
            output_parts.append(self.stdout)

        error_parts = []
        if self.stderr:
            error_parts.append(self.stderr)
        if self.exit_code and self.exit_code != 0:
            error_parts.append(f"Exit code: {self.exit_code}")

        return ContentResult(
            output="\n".join(output_parts) if output_parts else "",
            error="\n".join(error_parts) if error_parts else None,
        )


class GeminiShellTool(BaseTool):
    """Gemini CLI-style shell command execution.

    A simpler shell interface that executes a single command with optional
    working directory. Unlike ShellTool (OpenAI), this doesn't maintain
    persistent sessions - each command runs in a fresh subprocess.

    Parameters (matching Gemini CLI):
        command: The exact shell command to execute (required)
        description: Brief description of the command's purpose (optional)
        directory: Directory relative to project root to execute in (optional)

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
        super().__init__(
            env=None,
            name="run_shell_command",
            title="Shell Command",
            description=(
                "Execute a shell command. On Windows, uses powershell.exe. "
                "On other platforms, uses bash -c."
            ),
        )
        self._base_directory = os.path.abspath(base_directory)

    def _resolve_directory(self, directory: str | None) -> str:
        """Resolve directory relative to base directory."""
        if directory is None:
            return self._base_directory
        if os.path.isabs(directory):
            return directory
        return os.path.normpath(os.path.join(self._base_directory, directory))

    async def __call__(
        self,
        command: str,
        description: str | None = None,
        directory: str | None = None,
        timeout_ms: int | None = None,
    ) -> ContentResult:
        """Execute a shell command.

        Args:
            command: The exact shell command to execute
            description: Brief description of the command's purpose (for logging)
            directory: Directory relative to project root to execute in
            timeout_ms: Timeout in milliseconds (default: 120000)

        Returns:
            ContentResult with stdout/stderr
        """
        if not command:
            raise ToolError("command is required")

        work_dir = self._resolve_directory(directory)
        if not os.path.isdir(work_dir):
            raise ToolError(f"Directory does not exist: {work_dir}")

        timeout_sec = (timeout_ms / 1000.0) if timeout_ms else 120.0

        # Choose shell based on platform
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
            except TimeoutError:
                process.kill()
                await process.wait()
                return ContentResult(
                    output="",
                    error=f"Command timed out after {timeout_sec}s",
                    system="timeout",
                )

            stdout = stdout_bytes.decode("utf-8", errors="replace").rstrip("\n")
            stderr = stderr_bytes.decode("utf-8", errors="replace").rstrip("\n")

            output = ShellOutput(
                command=command,
                directory=work_dir,
                stdout=stdout,
                stderr=stderr,
                exit_code=process.returncode,
            )

            return output.to_content_result()

        except Exception as e:
            raise ToolError(f"Failed to execute command: {e}") from e


__all__ = ["GeminiShellTool", "ShellOutput"]
