"""SSHTool: capability base for tools driven by an ``SSHClient``."""

from __future__ import annotations

import re
import shlex

import asyncssh

from hud.agents.tools.base import AgentTool, tool_err, tool_ok
from hud.agents.tools.file_view import view_file
from hud.capabilities.ssh import SSHClient
from hud.types import TextContent, ToolResult
from hud.utils.process import bound_shell_output

MAX_SHELL_OUTPUT_LENGTH = 10 * 1024 * 1024


class SSHInfrastructureErrorResult(ToolResult):
    """Internal marker for SSH failures which count toward the circuit breaker."""


def _remote_error(exc: asyncssh.ProcessError) -> str:
    """The remote command's stderr: a failed file op is a tool outcome, not an exception."""
    stderr = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else exc.stderr
    return (stderr or "").strip() or f"exit {exc.exit_status}"


def _find_tests(path: str, pattern: str) -> str:
    """The ``find`` tests for a glob: the base name, or the whole path once it names a directory.

    ``**/`` stands for any number of directories, none included; ``-path`` lets ``*`` cross them.
    """
    if "/" not in pattern:
        return f"-name {shlex.quote(pattern)}"
    head, *rest = pattern.split("**/")
    globs = [re.sub(r"([*?\[\\])", r"\\\1", path.rstrip("/")) + "/" + head]
    for part in rest:
        globs = [glob + directories + part for glob in globs for directories in ("", "*/")]
    return " -o ".join(f"-path {shlex.quote(glob)}" for glob in globs)


class SSHTool(AgentTool[SSHClient]):
    """Capability base: tool driven by an ``SSHClient``."""

    client_type = SSHClient

    async def bash(self, command: str) -> ToolResult:
        """Run a shell command. Returns combined stdout/stderr + exit code."""
        completed = await self.client.run(command, check=False)
        stdout = completed.stdout if isinstance(completed.stdout, str) else ""
        stderr = completed.stderr if isinstance(completed.stderr, str) else ""
        stdout, stderr = bound_shell_output(stdout, stderr, MAX_SHELL_OUTPUT_LENGTH)
        body = f"$ {command}\n{stdout}"
        if stderr:
            body += f"\nstderr:\n{stderr}"
        body += f"\n(exit {completed.returncode})"
        return ToolResult(
            content=[TextContent(type="text", text=body)],
            isError=completed.returncode != 0,
        )

    async def file_read(self, path: str) -> ToolResult:
        try:
            return tool_ok(await self.client.read_text(path))
        except asyncssh.ProcessError as e:
            return tool_err(_remote_error(e))

    async def file_view(self, path: str) -> ToolResult:
        """Read a file as text or image content through SSH exec."""
        try:
            data = await self.client.read_bytes(path)
        except asyncssh.ProcessError as e:
            return tool_err(_remote_error(e))
        return view_file(data)

    async def file_write(self, path: str, content: str) -> ToolResult:
        try:
            await self.client.write_text(path, content)
        except asyncssh.ProcessError as e:
            return tool_err(_remote_error(e))
        return tool_ok(f"wrote {len(content.encode('utf-8'))} bytes to {path}")

    async def file_list(self, path: str = "/") -> ToolResult:
        try:
            names = await self.client.listdir(path)
        except asyncssh.ProcessError as e:
            return tool_err(_remote_error(e))
        return tool_ok("\n".join(names) if names else "(empty)")

    async def file_glob(self, pattern: str, path: str) -> ToolResult:
        """Find files under ``path`` whose name, or path below it, matches a glob."""
        return await self.bash(f"find {shlex.quote(path)} {_find_tests(path, pattern)}")
