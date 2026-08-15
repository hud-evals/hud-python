"""SSHTool: capability base for tools driven by an ``SSHClient``.

Provider tools (``ClaudeBashTool``, ``GeminiShellTool``, …) extend this and
use ``self.bash`` / ``self.file_*`` for execution; only the LLM-facing schema
differs between providers.
"""

from __future__ import annotations

import asyncssh
import mcp.types as mcp_types

from hud.agents.tools.base import AgentTool, tool_err, tool_ok
from hud.capabilities import SSHClient
from hud.types import MCPToolResult

MAX_SHELL_OUTPUT_LENGTH = 10 * 1024 * 1024
TRUNCATION_MARKER = "[truncated]"


class SSHInfrastructureErrorResult(MCPToolResult):
    """Internal marker for SSH failures which count toward the circuit breaker."""


def _remote_error(exc: asyncssh.ProcessError) -> str:
    """What the remote command printed to stderr — a failed file op is an
    ordinary tool outcome, so the agent gets the shell's message, not the
    exception's repr."""
    stderr = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else exc.stderr
    return (stderr or "").strip() or f"exit {exc.exit_status}"


class SSHTool(AgentTool[SSHClient]):
    """Capability base: tool driven by an ``SSHClient``."""

    client_type = SSHClient

    # ─── action helpers ───────────────────────────────────────────────

    async def bash(self, command: str) -> MCPToolResult:
        """Run a shell command. Returns combined stdout/stderr + exit code."""
        completed = await self.client.run(command, check=False)
        stdout = completed.stdout if isinstance(completed.stdout, str) else ""
        stderr = completed.stderr if isinstance(completed.stderr, str) else ""
        stdout, stderr = bound_shell_output(stdout, stderr, MAX_SHELL_OUTPUT_LENGTH)
        body = f"$ {command}\n{stdout}"
        if stderr:
            body += f"\nstderr:\n{stderr}"
        body += f"\n(exit {completed.returncode})"
        return MCPToolResult(
            content=[mcp_types.TextContent(type="text", text=body)],
            isError=completed.returncode != 0,
        )

    async def file_read(self, path: str) -> MCPToolResult:
        """Read a text file through SSH exec."""
        try:
            return tool_ok(await self.client.read_text(path))
        except asyncssh.ProcessError as e:
            return tool_err(_remote_error(e))

    async def file_write(self, path: str, content: str) -> MCPToolResult:
        """Write a text file through SSH exec."""
        try:
            await self.client.write_text(path, content)
        except asyncssh.ProcessError as e:
            return tool_err(_remote_error(e))
        return tool_ok(f"wrote {len(content)} bytes to {path}")

    async def file_list(self, path: str = "/") -> MCPToolResult:
        """List directory entries through SSH exec."""
        try:
            names = await self.client.listdir(path)
        except asyncssh.ProcessError as e:
            return tool_err(_remote_error(e))
        return tool_ok("\n".join(names) if names else "(empty)")


def bound_shell_output(stdout: str, stderr: str, limit: int) -> tuple[str, str]:
    if len(stdout) + len(stderr) <= limit:
        return stdout, stderr

    marker = TRUNCATION_MARKER[:limit]
    available = limit - len(marker)
    prefix_length = (available + 1) // 2
    suffix_length = available // 2

    stdout_prefix = stdout[:prefix_length]
    stderr_prefix = stderr[: max(0, prefix_length - len(stdout))]
    stderr_suffix = stderr[-suffix_length:] if suffix_length else ""
    stdout_suffix_length = max(0, suffix_length - len(stderr))
    stdout_suffix = stdout[-stdout_suffix_length:] if stdout_suffix_length else ""

    if len(stdout_prefix) + len(stdout_suffix) < len(stdout):
        stdout = stdout_prefix + marker + stdout_suffix
        stderr = stderr_prefix + stderr_suffix
    else:
        stdout = stdout_prefix + stdout_suffix
        stderr = stderr_prefix + marker + stderr_suffix
    return stdout, stderr


__all__ = ["SSHTool"]
