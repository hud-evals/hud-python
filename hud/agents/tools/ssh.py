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
        completed = await self.client.conn.run(command, check=False)
        stdout = completed.stdout if isinstance(completed.stdout, str) else ""
        stderr = completed.stderr if isinstance(completed.stderr, str) else ""
        body = f"$ {command}\n{stdout}"
        if stderr:
            body += f"\nstderr:\n{stderr}"
        body += f"\n(exit {completed.exit_status})"
        return MCPToolResult(
            content=[mcp_types.TextContent(type="text", text=body)],
            isError=bool(completed.exit_status),
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


__all__ = ["SSHTool"]
