"""SSHTool: capability base for tools driven by an ``SSHClient``.

Provider tools (``ClaudeBashTool``, ``GeminiShellTool``, …) extend this and
use ``self.bash`` / ``self.file_*`` for execution; only the LLM-facing schema
differs between providers.
"""

from __future__ import annotations

from typing import cast

import mcp.types as mcp_types

from hud.agents.tools.base import AgentTool, tool_ok
from hud.capabilities import SSHClient
from hud.types import MCPToolResult


class SSHTool(AgentTool[SSHClient]):
    """Capability base: tool driven by an ``SSHClient``."""

    client_type = SSHClient

    # ─── action helpers ───────────────────────────────────────────────

    async def bash_structured(self, command: str) -> tuple[str, str, int]:
        """Run a shell command, returning raw ``(stdout, stderr, exit_code)``.

        The canonical execution primitive; ``bash`` formats this into a single
        human-readable block, while providers that need separate streams (e.g.
        OpenAI's ``shell_call_output``) consume the tuple directly.
        """
        completed = await self.client.conn.run(command, check=False)
        stdout = completed.stdout if isinstance(completed.stdout, str) else ""
        stderr = completed.stderr if isinstance(completed.stderr, str) else ""
        exit_code = completed.exit_status if isinstance(completed.exit_status, int) else 1
        return stdout, stderr, exit_code

    async def bash(self, command: str) -> MCPToolResult:
        """Run a shell command. Returns combined stdout/stderr + exit code."""
        stdout, stderr, exit_code = await self.bash_structured(command)
        body = f"$ {command}\n{stdout}"
        if stderr:
            body += f"\nstderr:\n{stderr}"
        body += f"\n(exit {exit_code})"
        return MCPToolResult(
            content=[mcp_types.TextContent(type="text", text=body)],
            isError=bool(exit_code),
        )

    async def file_read(self, path: str) -> MCPToolResult:
        """Read a file via SFTP."""
        async with self.client.conn.start_sftp_client() as sftp, sftp.open(path, "rb") as f:
            raw = cast("bytes | str", await f.read())
        data = raw.encode("utf-8", errors="replace") if isinstance(raw, str) else raw
        return tool_ok(data.decode("utf-8", errors="replace"))

    async def file_write(self, path: str, content: str) -> MCPToolResult:
        """Write a file via SFTP (overwrites)."""
        async with self.client.conn.start_sftp_client() as sftp, sftp.open(path, "wb") as f:
            await f.write(content.encode("utf-8"))
        return tool_ok(f"wrote {len(content)} bytes to {path}")

    async def file_list(self, path: str = "/") -> MCPToolResult:
        """List directory entries via SFTP."""
        async with self.client.conn.start_sftp_client() as sftp:
            entries = cast("list[bytes | str]", await sftp.listdir(path))
        names = sorted(
            (e if isinstance(e, str) else e.decode("utf-8", errors="replace")) for e in entries
        )
        names = [n for n in names if n not in (".", "..")]
        return tool_ok("\n".join(names) if names else "(empty)")


__all__ = ["SSHTool"]
