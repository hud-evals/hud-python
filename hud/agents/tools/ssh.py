"""SSHTool: capability base for tools driven by an ``SSHClient``.

Provider tools (``ClaudeBashTool``, ``GeminiShellTool``, …) extend this and
use ``self.bash`` / ``self.file_*`` for execution; only the LLM-facing schema
differs between providers.
"""

from __future__ import annotations

from typing import cast

import mcp.types as mcp_types

from hud.agents.tools.base import AgentTool
from hud.capabilities import SSHClient
from hud.types import MCPToolResult


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
        """Read a file via SFTP."""
        async with self.client.conn.start_sftp_client() as sftp, sftp.open(path, "rb") as f:
            raw = cast("bytes | str", await f.read())
        data = raw.encode("utf-8", errors="replace") if isinstance(raw, str) else raw
        return _ok(data.decode("utf-8", errors="replace"))

    async def file_write(self, path: str, content: str) -> MCPToolResult:
        """Write a file via SFTP (overwrites)."""
        async with self.client.conn.start_sftp_client() as sftp, sftp.open(path, "wb") as f:
            await f.write(content.encode("utf-8"))
        return _ok(f"wrote {len(content)} bytes to {path}")

    async def file_list(self, path: str = "/") -> MCPToolResult:
        """List directory entries via SFTP."""
        async with self.client.conn.start_sftp_client() as sftp:
            entries = cast("list[bytes | str]", await sftp.listdir(path))
        names = sorted(
            (e if isinstance(e, str) else e.decode("utf-8", errors="replace"))
            for e in entries
        )
        names = [n for n in names if n not in (".", "..")]
        return _ok("\n".join(names) if names else "(empty)")


def _ok(text: str) -> MCPToolResult:
    return MCPToolResult(content=[mcp_types.TextContent(type="text", text=text)])


def result_text(result: MCPToolResult) -> str:
    """Extract concatenated text from a MCPToolResult's TextContent blocks."""
    return "".join(
        block.text for block in result.content if isinstance(block, mcp_types.TextContent)
    )


__all__ = ["SSHTool", "result_text"]
