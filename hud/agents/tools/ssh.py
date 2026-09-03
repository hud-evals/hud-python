"""SSHTool: capability base for tools driven by an ``SSHClient``.

Provider tools (``ClaudeBashTool``, ``GeminiShellTool``, …) extend this and
use ``self.bash`` / ``self.file_*`` for execution; only the LLM-facing schema
differs between providers.
"""

from __future__ import annotations

import asyncssh
import mcp.types as mcp_types

from hud.agents.tools.base import AgentTool, result_text, tool_err, tool_ok
from hud.capabilities import SSHClient
from hud.capabilities.ssh import (
    TOOL_MAX_OUTPUT_CHARS_PARAM,
    TOOL_MAX_TOTAL_OUTPUT_CHARS_PARAM,
    TOOL_OUTPUT_BUDGET_MARKER_PARAM,
)
from hud.types import MCPToolResult

MAX_SHELL_OUTPUT_LENGTH = 10 * 1024 * 1024
TRUNCATION_MARKER = "[truncated]"
OUTPUT_BUDGET_EXHAUSTED = "[tool output budget exhausted; return an unknown result]"


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

    def output_limit(self) -> int:
        value = self.client.capability.params.get(TOOL_MAX_OUTPUT_CHARS_PARAM)
        if value is None:
            return MAX_SHELL_OUTPUT_LENGTH
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{TOOL_MAX_OUTPUT_CHARS_PARAM} must be a positive integer")
        return min(value, MAX_SHELL_OUTPUT_LENGTH)

    async def bound_output(
        self,
        stdout: str,
        stderr: str,
        *,
        limit: int | None = None,
    ) -> tuple[str, str]:
        configured_limit = self.output_limit()
        if limit is None:
            per_result_limit = configured_limit
        else:
            if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
                raise ValueError("output limit must be a positive integer")
            per_result_limit = min(limit, configured_limit)
        desired = min(len(stdout) + len(stderr), per_result_limit)
        notice_chars = len(OUTPUT_BUDGET_EXHAUSTED) + 1
        if (
            self.client.capability.params.get(TOOL_MAX_TOTAL_OUTPUT_CHARS_PARAM) is not None
            and per_result_limit < notice_chars
        ):
            raise ValueError(f"{TOOL_MAX_OUTPUT_CHARS_PARAM} must fit the output budget notice")
        allowed, exhausted, notify = await self.client.claim_output_chars(
            desired,
            exhaustion_notice_chars=notice_chars,
        )
        if notify:
            allowed = min(allowed, per_result_limit - notice_chars)
        stdout, stderr = bound_shell_output(stdout, stderr, allowed)
        if exhausted:
            marker_path = self.client.capability.params.get(TOOL_OUTPUT_BUDGET_MARKER_PARAM)
            if isinstance(marker_path, str) and marker_path:
                await self.client.write_text(marker_path, OUTPUT_BUDGET_EXHAUSTED)
            if notify:
                stderr = f"{stderr}\n{OUTPUT_BUDGET_EXHAUSTED}".lstrip()
        return stdout, stderr

    async def bound_text(self, text: str, *, limit: int | None = None) -> str:
        text, warning = await self.bound_output(text, "", limit=limit)
        return text + (f"\n{warning}" if warning else "")

    async def bash(self, command: str) -> MCPToolResult:
        """Run a shell command. Returns combined stdout/stderr + exit code."""
        completed = await self.client.run(command, check=False)
        stdout = completed.stdout if isinstance(completed.stdout, str) else ""
        stderr = completed.stderr if isinstance(completed.stderr, str) else ""
        body = f"$ {command}\n{stdout}"
        if stderr:
            body += f"\nstderr:\n{stderr}"
        body += f"\n(exit {completed.returncode})"
        body = await self.bound_text(body)
        return MCPToolResult(
            content=[mcp_types.TextContent(type="text", text=body)],
            isError=completed.returncode != 0,
        )

    async def raw_file_read(self, path: str) -> MCPToolResult:
        """Read a text file without changing content used by mutation helpers."""
        try:
            text = await self.client.read_text(path)
        except asyncssh.ProcessError as e:
            return tool_err(_remote_error(e))
        return tool_ok(text)

    async def file_read(self, path: str) -> MCPToolResult:
        """Read a text file and bound the provider-visible result."""
        result = await self.raw_file_read(path)
        if result.isError:
            return result
        return tool_ok(await self.bound_text(result_text(result)))

    async def file_write(self, path: str, content: str) -> MCPToolResult:
        """Write a text file through SSH exec."""
        try:
            await self.client.write_text(path, content)
        except asyncssh.ProcessError as e:
            return tool_err(_remote_error(e))
        return tool_ok(await self.bound_text(f"wrote {len(content)} bytes to {path}"))

    async def raw_file_list(self, path: str = "/") -> MCPToolResult:
        """List directory entries before provider-facing pagination."""
        try:
            names = await self.client.listdir(path)
        except asyncssh.ProcessError as e:
            return tool_err(_remote_error(e))
        return tool_ok("\n".join(names) if names else "(empty)")

    async def file_list(self, path: str = "/") -> MCPToolResult:
        """List directory entries and bound the provider-visible result."""
        result = await self.raw_file_list(path)
        if result.isError:
            return result
        return tool_ok(await self.bound_text(result_text(result)))


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
