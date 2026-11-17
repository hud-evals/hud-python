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


def _expand_braces(glob: str) -> list[str]:
    """``glob`` with each ``{a,b}`` group expanded, innermost first."""
    group = re.search(r"\{([^{}]*,[^{}]*)\}", glob)
    if group is None:
        return [glob]
    return [
        expanded
        for option in group[1].split(",")
        for expanded in _expand_braces(glob[: group.start()] + option + glob[group.end() :])
    ]


def _find_glob(root: str, glob: str) -> str:
    """The ``find`` test for paths below ``root`` that match ``glob``.

    Braces expand. A glob without ``/`` matches base names at any depth; one with ``/``
    matches the path below ``root``, where ``**/`` stands for zero or more directories and,
    as ``find -path`` does, ``*`` also crosses directories.
    """
    prefix = re.sub(r"([*?\[\\])", r"\\\1", root.rstrip("/")) + "/"
    tests: list[str] = []
    for option in _expand_braces(glob):
        if "/" not in option:
            tests.append(f"-name {shlex.quote(option)}")
            continue
        head, *rest = option.split("**/")
        paths = [prefix + head]
        for part in rest:
            paths = [path + directories + part for path in paths for directories in ("", "*/")]
        tests += [f"-path {shlex.quote(path)}" for path in paths]
    return "\\( " + " -o ".join(tests) + " \\)"


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

    async def file_grep(self, pattern: str, path: str, include: str | None = None) -> ToolResult:
        """Search files under ``path`` for an extended regex; ``include`` filters them by glob.

        The pattern reaches grep through ``PATTERN``, since ``find`` parses ``{}`` and ``;``
        in its own ``-exec`` arguments.
        """
        files = f"find -H {shlex.quote(path)} -type f"
        if include:
            files += f" {_find_glob(path, include)}"
        grep = """sh -c 'grep -HnE -e "$PATTERN" -- "$@"' sh {} +"""
        return await self.bash(f"PATTERN={shlex.quote(pattern)} {files} -exec {grep}")

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
        """Find the paths under ``path`` that match a glob."""
        return await self.bash(f"find -H {shlex.quote(path)} {_find_glob(path, pattern)}")
