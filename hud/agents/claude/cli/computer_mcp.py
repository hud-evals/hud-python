"""MCP server that exposes computer-use over VNC.

Single tool ``computer`` backed by ``ClaudeComputerTool`` / ``RFBTool``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import secrets
import shlex
import sys
from contextlib import asynccontextmanager
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

import asyncssh
import fastmcp
from fastmcp.exceptions import ToolError
from pydantic import TypeAdapter

from hud.agents.claude.tools.computer import ClaudeComputerTool
from hud.agents.tools.base import AgentToolSpec, result_text
from hud.capabilities import Capability
from hud.capabilities.rfb import RFBClient, ScreenshotEncoding, WebPScreenshotEncoding

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping

    from hud.capabilities import SSHClient

_DEFAULT_SCREENSHOT_ENCODING = WebPScreenshotEncoding()
RFB_CAPABILITY_ENV = "HUD_RFB_CAPABILITY"
SCREENSHOT_ENCODING_ENV = "HUD_SCREENSHOT_ENCODING"
_PROCESS_CLOSE_TIMEOUT_S = 5.0
_BRIDGE_READY_TIMEOUT_S = 5.0
_REMOTE_TMP = PurePosixPath("/") / "tmp"

logger = logging.getLogger(__name__)


def create_computer_mcp(
    rfb: RFBClient,
    screenshot_encoding: ScreenshotEncoding = _DEFAULT_SCREENSHOT_ENCODING,
) -> fastmcp.FastMCP:
    """Build a FastMCP server with one ``computer`` tool backed by ``rfb``."""

    mcp = fastmcp.FastMCP("computer-use")
    tool = ClaudeComputerTool(
        spec=AgentToolSpec(api_type="computer", api_name="computer"),
        client=rfb,
        screenshot_encoding=screenshot_encoding,
    )

    @mcp.tool()
    async def computer(
        action: str,
        coordinate: list[int] | None = None,
        text: str | None = None,
        scroll_direction: str | None = None,
        scroll_amount: int | None = None,
        start_coordinate: list[int] | None = None,
        duration: float | None = None,
        repeat: int | None = None,
        region: list[int] | None = None,
    ) -> list[Any]:
        """Control a remote screen — screenshot, click, type, key, scroll, move, drag, wait, zoom.

        Actions: screenshot, left_click, right_click, middle_click, double_click,
        triple_click, mouse_move, move, type, key, scroll, left_click_drag, drag,
        wait, hold_key, cursor_position, zoom, left_mouse_down, left_mouse_up.

        Returns the resulting screenshot image so you can see the screen state.
        """
        arguments = {
            name: value
            for name, value in {
                "action": action,
                "coordinate": coordinate,
                "text": text,
                "scroll_direction": scroll_direction,
                "scroll_amount": scroll_amount,
                "start_coordinate": start_coordinate,
                "duration": duration,
                "repeat": repeat,
                "region": region,
            }.items()
            if value is not None
        }
        result = await tool.execute(arguments)
        if result.isError:
            raise ToolError(result_text(result) or "computer action failed")
        return result.content

    return mcp


async def run_computer_mcp(environ: Mapping[str, str] = os.environ) -> None:
    """Run computer-use over stdio in a controller-side child process."""
    capability = Capability.from_manifest(json.loads(environ[RFB_CAPABILITY_ENV]))
    screenshot_encoding = TypeAdapter(ScreenshotEncoding).validate_json(
        environ[SCREENSHOT_ENCODING_ENV]
    )

    rfb = await RFBClient.connect(capability)
    try:
        await create_computer_mcp(rfb, screenshot_encoding).run_async(
            transport="stdio",
            show_banner=False,
        )
    finally:
        await rfb.close()


@asynccontextmanager
async def bridge_computer_mcp(
    ssh: SSHClient,
    capability: Capability,
    screenshot_encoding: ScreenshotEncoding = _DEFAULT_SCREENSHOT_ENCODING,
    *,
    shell: str,
) -> AsyncIterator[dict[str, Any]]:
    """Bridge a controller-side computer MCP process into a remote POSIX shell."""
    if shell in {"cmd", "powershell"}:
        raise RuntimeError("ClaudeCLIAgent computer use requires a POSIX workspace")

    token = secrets.token_hex(16)
    request_path = str(_REMOTE_TMP / f"hud-computer-{token}.request")
    response_path = str(_REMOTE_TMP / f"hud-computer-{token}.response")
    bridge = await ssh.create_process(_bridge_command(request_path, response_path))
    local: asyncio.subprocess.Process | None = None
    tasks: list[asyncio.Task[None]] = []
    try:
        ready = await asyncio.wait_for(bridge.stderr.readline(), _BRIDGE_READY_TIMEOUT_S)
        if ready != b"ready\n":
            detail = ready.decode("utf-8", "replace").strip()
            raise RuntimeError(detail or "computer MCP SSH bridge did not become ready")

        environ = {
            **os.environ,
            RFB_CAPABILITY_ENV: json.dumps(capability.to_manifest(), separators=(",", ":")),
            SCREENSHOT_ENCODING_ENV: screenshot_encoding.model_dump_json(),
        }
        local = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "hud.agents.claude.cli.computer_mcp",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=environ,
        )
        assert local.stdin is not None
        assert local.stdout is not None
        assert local.stderr is not None
        tasks = [
            asyncio.create_task(_copy_stream(bridge.stdout, local.stdin)),
            asyncio.create_task(_copy_stream(local.stdout, bridge.stdin)),
            asyncio.create_task(_log_stream(bridge.stderr, "SSH bridge")),
            asyncio.create_task(_log_stream(local.stderr, "computer MCP")),
        ]
        yield _relay_config(request_path, response_path)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        bridge.stdin.close()
        bridge.channel.close()
        with contextlib.suppress(OSError, TimeoutError, asyncssh.Error):
            await asyncio.wait_for(bridge.wait_closed(), _PROCESS_CLOSE_TIMEOUT_S)
        if local is not None:
            if local.stdin is not None:
                local.stdin.close()
            if local.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    local.terminate()
            try:
                await asyncio.wait_for(local.wait(), _PROCESS_CLOSE_TIMEOUT_S)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    local.kill()
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(local.wait(), _PROCESS_CLOSE_TIMEOUT_S)


def _bridge_command(request_path: str, response_path: str) -> str:
    request = shlex.quote(request_path)
    response = shlex.quote(response_path)
    cleanup = shlex.quote(f"rm -f -- {request} {response}")
    return (
        "set -eu; umask 077; "
        f"rm -f -- {request} {response}; mkfifo -- {request} {response}; "
        f"trap {cleanup} EXIT HUP INT TERM; "
        "printf 'ready\\n' >&2; "
        f"cat {request} & reader=$!; cat > {response}; wait $reader"
    )


def _relay_config(request_path: str, response_path: str) -> dict[str, Any]:
    request = shlex.quote(request_path)
    response = shlex.quote(response_path)
    script = f"cat {response} & reader=$!; cat > {request}; wait $reader"
    return {"type": "stdio", "command": "sh", "args": ["-c", script]}


async def _copy_stream(
    reader: asyncio.StreamReader | asyncssh.SSHReader[bytes],
    writer: asyncio.StreamWriter | asyncssh.SSHWriter[bytes],
) -> None:
    try:
        while chunk := await reader.read(65536):
            writer.write(chunk)
            await writer.drain()
    finally:
        writer.close()


async def _log_stream(
    reader: asyncio.StreamReader | asyncssh.SSHReader[bytes],
    source: str,
) -> None:
    while line := await reader.readline():
        logger.warning("%s: %s", source, line.decode("utf-8", "replace").rstrip())


if __name__ == "__main__":
    asyncio.run(run_computer_mcp())


__all__ = [
    "RFB_CAPABILITY_ENV",
    "SCREENSHOT_ENCODING_ENV",
    "bridge_computer_mcp",
    "create_computer_mcp",
    "run_computer_mcp",
]
