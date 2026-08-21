"""Process boundary for JSONL CLI agents."""

from __future__ import annotations

import asyncio
import base64
import contextlib
from typing import TYPE_CHECKING

import asyncssh

if TYPE_CHECKING:
    from collections.abc import Callable

    from hud.capabilities import SSHClient

WINDOWS_SHELLS = ("cmd", "powershell")
PROCESS_CLOSE_TIMEOUT_S = 5.0


def powershell_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def powershell(script: str) -> str:
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    return f"powershell -NoProfile -NonInteractive -EncodedCommand {encoded}"


async def run_jsonl(
    ssh: SSHClient,
    command: str,
    consume: Callable[[str], None],
    *,
    input_text: str | None = None,
) -> tuple[int, str]:
    """Stream one remote JSONL process and own its cancellation cleanup."""
    process = await ssh.create_process(command)
    stderr_task = asyncio.create_task(process.stderr.read())
    try:
        if input_text is not None:
            process.stdin.write(input_text.encode())
            await process.stdin.drain()
            process.stdin.write_eof()
        while line := await process.stdout.readline():
            consume(line.decode(errors="replace"))
        await process.wait_closed()
        stderr = (await stderr_task).decode(errors="replace")
    except BaseException:
        process.close()
        if not stderr_task.done():
            stderr_task.cancel()
        await asyncio.gather(stderr_task, return_exceptions=True)
        with contextlib.suppress(OSError, TimeoutError, asyncssh.Error):
            async with asyncio.timeout(PROCESS_CLOSE_TIMEOUT_S):
                await process.wait_closed()
        raise

    if process.returncode is None:
        raise RuntimeError("CLI process closed without an exit status")
    return process.returncode, stderr
