"""What a managed process group reports when it ends in each of its ways."""

from __future__ import annotations

import asyncio
import shlex
import sys

import pytest

from hud.utils.process import ProcessResult, create_process_group_exec

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups"),
]


async def _run(script: str, max_wait: float | None = None) -> ProcessResult:
    group = await create_process_group_exec(
        "sh",
        "-c",
        script,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    return await group.complete(max_wait=max_wait)


async def test_a_child_outside_the_group_cannot_hold_completion_open() -> None:
    child = shlex.quote(sys.executable)
    result = await asyncio.wait_for(
        _run(f"{child} -c 'import os, time; os.setsid(); time.sleep(30)' & echo retained"),
        10,
    )

    assert result.returncode == 0
    assert result.stdout == b"retained\n"


async def test_a_cancelled_call_leaves_nothing_running() -> None:
    """A cancelled rollout unwinds through here, and the group is this call's
    to release however it exits."""
    group = await create_process_group_exec(
        "sh",
        "-c",
        "sleep 30",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    task = asyncio.create_task(group.complete())
    await asyncio.sleep(0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert group.returncode is not None
