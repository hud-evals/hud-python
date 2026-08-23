"""What a managed process group reports when it ends in each of its ways."""

from __future__ import annotations

import asyncio
import os
import shlex
import signal
import sys
from io import StringIO
from types import SimpleNamespace
from typing import cast

import pytest

from hud.utils.process import ProcessGroup, ProcessResult, create_process_group_exec, stream_output

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


async def test_a_finished_process_is_not_a_timeout_because_a_child_holds_its_pipes() -> None:
    """Starting a service and exiting is how a verifier ends: it has already
    written its verdict. Waiting on the inherited pipes instead of on the
    process reports that as a timeout, and scores a passing task zero."""
    result = await _run("echo verdict-written; sleep 30 & exit 0", max_wait=10)

    assert result.timed_out is False
    assert result.returncode == 0
    assert b"verdict-written" in result.stdout


async def test_a_process_that_overran_keeps_what_it_printed() -> None:
    """The output is the evidence of how far it got; reporting only that the
    deadline passed throws away the one thing that explains the timeout."""
    result = await _run("echo progress-so-far; sleep 30", max_wait=1)

    assert result.timed_out is True
    assert b"progress-so-far" in result.stdout


async def test_a_chatty_process_does_not_block_on_a_full_pipe() -> None:
    """Reading only after exit deadlocks once the pipe buffer fills (~64KB)."""
    result = await asyncio.wait_for(_run("yes hud | head -c 500000; exit 0", max_wait=30), 30)

    assert result.timed_out is False
    assert len(result.stdout) == 500000


async def test_stream_output_drains_lines_larger_than_the_reader_limit() -> None:
    source = asyncio.StreamReader()
    source.feed_data(b"x" * 100_000)
    source.feed_eof()
    output = StringIO()

    await stream_output(source, output)

    assert output.getvalue() == "x" * 100_000


async def test_stream_output_preserves_utf8_split_across_chunks() -> None:
    async def chunks():
        yield b"price: \xe2"
        yield b"\x82\xac10"

    output = StringIO()

    await stream_output(chunks(), output)

    assert output.getvalue() == "price: €10"


async def test_a_child_outside_the_group_cannot_hold_completion_open() -> None:
    child = shlex.quote(sys.executable)
    result = await asyncio.wait_for(
        _run(f"{child} -c 'import os, time; os.setsid(); time.sleep(2)' & echo retained"),
        1,
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


async def test_teardown_survives_a_process_group_becoming_unsignalable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals: list[int] = []

    def killpg(_process_group: int, sent: int) -> None:
        signals.append(sent)
        if sent in (0, signal.SIGKILL):
            raise PermissionError

    monkeypatch.setattr(os, "killpg", killpg)
    process = cast(
        "asyncio.subprocess.Process",
        SimpleNamespace(pid=1234, returncode=0),
    )

    await ProcessGroup(process, term_timeout=0).terminate()

    assert signals == [signal.SIGTERM, 0, signal.SIGKILL]
