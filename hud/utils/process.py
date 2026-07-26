"""Managed subprocess helpers."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from dataclasses import dataclass
from typing import Any

_PROCESS_EXIT_POLL_INTERVAL = 0.05


@dataclass(slots=True)
class ProcessGroup:
    """Subprocess whose descendants share a teardown boundary.

    POSIX processes are spawned in a new session, making ``process.pid`` the
    process-group id. Teardown always targets that group, even if the leader has
    already exited and only background children remain.
    """

    process: asyncio.subprocess.Process
    term_timeout: float = 1.0
    kill_timeout: float | None = 1.0
    settle_time: float = 0.0

    @property
    def stdout(self) -> asyncio.StreamReader | None:
        return self.process.stdout

    @property
    def stderr(self) -> asyncio.StreamReader | None:
        return self.process.stderr

    @property
    def returncode(self) -> int | None:
        return self.process.returncode

    async def wait(self) -> int:
        """Wait for the process leader without requiring inherited pipes to close."""
        returncode = self.process.returncode
        if returncode is not None:
            return returncode

        wait_task = asyncio.create_task(self.process.wait())
        try:
            while True:
                done, _ = await asyncio.wait(
                    (wait_task,),
                    timeout=_PROCESS_EXIT_POLL_INTERVAL,
                )
                if done:
                    return await wait_task
                returncode = self.process.returncode
                if returncode is not None:
                    return returncode
        finally:
            wait_task.cancel()
            await asyncio.gather(wait_task, return_exceptions=True)

    async def communicate(
        self,
        input: bytes | None = None,
        *,
        max_wait: float | None = None,
    ) -> tuple[bytes, bytes]:
        try:
            if max_wait is None:
                result = await self.process.communicate(input=input)
            else:
                result = await asyncio.wait_for(self.process.communicate(input=input), max_wait)
        finally:
            await self.terminate()
        return result

    async def terminate(self) -> None:
        await _terminate_process_group(
            self.process,
            term_timeout=self.term_timeout,
            kill_timeout=self.kill_timeout,
            settle_time=self.settle_time,
        )


async def create_process_group_exec(
    *cmd: str,
    term_timeout: float = 1.0,
    kill_timeout: float | None = 1.0,
    settle_time: float = 0.0,
    **kwargs: Any,
) -> ProcessGroup:
    if hasattr(os, "killpg"):
        kwargs["start_new_session"] = True
    process = await asyncio.create_subprocess_exec(*cmd, **kwargs)
    return ProcessGroup(
        process=process,
        term_timeout=term_timeout,
        kill_timeout=kill_timeout,
        settle_time=settle_time,
    )


async def _terminate_process_group(
    proc: asyncio.subprocess.Process,
    *,
    term_timeout: float,
    kill_timeout: float | None = None,
    settle_time: float = 0.0,
) -> None:
    if not hasattr(os, "killpg"):
        if proc.returncode is not None:
            return
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), term_timeout)
        except TimeoutError:
            proc.kill()
            if kill_timeout is None:
                await proc.wait()
            else:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(proc.wait(), kill_timeout)
        return

    loop = asyncio.get_running_loop()
    term_deadline = loop.time() + term_timeout + settle_time
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        if proc.returncode is None:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(proc.wait(), term_timeout)
        return

    if proc.returncode is None:
        remaining = max(0.0, term_deadline - loop.time())
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(proc.wait(), remaining)

    remaining = max(0.0, term_deadline - loop.time())
    if await _wait_for_process_group_exit(proc.pid, remaining):
        return

    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)

    if proc.returncode is None:
        if kill_timeout is None:
            await proc.wait()
        else:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(proc.wait(), kill_timeout)


async def _wait_for_process_group_exit(process_group: int, max_wait: float) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max_wait
    while True:
        try:
            os.killpg(process_group, 0)
        except ProcessLookupError:
            return True
        if loop.time() >= deadline:
            return False
        await asyncio.sleep(min(_PROCESS_EXIT_POLL_INTERVAL, deadline - loop.time()))
