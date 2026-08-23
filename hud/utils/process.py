"""Managed subprocess helpers."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import AsyncIterable
    from typing import TextIO

_PROCESS_EXIT_POLL_INTERVAL = 0.05


def write_output(output: TextIO, chunk: str | bytes) -> None:
    output.write(chunk.decode("utf-8", "replace") if isinstance(chunk, bytes) else chunk)
    output.flush()


async def stream_output(
    source: AsyncIterable[str] | AsyncIterable[bytes],
    output: TextIO,
) -> None:
    async for chunk in source:
        write_output(output, chunk)


@dataclass(frozen=True, slots=True)
class ProcessResult:
    """Captured outcome of a managed process group."""

    returncode: int | None
    stdout: bytes
    stderr: bytes
    timed_out: bool = False


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

    async def complete(
        self,
        *,
        max_wait: float | None = None,
    ) -> ProcessResult:
        """Capture output and teardown, reporting timeout as process data.

        The deadline follows the process leader rather than pipe EOF: a
        background child may inherit the pipes after the leader has finished.
        """
        stdout = bytearray()
        stderr = bytearray()

        async def read_into(stream: asyncio.StreamReader, output: bytearray) -> None:
            while chunk := await stream.read(65536):
                output.extend(chunk)

        stdout_read = (
            asyncio.create_task(read_into(self.process.stdout, stdout))
            if self.process.stdout is not None
            else None
        )
        stderr_read = (
            asyncio.create_task(read_into(self.process.stderr, stderr))
            if self.process.stderr is not None
            else None
        )
        readers = tuple(reader for reader in (stdout_read, stderr_read) if reader is not None)
        timed_out = False
        try:
            try:
                await asyncio.wait_for(self.wait(), max_wait)
            except TimeoutError:
                timed_out = True
            returncode = self.returncode
        finally:
            try:
                await self.terminate()
            finally:
                if readers:
                    done, pending = await asyncio.wait(
                        readers,
                        timeout=_PROCESS_EXIT_POLL_INTERVAL,
                    )
                    for reader in pending:
                        reader.cancel()
                    if pending:
                        # asyncio exposes no public way to close subprocess pipes
                        # still held by a child which left the managed group.
                        cast("Any", self.process)._transport.close()
                    await asyncio.gather(*pending, return_exceptions=True)
                    for reader in done:
                        reader.result()
        return ProcessResult(
            returncode,
            bytes(stdout),
            bytes(stderr),
            timed_out,
        )

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

    with contextlib.suppress(ProcessLookupError, PermissionError):
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
        except PermissionError:
            return False
        if loop.time() >= deadline:
            return False
        await asyncio.sleep(min(_PROCESS_EXIT_POLL_INTERVAL, deadline - loop.time()))
