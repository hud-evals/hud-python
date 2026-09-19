"""Managed subprocess helpers."""

from __future__ import annotations

import array
import asyncio
import codecs
import contextlib
import errno
import io
import os
import signal
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

if sys.platform != "win32":
    import fcntl
    import termios

if TYPE_CHECKING:
    from collections.abc import AsyncIterable, Callable
    from typing import TextIO

_PROCESS_EXIT_POLL_INTERVAL = 0.05
OUTPUT_DRAIN_TIMEOUT = 1.0


def output_writer(
    output: TextIO,
    *,
    capture: Callable[[str], None] | None = None,
) -> tuple[Callable[[str | bytes], None], Callable[[], None]]:
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    sink_failed = False

    def write(chunk: str | bytes) -> None:
        nonlocal sink_failed
        text = decoder.decode(chunk) if isinstance(chunk, bytes) else chunk
        if capture is not None and text:
            capture(text)
        if sink_failed:
            return
        try:
            if isinstance(chunk, bytes) and isinstance(output, io.TextIOWrapper):
                output.flush()
                output.buffer.write(chunk)
                output.buffer.flush()
            elif text:
                output.write(text)
                output.flush()
        except Exception:
            sink_failed = True

    def finish() -> None:
        nonlocal sink_failed
        text = decoder.decode(b"", final=True)
        if capture is not None and text:
            capture(text)
        if not text or isinstance(output, io.TextIOWrapper) or sink_failed:
            return
        try:
            output.write(text)
            output.flush()
        except Exception:
            sink_failed = True

    return write, finish


async def stream_output(
    source: AsyncIterable[str] | AsyncIterable[bytes],
    output: TextIO,
    *,
    capture: Callable[[str], None] | None = None,
) -> None:
    write, finish = output_writer(output, capture=capture)
    try:
        if isinstance(source, asyncio.StreamReader):
            while chunk := await source.read(65536):
                write(chunk)
        else:
            async for chunk in source:
                write(chunk)
    finally:
        finish()


async def finish_output(*tasks: asyncio.Task[None]) -> None:
    if not tasks:
        return
    _, pending = await asyncio.wait(tasks, timeout=OUTPUT_DRAIN_TIMEOUT)
    for task in pending:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


class _OutputProtocol(asyncio.StreamReaderProtocol):
    def __init__(self, reader: asyncio.StreamReader, *, terminal: bool) -> None:
        super().__init__(reader)
        self.terminal = terminal

    def connection_lost(self, exc: Exception | None) -> None:
        if self.terminal and isinstance(exc, OSError) and exc.errno == errno.EIO:
            exc = None
        super().connection_lost(exc)


class ProcessOutput:
    """Capture a command's pipe, retaining bytes pending when its leader exits."""

    def __init__(self, fd: int) -> None:
        self.file = os.fdopen(fd, "rb", buffering=0)
        self.reader = asyncio.StreamReader()
        self.transport: asyncio.ReadTransport | None = None

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        transport, _ = await loop.connect_read_pipe(
            lambda: _OutputProtocol(self.reader, terminal=self.file.isatty()), self.file
        )
        self.transport = transport

    def finish(self) -> None:
        transport = self.transport
        if transport is None:
            self.file.close()
            return
        if transport.is_closing():
            return
        transport.pause_reading()
        try:
            # Descendants may retain the write end. Capture bytes already in
            # the pipe without waiting for those descendants to exit.
            available = array.array("i", [0])
            fcntl.ioctl(self.file.fileno(), termios.FIONREAD, available, True)
            remaining = available[0]
            while remaining:
                chunk = os.read(self.file.fileno(), min(remaining, 65536))
                if not chunk:
                    break
                self.reader.feed_data(chunk)
                remaining -= len(chunk)
        finally:
            transport.close()


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
