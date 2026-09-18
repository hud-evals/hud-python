from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any


class FakeReader:
    def __init__(self, value: str, *, pause_after: int | None = None) -> None:
        self._raw = value.encode()
        self._lines = self._raw.splitlines(keepends=True)
        self._pause_after = pause_after
        self._index = 0
        self.blocked = asyncio.Event()
        self.release = asyncio.Event()

    async def readline(self) -> bytes:
        if self._pause_after == self._index:
            self.blocked.set()
            await self.release.wait()
            self._pause_after = None
        if self._index == len(self._lines):
            return b""
        line = self._lines[self._index]
        self._index += 1
        return line

    async def read(self) -> bytes:
        return self._raw


class FakeWriter:
    def __init__(self) -> None:
        self.data = bytearray()
        self.eof = False

    def write(self, data: bytes) -> None:
        self.data.extend(data)

    async def drain(self) -> None:
        pass

    def write_eof(self) -> None:
        self.eof = True


class FakeProcess:
    def __init__(
        self,
        stdout: str,
        *,
        stderr: str = "",
        exit_status: int | None = 0,
        returncode: int | None = None,
        pause_after: int | None = None,
    ) -> None:
        self.stdin = FakeWriter()
        self.stdout = FakeReader(stdout, pause_after=pause_after)
        self.stderr = FakeReader(stderr)
        self.exit_status = exit_status
        self.returncode = exit_status if returncode is None else returncode
        self.closed = False
        self.terminated = False

    def terminate(self) -> None:
        self.terminated = True

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        pass


def fake_run() -> Any:
    trace = SimpleNamespace(status=None, content="", extra={})
    steps: list[Any] = []
    return SimpleNamespace(trace=trace, record=steps.append, steps=steps)
