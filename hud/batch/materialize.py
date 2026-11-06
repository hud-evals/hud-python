"""Materialization: how many rows exist at once, and where to pick up again.

:class:`Materialization` is the policy; :class:`RowStream` applies it to a
:class:`~hud.batch.Dataset`: ``parallel_readers`` reader tasks walk disjoint
sub-shards of this process's shard in storage order and fill one queue bounded
by ``prefetch_rows``, so the consumer always has rows ready and never more than
a window of them in memory. The stream also keeps a per-reader *checkpoint*:
the last row delivered before any row still being worked on, so a crash resumes
without skipping an in-flight row (the ledger de-duplicates the few it redoes).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from .dataset import Cursor

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Mapping, Sequence

    from .dataset import Dataset, Row


class Materialization(BaseModel):
    """Lazy row materialization policy.

    Rows are always produced lazily in storage order. ``shards``/``shard``
    split the dataset's row groups across processes; ``parallel_readers``
    splits this process's shard across reader tasks; ``prefetch_rows`` bounds
    the rows decoded ahead of the consumer.
    """

    model_config = ConfigDict(frozen=True)

    prefetch_rows: int = Field(default=2000, ge=1)
    parallel_readers: int = Field(default=1, ge=1)
    shards: int = Field(default=1, ge=1)
    shard: int = Field(default=0, ge=0)

    def reader_shard(self, reader: int) -> tuple[int, int]:
        """The dataset shard reader ``reader`` of this process walks.

        Readers interleave at row-group granularity inside the process shard, so
        the walk is deterministic for a given ``(shards, parallel_readers)``.
        """
        if self.shard >= self.shards:
            raise ValueError(f"shard {self.shard} is out of range for {self.shards} shards")
        if not 0 <= reader < self.parallel_readers:
            raise ValueError(f"reader {reader} is out of range")
        return (self.shard * self.parallel_readers + reader, self.shards * self.parallel_readers)


def cursor_of(row: Row) -> Cursor:
    return Cursor(file=row["_file_index"], row_group=row["_row_group"], row=row["_row"])


class _Reader:
    """One reader's delivered-but-unfinished rows, and the checkpoint behind them."""

    def __init__(self, start: Cursor | None) -> None:
        self.checkpoint = start
        self.last_delivered = start
        self._outstanding: dict[tuple[int, int, int], Cursor] = {}
        self._previous: dict[tuple[int, int, int], Cursor | None] = {}

    def delivered(self, cursor: Cursor) -> None:
        self._previous[cursor.key()] = self.last_delivered
        self._outstanding[cursor.key()] = cursor
        self.last_delivered = cursor

    def done(self, cursor: Cursor) -> None:
        self._outstanding.pop(cursor.key())
        self._previous.pop(cursor.key())
        if self._outstanding:
            earliest = min(self._outstanding)
            self.checkpoint = self._previous[earliest]
        else:
            self.checkpoint = self.last_delivered


_END = object()


class RowStream:
    """Rows of one process shard, prefetched by parallel readers, with resume checkpoints.

    Iterate it for ``(reader, row)`` pairs; call :meth:`done` with the same pair
    once the row's work is complete. :attr:`checkpoints` then names, per reader,
    the cursor a restart may resume from. ``skip`` drops rows (e.g. already in
    the ledger) before they count as delivered.
    """

    def __init__(
        self,
        dataset: Dataset,
        materialization: Materialization,
        *,
        columns: Sequence[str] | None = None,
        where: Any | None = None,
        resume: Mapping[int, Cursor | None] | None = None,
        skip: Callable[[Row], bool] | None = None,
    ) -> None:
        self.dataset = dataset
        self.materialization = materialization
        self.columns = columns
        self.where = where
        self.skip = skip
        self._readers = {
            reader: _Reader((resume or {}).get(reader))
            for reader in range(materialization.parallel_readers)
        }
        self._queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=materialization.prefetch_rows)
        self._tasks: list[asyncio.Task[None]] = []

    @property
    def checkpoints(self) -> dict[int, Cursor | None]:
        return {reader: state.checkpoint for reader, state in self._readers.items()}

    def done(self, reader: int, row: Row) -> None:
        self._readers[reader].done(cursor_of(row))

    async def _read(self, reader: int) -> None:
        state = self._readers[reader]
        rows = self.dataset.iter_rows(
            columns=self.columns,
            where=self.where,
            shard=self.materialization.reader_shard(reader),
            cursor=state.last_delivered,
        )
        try:
            async for row in rows:
                if self.skip is not None and self.skip(row):
                    continue
                await self._queue.put((reader, row))
        finally:
            await self._queue.put(_END)

    async def __aiter__(self) -> AsyncGenerator[tuple[int, Row], None]:
        if self._tasks:
            raise RuntimeError("a RowStream can be iterated once")
        loop = asyncio.get_running_loop()
        self._tasks = [loop.create_task(self._read(reader)) for reader in self._readers]
        try:
            remaining = len(self._tasks)
            while remaining:
                item = await self._queue.get()
                if item is _END:
                    remaining -= 1
                    for task in self._tasks:
                        if task.done():
                            task.result()  # a reader that failed stops the stream
                    continue
                reader, row = item
                self._readers[reader].delivered(cursor_of(row))
                yield reader, row
        finally:
            for task in self._tasks:
                task.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)


__all__ = ["Materialization", "RowStream", "cursor_of"]
