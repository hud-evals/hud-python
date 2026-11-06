"""Parquet datasets read lazily, in storage order, from HF / S3 / GCS / local disk.

A :class:`Dataset` is a frozen reference (an fsspec url plus the columns to
read). Rows come out of :meth:`Dataset.iter_rows` one Parquet row group at a
time, in file order, so one range read serves every row of a row group and a
500M-row dataset never has to be listed, downloaded, or held in memory. A
:class:`Cursor` names a position in that order; sharding assigns whole row
groups to shards so shards never overlap and resume stays per-shard.
"""

from __future__ import annotations

import asyncio
import fnmatch
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

try:
    import fsspec
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "hud.batch needs the batch extra. Install with: pip install 'hud[batch]'"
    ) from exc

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Sequence

#: One dataset row: its columns plus the ``_file`` / ``_file_index`` /
#: ``_row_group`` / ``_row`` locator fields a binding can read.
Row = dict[str, Any]

#: Column names the reader stamps on every row; a dataset column may not use them.
LOCATOR_FIELDS = ("_file", "_file_index", "_row_group", "_row")

_GLOB_CHARS = frozenset("*?[")


class Cursor(BaseModel):
    """A position in a dataset's storage order: the last row already consumed."""

    model_config = ConfigDict(frozen=True)

    file: int = Field(ge=0)
    row_group: int = Field(ge=0)
    row: int = Field(ge=0)

    def key(self) -> tuple[int, int, int]:
        return (self.file, self.row_group, self.row)


class _RowGroupCache:
    """A small LRU of decoded row groups with a TTL, for repeated reads of the same group."""

    def __init__(self, maxsize: int, ttl_s: float) -> None:
        self.maxsize = maxsize
        self.ttl_s = ttl_s
        self._entries: OrderedDict[tuple[int, int, tuple[str, ...] | None], tuple[float, Any]] = (
            OrderedDict()
        )

    def get(self, key: tuple[int, int, tuple[str, ...] | None]) -> Any | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        stored_at, table = entry
        if time.monotonic() - stored_at > self.ttl_s:
            del self._entries[key]
            return None
        self._entries.move_to_end(key)
        return table

    def put(self, key: tuple[int, int, tuple[str, ...] | None], table: Any) -> None:
        if self.maxsize == 0:
            return
        self._entries[key] = (time.monotonic(), table)
        self._entries.move_to_end(key)
        while len(self._entries) > self.maxsize:
            self._entries.popitem(last=False)


class Dataset(BaseModel):
    """A reference to Parquet files behind an fsspec url.

    ``uri`` may contain a glob (``hf://datasets/org/repo@rev/data/*.parquet``);
    files are read in sorted path order. ``columns`` restricts what is decoded
    per row group (unset reads every column). ``storage_options`` are passed to
    the filesystem (credentials come from the environment by default: ``HF_TOKEN``,
    AWS / GCP credential chains).
    """

    model_config = ConfigDict(frozen=True)

    uri: str = Field(min_length=1)
    columns: tuple[str, ...] | None = None
    storage_options: dict[str, Any] = Field(default_factory=dict, exclude=True)
    cache_row_groups: int = Field(default=2, ge=0)
    cache_ttl_s: float = Field(default=600.0, gt=0)

    _files: list[str] | None = PrivateAttr(default=None)
    _parquet: dict[int, Any] = PrivateAttr(default_factory=dict)
    _cache: _RowGroupCache | None = PrivateAttr(default=None)

    # ─── constructors ────────────────────────────────────────────────────

    @classmethod
    def hf(
        cls,
        repo: str,
        *,
        revision: str = "main",
        path: str = "**/*.parquet",
        columns: Sequence[str] | None = None,
        **options: Any,
    ) -> Dataset:
        """A Hugging Face dataset repo at a pinned ``revision`` (branch, tag, or sha)."""
        return cls(
            uri=f"hf://datasets/{repo}@{revision}/{path.lstrip('/')}",
            columns=tuple(columns) if columns is not None else None,
            **options,
        )

    @classmethod
    def s3(cls, uri: str, *, columns: Sequence[str] | None = None, **options: Any) -> Dataset:
        return cls._at("s3://", uri, columns, options)

    @classmethod
    def gcs(cls, uri: str, *, columns: Sequence[str] | None = None, **options: Any) -> Dataset:
        return cls._at("gs://", uri, columns, options)

    @classmethod
    def file(cls, path: str, *, columns: Sequence[str] | None = None, **options: Any) -> Dataset:
        return cls._at("file://", path, columns, options)

    @classmethod
    def _at(
        cls,
        scheme: str,
        uri: str,
        columns: Sequence[str] | None,
        options: dict[str, Any],
    ) -> Dataset:
        if "://" not in uri:
            uri = scheme + uri
        if not uri.startswith(scheme):
            raise ValueError(f"expected a {scheme} url, got {uri!r}")
        return cls(uri=uri, columns=tuple(columns) if columns is not None else None, **options)

    # ─── layout ──────────────────────────────────────────────────────────

    def files(self) -> list[str]:
        """The Parquet files this dataset reads, in storage (sorted path) order."""
        if self._files is None:
            fs, path = fsspec.core.url_to_fs(self.uri, **self.storage_options)
            if _GLOB_CHARS & set(path):
                found = sorted(
                    entry
                    for entry in fs.glob(path)
                    if fnmatch.fnmatch(entry.rsplit("/", 1)[-1], path.rsplit("/", 1)[-1])
                )
            else:
                found = [path]
            if not found:
                raise FileNotFoundError(f"no files match {self.uri}")
            self._files = found
        return list(self._files)

    def _parquet_file(self, file_index: int) -> Any:
        parquet = self._parquet.get(file_index)
        if parquet is None:
            fs, _ = fsspec.core.url_to_fs(self.uri, **self.storage_options)
            parquet = pq.ParquetFile(fs.open(self.files()[file_index], "rb"))
            self._parquet[file_index] = parquet
        return parquet

    def row_groups(self) -> list[tuple[int, int]]:
        """Every ``(file_index, row_group)`` in storage order (reads each file's footer)."""
        return [
            (file_index, group)
            for file_index in range(len(self.files()))
            for group in range(self._parquet_file(file_index).num_row_groups)
        ]

    def shard_row_groups(self, shard: tuple[int, int] | None) -> list[tuple[int, int]]:
        """The row groups of ``shard = (index, count)``: ordinal ``j`` belongs to ``j % count``."""
        groups = self.row_groups()
        if shard is None:
            return groups
        index, count = shard
        if count < 1 or not 0 <= index < count:
            raise ValueError(f"shard must be (index, count) with 0 <= index < count, got {shard}")
        return groups[index::count]

    # ─── reads ───────────────────────────────────────────────────────────

    def read_row_group(
        self, file_index: int, row_group: int, columns: Sequence[str] | None = None
    ) -> Any:
        """One decoded row group as a ``pyarrow.Table`` with a ``_row`` ordinal column."""
        wanted = tuple(columns) if columns is not None else self.columns
        key = (file_index, row_group, wanted)
        if self._cache is None:
            self._cache = _RowGroupCache(self.cache_row_groups, self.cache_ttl_s)
        table = self._cache.get(key)
        if table is None:
            table = self._parquet_file(file_index).read_row_group(
                row_group, columns=list(wanted) if wanted is not None else None
            )
            for name in LOCATOR_FIELDS:
                if name in table.column_names:
                    raise ValueError(f"dataset column {name!r} collides with a locator field")
            table = table.append_column("_row", pa.array(range(table.num_rows), pa.int64()))
            self._cache.put(key, table)
        return table

    def read_row(self, cursor: Cursor, columns: Sequence[str] | None = None) -> Row:
        """The single row at ``cursor``."""
        table = self.read_row_group(cursor.file, cursor.row_group, columns)
        if cursor.row >= table.num_rows:
            raise IndexError(f"{cursor} is past the end of its row group ({table.num_rows} rows)")
        return self._rows(table.slice(cursor.row, 1), cursor.file, cursor.row_group)[0]

    def _rows(self, table: Any, file_index: int, row_group: int) -> list[Row]:
        path = self.files()[file_index]
        rows = table.to_pylist()
        for row in rows:
            row["_file"] = path
            row["_file_index"] = file_index
            row["_row_group"] = row_group
        return rows

    async def iter_rows(
        self,
        *,
        columns: Sequence[str] | None = None,
        where: Any | Callable[[Row], bool] | None = None,
        shard: tuple[int, int] | None = None,
        cursor: Cursor | None = None,
    ) -> AsyncIterator[Row]:
        """Rows in storage order, one row group at a time, strictly after ``cursor``.

        ``where`` is a ``pyarrow.compute.Expression`` (applied per row group,
        before decoding) or a Python predicate on the row. ``shard`` restricts
        the walk to that shard's row groups. Decoding runs in a worker thread so
        a slow object store never blocks the event loop.
        """
        expression = where if isinstance(where, pc.Expression) else None
        predicate = where if callable(where) and expression is None else None
        after = cursor.key() if cursor is not None else None
        for file_index, group in self.shard_row_groups(shard):
            if after is not None and (file_index, group) < after[:2]:
                continue
            table = await asyncio.to_thread(self.read_row_group, file_index, group, columns)
            if after is not None and (file_index, group) == after[:2]:
                table = table.slice(after[2] + 1)
            if expression is not None:
                table = table.filter(expression)
            for row in self._rows(table, file_index, group):
                if predicate is None or predicate(row):
                    yield row


__all__ = ["LOCATOR_FIELDS", "Cursor", "Dataset", "Row"]
