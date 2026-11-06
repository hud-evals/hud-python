"""``RowStream``: bounded prefetch, parallel readers over disjoint sub-shards, checkpoints."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from hud.batch import Cursor, Dataset
from hud.batch.materialize import Materialization, RowStream, cursor_of

if TYPE_CHECKING:
    from pathlib import Path


def dataset_at(documents: Path) -> Dataset:
    return Dataset.file(f"{documents}/*.parquet", columns=["id", "n_chars"])


async def drain(stream: RowStream) -> list[tuple[int, str]]:
    out = []
    async for reader, row in stream:
        out.append((reader, row["id"]))
        stream.done(reader, row)
    return out


def test_reader_shards_interleave_inside_the_process_shard() -> None:
    policy = Materialization(shards=4, shard=1, parallel_readers=3)

    assert [policy.reader_shard(r) for r in range(3)] == [(3, 12), (4, 12), (5, 12)]
    with pytest.raises(ValueError, match="out of range"):
        Materialization(shards=2, shard=2).reader_shard(0)


async def test_readers_together_cover_the_shard_exactly_once(documents: Path) -> None:
    dataset = dataset_at(documents)
    everything = [row["id"] async for row in dataset.iter_rows()]

    split = [
        await drain(RowStream(dataset, Materialization(shards=2, shard=s, parallel_readers=3)))
        for s in range(2)
    ]

    seen = [id for part in split for _reader, id in part]
    assert sorted(seen) == sorted(everything) and len(seen) == len(everything)
    assert {reader for part in split for reader, _id in part} == {0, 1, 2}


async def test_each_reader_delivers_in_storage_order_and_skip_drops_rows(documents: Path) -> None:
    dataset = dataset_at(documents)
    stream = RowStream(
        dataset,
        Materialization(parallel_readers=2),
        skip=lambda row: row["n_chars"] == 500,
    )

    delivered = await drain(stream)

    per_reader = {0: [], 1: []}
    for reader, id in delivered:
        per_reader[reader].append(id)
    for ids in per_reader.values():
        assert ids == sorted(ids, key=lambda id: tuple(int(p) for p in id.split("-")[1:]))
    assert all(not id.endswith("-4") for _r, id in delivered)
    assert len(delivered) == 27


async def test_prefetch_bounds_how_far_readers_run_ahead(documents: Path) -> None:
    dataset = dataset_at(documents)
    stream = RowStream(dataset, Materialization(prefetch_rows=3))

    iterator = stream.__aiter__()
    first = await iterator.__anext__()
    await asyncio.sleep(0.05)

    assert stream._queue.qsize() == 3  # noqa: SLF001 - the bound is the behavior under test
    stream.done(*first)
    async for reader, row in iterator:
        stream.done(reader, row)


async def test_checkpoints_trail_the_earliest_unfinished_row(documents: Path) -> None:
    dataset = dataset_at(documents)
    stream = RowStream(dataset, Materialization())
    iterator = stream.__aiter__()

    rows = [await iterator.__anext__() for _ in range(4)]
    assert stream.checkpoints == {0: None}  # nothing finished yet

    stream.done(*rows[1])
    stream.done(*rows[2])
    assert stream.checkpoints == {0: None}  # row 0 still in flight

    stream.done(*rows[0])
    assert stream.checkpoints == {0: cursor_of(rows[2][1])}  # 0..2 done, 3 in flight

    stream.done(*rows[3])
    assert stream.checkpoints == {0: cursor_of(rows[3][1])}
    async for reader, row in iterator:
        stream.done(reader, row)
    assert stream.checkpoints == {0: Cursor(file=2, row_group=2, row=1)}


async def test_resuming_from_checkpoints_yields_the_rest_without_duplicates(
    documents: Path,
) -> None:
    dataset = dataset_at(documents)
    policy = Materialization(parallel_readers=2)
    first = RowStream(dataset, policy)
    iterator = first.__aiter__()

    done_ids = []
    for _ in range(9):
        reader, row = await iterator.__anext__()
        first.done(reader, row)
        done_ids.append(row["id"])
    await iterator.aclose()
    checkpoints = first.checkpoints

    rest = [id for _r, id in await drain(RowStream(dataset, policy, resume=checkpoints))]

    everything = [row["id"] async for row in dataset.iter_rows()]
    assert sorted(done_ids + rest) == sorted(everything)
    assert not set(done_ids) & set(rest)


async def test_a_failing_reader_fails_the_stream(documents: Path) -> None:
    dataset = dataset_at(documents)

    def boom(row: dict[str, object]) -> bool:
        raise RuntimeError("reader exploded")

    with pytest.raises(RuntimeError, match="reader exploded"):
        await drain(RowStream(dataset, Materialization(parallel_readers=2), skip=boom))


async def test_a_stream_iterates_once(documents: Path) -> None:
    stream = RowStream(dataset_at(documents), Materialization())
    await drain(stream)
    with pytest.raises(RuntimeError, match="iterated once"):
        await drain(stream)
