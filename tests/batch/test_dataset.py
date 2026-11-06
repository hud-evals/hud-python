"""``Dataset``: Parquet rows in storage order, sharded by row group, resumable by cursor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pyarrow.compute as pc
import pytest

from hud.batch import Cursor, Dataset

if TYPE_CHECKING:
    from pathlib import Path


async def collect(dataset: Dataset, **kwargs: Any) -> list[str]:
    return [row["id"] async for row in dataset.iter_rows(**kwargs)]


def test_a_glob_lists_files_in_sorted_order_and_row_groups_follow(documents: Path) -> None:
    dataset = Dataset.file(f"{documents}/*.parquet")

    assert [path.rsplit("/", 1)[-1] for path in dataset.files()] == [
        "part-00000.parquet",
        "part-00001.parquet",
        "part-00002.parquet",
    ]
    # 10 rows in groups of 4 -> 3 groups per file.
    assert dataset.row_groups() == [(f, g) for f in range(3) for g in range(3)]


def test_a_missing_glob_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Dataset.file(f"{tmp_path}/nothing/*.parquet").files()


def test_constructors_build_the_expected_urls() -> None:
    assert Dataset.hf("org/repo", revision="abc123", path="data/*.parquet").uri == (
        "hf://datasets/org/repo@abc123/data/*.parquet"
    )
    assert Dataset.s3("bucket/prefix/*.parquet").uri == "s3://bucket/prefix/*.parquet"
    assert Dataset.gcs("gs://bucket/x.parquet").uri == "gs://bucket/x.parquet"
    with pytest.raises(ValueError, match="expected a s3://"):
        Dataset.s3("gs://bucket/x.parquet")


async def test_rows_come_out_in_storage_order_with_locator_fields(documents: Path) -> None:
    dataset = Dataset.file(f"{documents}/*.parquet", columns=["id"])

    rows = [row async for row in dataset.iter_rows()]

    assert [row["id"] for row in rows] == [f"doc-{f}-{r}" for f in range(3) for r in range(10)]
    first, last = rows[0], rows[-1]
    assert (first["_file_index"], first["_row_group"], first["_row"]) == (0, 0, 0)
    assert (last["_file_index"], last["_row_group"], last["_row"]) == (2, 2, 1)
    assert first["_file"].endswith("part-00000.parquet")
    assert "text" not in first  # only the requested columns are decoded


async def test_shards_partition_row_groups_without_overlap(documents: Path) -> None:
    dataset = Dataset.file(f"{documents}/*.parquet")

    shards = [await collect(dataset, shard=(i, 4)) for i in range(4)]

    assert sorted(id for shard in shards for id in shard) == sorted(await collect(dataset))
    assert all(set(a).isdisjoint(b) for a in shards for b in shards if a is not b)
    # 9 row groups (sizes 4,4,2 per file) over 4 shards: ordinals 0,4,8 / 1,5 / 2,6 / 3,7.
    assert [len(s) for s in shards] == [4 + 4 + 2, 4 + 2, 2 + 4, 4 + 4]
    with pytest.raises(ValueError, match="shard must be"):
        dataset.shard_row_groups((4, 4))


async def test_a_cursor_resumes_strictly_after_the_consumed_row(documents: Path) -> None:
    dataset = Dataset.file(f"{documents}/*.parquet")
    everything = await collect(dataset)

    # Mid row group, and exactly at a row-group boundary.
    assert await collect(dataset, cursor=Cursor(file=1, row_group=0, row=1)) == everything[12:]
    assert await collect(dataset, cursor=Cursor(file=1, row_group=0, row=3)) == everything[14:]
    assert await collect(dataset, cursor=Cursor(file=2, row_group=2, row=1)) == []


async def test_an_arrow_expression_filters_before_decoding_and_a_predicate_after(
    documents: Path,
) -> None:
    dataset = Dataset.file(f"{documents}/*.parquet")

    small = await collect(dataset, where=pc.field("n_chars") <= 300)
    gos = await collect(dataset, where=lambda row: row["language"] == "go")

    assert small == [f"doc-{f}-{r}" for f in range(3) for r in range(3)]
    assert gos == [f"doc-{f}-{r}" for f in range(3) for r in range(1, 10, 2)]


def test_read_row_returns_the_row_at_a_cursor_and_reuses_the_cached_group(
    documents: Path,
) -> None:
    dataset = Dataset.file(f"{documents}/*.parquet")

    row = dataset.read_row(Cursor(file=2, row_group=1, row=3))
    again = dataset.read_row_group(2, 1)

    assert (row["id"], row["_row"]) == ("doc-2-7", 3)
    assert again is dataset.read_row_group(2, 1)
    with pytest.raises(IndexError):
        dataset.read_row(Cursor(file=2, row_group=2, row=5))


def test_a_dataset_column_may_not_shadow_a_locator_field(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.table({"_row": [1], "id": ["x"]}), tmp_path / "bad.parquet")

    with pytest.raises(ValueError, match="collides with a locator field"):
        Dataset.file(str(tmp_path / "bad.parquet")).read_row_group(0, 0)
