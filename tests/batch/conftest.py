"""Parquet fixtures for the batch tests: a few files, a few row groups each."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

if TYPE_CHECKING:
    from pathlib import Path


def write_documents(
    root: Path, *, files: int = 3, rows_per_file: int = 10, row_group_size: int = 4
) -> list[Path]:
    """``files`` Parquet files of ``rows_per_file`` documents each, in ``row_group_size`` groups.

    Document ids are ``doc-<file>-<row>``; ``n_chars`` grows with the row so a
    predicate can select a known subset; ``language`` alternates.
    """
    root.mkdir(parents=True, exist_ok=True)
    written = []
    for file_index in range(files):
        table = pa.table(
            {
                "id": [f"doc-{file_index}-{row}" for row in range(rows_per_file)],
                "n_chars": [100 * (row + 1) for row in range(rows_per_file)],
                "language": [("python", "go")[row % 2] for row in range(rows_per_file)],
                "text": [f"text {file_index}/{row}" for row in range(rows_per_file)],
            }
        )
        path = root / f"part-{file_index:05d}.parquet"
        pq.write_table(table, path, row_group_size=row_group_size)
        written.append(path)
    return written


@pytest.fixture
def documents(tmp_path: Path) -> Path:
    write_documents(tmp_path / "data")
    return tmp_path / "data"
