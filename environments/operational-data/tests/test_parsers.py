"""How an unreadable upload is reported."""

from pathlib import Path

import pytest

from data_sources.base import DataSource, ExportError, TaskContext
from data_sources.sources import SOURCES


@pytest.mark.parametrize("source", list(SOURCES.values()), ids=list(SOURCES))
def test_an_unreadable_upload_raises_export_error(
    source: type[DataSource], tmp_path: Path, context: TaskContext
):
    path = tmp_path / "export.bin"
    path.write_bytes(b"\x00\x01 not an export \xff")
    with pytest.raises(ExportError):
        source.load([path], context)
