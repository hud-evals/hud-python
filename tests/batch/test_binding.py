"""``Binding``: one dataset row -> one task row, declaratively."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from hud import Environment
from hud.batch import Dataset
from hud.batch.binding import Binding, compile_where, resolve_path

if TYPE_CHECKING:
    from pathlib import Path

env = Environment("docs")


@env.template()
async def scan(record_id: str, locator: dict[str, Any], limit: int = 10):
    yield "go"
    yield 1.0


ROW = {
    "id": "doc-7",
    "n_chars": 1234,
    "language": "python",
    "_file": "data/part-00001.parquet",
    "_file_index": 1,
    "_row_group": 3,
    "_row": 42,
}


def test_paths_resolve_into_the_row_and_literals_pass_through() -> None:
    binding = Binding(
        row_id="id",
        args={
            "record_id": "$.id",
            "locator": {"file": "$._file", "row_group": "$._row_group", "row": "$._row"},
            "limit": 5,
        },
        columns={"language": "$.language", "suite": "pii"},
    )

    task = binding.bind("b1", "docs", "scan", ROW)

    assert (task.env, task.id, task.slug) == ("docs", "scan", "b1:doc-7")
    assert task.args == {
        "record_id": "doc-7",
        "locator": {"file": "data/part-00001.parquet", "row_group": 3, "row": 42},
        "limit": 5,
    }
    assert task.columns == {"language": "python", "suite": "pii"}


def test_a_missing_path_or_empty_row_id_is_an_error_not_a_none() -> None:
    with pytest.raises(KeyError, match="no 'missing' in row"):
        resolve_path(ROW, "$.missing")
    with pytest.raises(ValueError, match="row_id 'id' is empty"):
        Binding(row_id="id", args={}).bind("b", "docs", "scan", {**ROW, "id": ""})


def test_required_columns_cover_ids_args_columns_and_where() -> None:
    binding = Binding(
        row_id="id",
        args={"record_id": "$.id", "locator": {"file": "$._file"}},
        columns={"language": "$.language"},
        where="n_chars <= 100000 and license != 'none'",
    )

    assert binding.required_columns() == {"id", "_file", "language", "n_chars", "license"}


def test_validation_against_the_template_signature() -> None:
    Binding(row_id="id", args={"record_id": "$.id", "locator": {}}).validate_template(scan)

    with pytest.raises(ValueError, match="missing required args \\['locator'\\]"):
        Binding(row_id="id", args={"record_id": "$.id"}).validate_template(scan)
    with pytest.raises(ValueError, match="not parameters of template 'scan'"):
        Binding(
            row_id="id", args={"record_id": "$.id", "locator": {}, "extra": 1}
        ).validate_template(scan)
    with pytest.raises(ValueError, match="where: cannot parse"):
        Binding(
            row_id="id", args={"record_id": "$.id", "locator": {}}, where="n_chars ~ 3"
        ).validate_template(scan)


@pytest.mark.parametrize(
    ("where", "expected"),
    [
        ("n_chars <= 300", [f"doc-{f}-{r}" for f in range(3) for r in range(3)]),
        ("language == 'go' and n_chars < 500", [f"doc-{f}-{r}" for f in range(3) for r in (1, 3)]),
        ("n_chars == 100 or n_chars >= 1000", [f"doc-{f}-{r}" for f in range(3) for r in (0, 9)]),
        ('language != "go" and n_chars > 800', [f"doc-{f}-8" for f in range(3)]),
    ],
)
async def test_where_compiles_to_a_pushed_down_filter(
    documents: Path, where: str, expected: list[str]
) -> None:
    dataset = Dataset.file(f"{documents}/*.parquet")
    binding = Binding(row_id="id", args={}, where=where)

    rows = [row["id"] async for row in dataset.iter_rows(where=binding.expression())]

    assert rows == expected


@pytest.mark.parametrize(
    "text",
    ["", "n_chars", "n_chars <= ", "<= 3", "n_chars <= 3 extra", "and == 1", "x == null or"],
)
def test_malformed_where_clauses_are_rejected(text: str) -> None:
    with pytest.raises(ValueError, match="where:"):
        compile_where(text)
