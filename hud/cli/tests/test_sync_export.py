"""``hud sync tasks --export``: the CSV spreadsheet view of task rows."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hud.cli.sync import _write_csv
from hud.eval import Task

if TYPE_CHECKING:
    from pathlib import Path


def test_write_csv_flattens_args(tmp_path: Path) -> None:
    rows = [
        Task(env="e", id="solve", args={"n": 1}, slug="one"),
        Task(env="e", id="solve", args={"n": {"x": 2}}, slug="two"),
    ]
    rows = [row.model_dump() for row in rows]

    out = tmp_path / "tasks.csv"
    _write_csv(out, rows)

    csv_text = out.read_text()
    assert "slug,id,env,arg:n" in csv_text
    assert "one,solve,e,1" in csv_text
    assert 'two,solve,e,"{""x"": 2}"' in csv_text
