"""``hud sync tasks --export``: the CSV spreadsheet view of task rows."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hud.cli.sync import _write_csv
from hud.environment import Environment
from hud.eval import task

if TYPE_CHECKING:
    from pathlib import Path


def test_write_csv_flattens_args_and_columns(tmp_path: Path) -> None:
    env = Environment("e")
    rows = [
        task(env, "solve", slug="one", columns={"tier": "easy"}, n=1).to_dict(),
        task(env, "solve", slug="two", columns={"tier": "hard"}, n={"x": 2}).to_dict(),
    ]

    out = tmp_path / "tasks.csv"
    _write_csv(out, rows)

    csv_text = out.read_text()
    assert "slug,task,env,arg:n,col:tier" in csv_text
    assert "one,solve,e,1,easy" in csv_text
    assert 'two,solve,e,"{""x"": 2}",hard' in csv_text
