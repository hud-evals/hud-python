"""Focused behavior for ``hud sync`` commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import typer

import hud.cli.sync as sync_module
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


def test_sync_env_noninteractive_requires_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(sync_module, "require_api_key", lambda _: None)
    monkeypatch.setattr(sync_module.PlatformClient, "from_settings", lambda: object())
    monkeypatch.setattr(sync_module, "is_interactive", lambda: False)

    with pytest.raises(typer.Exit) as exc_info:
        sync_module.sync_env_command(name=None, directory=str(tmp_path), yes=False)

    assert exc_info.value.exit_code == 2
