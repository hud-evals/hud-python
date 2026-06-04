"""``hud.eval.harbor.export`` — turn a task source into Harbor task folders."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

from hud.eval.harbor import export

if TYPE_CHECKING:
    from pathlib import Path

_ENV_PY = """\
from hud import Environment

env = Environment("demo")


@env.task()
async def solve(n: int = 1):
    yield f"solve {n}"
    yield 1.0


tasks = [solve(n=2)]
"""


def _write_env(tmp_path: Path) -> Path:
    src = tmp_path / "env.py"
    src.write_text(textwrap.dedent(_ENV_PY), encoding="utf-8")
    return src


async def test_export_writes_task_folder(tmp_path: Path) -> None:
    src = _write_env(tmp_path)
    out = tmp_path / "out"

    created = await export(str(src), out)

    assert len(created) == 1
    task_dir = created[0]
    assert (task_dir / "task.toml").exists()
    assert (task_dir / "instruction.md").read_text(encoding="utf-8") == "solve 2"
    test_sh = (task_dir / "tests" / "test.sh").read_text(encoding="utf-8")
    assert "hud client run" in test_sh
    assert "solve" in test_sh


async def test_export_copies_dockerfile_when_present(tmp_path: Path) -> None:
    _write_env(tmp_path)
    (tmp_path / "Dockerfile").write_text("FROM python:3.11\n", encoding="utf-8")
    out = tmp_path / "out"

    created = await export(str(tmp_path), out)

    assert created
    assert (created[0] / "environment" / "Dockerfile").read_text(encoding="utf-8").startswith(
        "FROM python:3.11"
    )
