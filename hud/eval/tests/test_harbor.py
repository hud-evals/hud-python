"""``hud.eval.harbor.export`` — turn a task source into Harbor task folders."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

import pytest

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

_DOCKERFILE = """\
FROM python:3.11-slim
RUN pip install hud-python
COPY env.py ./
CMD ["hud", "dev"]
"""


def _write_env(tmp_path: Path, *, dockerfile: bool = True) -> Path:
    src = tmp_path / "env.py"
    src.write_text(textwrap.dedent(_ENV_PY), encoding="utf-8")
    if dockerfile:
        (tmp_path / "Dockerfile").write_text(_DOCKERFILE, encoding="utf-8")
    return src


async def test_export_writes_task_folder(tmp_path: Path) -> None:
    src = _write_env(tmp_path)
    out = tmp_path / "out"

    created = await export(str(src), out)

    assert len(created) == 1
    task_dir = created[0]
    assert (task_dir / "task.toml").exists()
    assert (task_dir / "instruction.md").exists()
    assert (task_dir / "tests" / "test.sh").exists()
    assert (task_dir / "environment" / "Dockerfile").exists()
    assert (task_dir / "environment" / "hud_entrypoint.sh").exists()


async def test_requires_dockerfile(tmp_path: Path) -> None:
    _write_env(tmp_path, dockerfile=False)
    with pytest.raises(FileNotFoundError, match="Dockerfile"):
        await export(str(tmp_path / "env.py"), tmp_path / "out")


async def test_instruction_has_prompt_and_answer_convention(tmp_path: Path) -> None:
    _write_env(tmp_path)
    created = await export(str(tmp_path / "env.py"), tmp_path / "out")
    instruction = (created[0] / "instruction.md").read_text(encoding="utf-8")
    assert instruction.startswith("solve 2")  # the materialized prompt
    assert "/workspace/answer.txt" in instruction  # the answer convention


async def test_task_toml_is_harbor_native(tmp_path: Path) -> None:
    _write_env(tmp_path)
    created = await export(str(tmp_path / "env.py"), tmp_path / "out")
    toml = (created[0] / "task.toml").read_text(encoding="utf-8")
    assert 'version = "1.0"' in toml
    assert "name = " in toml
    assert "[verifier]" in toml and "[agent]" in toml
    assert "timeout_sec" in toml
    # HUD task/args preserved as metadata for the record.
    assert "hud_task" in toml and "hud_args" in toml


async def test_scripts_drive_hud_task_lifecycle(tmp_path: Path) -> None:
    _write_env(tmp_path)
    created = await export(str(tmp_path / "env.py"), tmp_path / "out")
    boot = (created[0] / "environment" / "hud_entrypoint.sh").read_text(encoding="utf-8")
    test_sh = (created[0] / "tests" / "test.sh").read_text(encoding="utf-8")

    # Boot serves the channel, parks the run via setup, then hands off.
    assert "hud dev env:env" in boot
    assert "hud task start 'solve'" in boot
    assert 'exec "$@"' in boot
    # Verifier grades the parked run and writes the Harbor reward.
    assert "hud task grade 'solve'" in test_sh
    assert "--answer-file" in test_sh
    assert "/logs/verifier/reward.txt" in test_sh


async def test_dockerfile_neutralizes_cmd_and_bakes_boot(tmp_path: Path) -> None:
    _write_env(tmp_path)
    created = await export(str(tmp_path / "env.py"), tmp_path / "out")
    dockerfile = (created[0] / "environment" / "Dockerfile").read_text(encoding="utf-8")
    assert "# [hud original]" in dockerfile  # original CMD neutralized
    assert 'ENTRYPOINT ["/hud_entrypoint.sh"]' in dockerfile
    # The env build context is copied so the image can be rebuilt under Harbor.
    assert (created[0] / "environment" / "env.py").exists()


async def test_custom_answer_file(tmp_path: Path) -> None:
    _write_env(tmp_path)
    created = await export(
        str(tmp_path / "env.py"), tmp_path / "out", answer_file="/app/out.txt"
    )
    assert "/app/out.txt" in (created[0] / "instruction.md").read_text(encoding="utf-8")
    assert "/app/out.txt" in (created[0] / "tests" / "test.sh").read_text(encoding="utf-8")
