"""``integrations.harbor`` — load Harbor task dirs as a Taskset; export HUD tasks."""

from __future__ import annotations

import asyncio
import json
import textwrap
from typing import TYPE_CHECKING

import pytest

from integrations.harbor import export

if TYPE_CHECKING:
    from pathlib import Path

# ─── export: HUD tasks -> Harbor task folders ───────────────────────────

_ENV_PY = """\
from hud import Environment

env = Environment("demo")


@env.template()
async def solve(n: int = 1):
    yield f"solve {n}"
    yield 1.0


tasks = [solve(n=2)]
"""

_DOCKERFILE = """\
FROM python:3.11-slim
RUN pip install hud
COPY env.py ./
CMD ["hud", "serve", "env:env"]
"""


def _write_env(tmp_path: Path, *, dockerfile: bool = True, args: dict | None = None) -> Path:
    src = tmp_path / "env.py"
    body = textwrap.dedent(_ENV_PY)
    if args is not None:
        body = body.replace(
            "async def solve(n: int = 1):", "async def solve(n: int = 1, prompt: str = ''):"
        ).replace("tasks = [solve(n=2)]", f"tasks = [solve(**{args!r})]")
    src.write_text(body, encoding="utf-8")
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
    assert "hud serve env:env" in boot
    assert "hud task start solve" in boot
    assert "python3 -c" not in boot
    assert 'exec "$@"' in boot
    # Verifier grades the parked run and writes the Harbor reward.
    assert "hud task grade solve" in test_sh
    assert "--answer-file" in test_sh
    assert "/logs/verifier/reward.txt" in test_sh


async def test_scripts_survive_arguments_that_need_quoting(tmp_path: Path) -> None:
    # Task ids and args are data: an apostrophe must not end the shell string.
    _write_env(tmp_path, args={"prompt": "what's next?"})
    created = await export(str(tmp_path / "env.py"), tmp_path / "out")

    for script in ("environment/hud_entrypoint.sh", "tests/test.sh"):
        text = (created[0] / script).read_text(encoding="utf-8")
        proc = await asyncio.create_subprocess_exec(
            "sh", "-n", "-", stdin=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        _out, err = await proc.communicate(text.encode())
        assert proc.returncode == 0, f"{script} is not valid shell: {err.decode()}"


async def test_environment_context_keeps_its_own_json_files(tmp_path: Path) -> None:
    # Only the copied taskset source is dropped; a build context may need its
    # own JSON (package.json and friends).
    _write_env(tmp_path)
    (tmp_path / "package.json").write_text('{"name": "app"}', encoding="utf-8")

    created = await export(str(tmp_path / "env.py"), tmp_path / "out")

    assert (created[0] / "environment" / "package.json").is_file()


async def test_dockerfile_appends_the_harbor_runtime(tmp_path: Path) -> None:
    _write_env(tmp_path)
    created = await export(str(tmp_path / "env.py"), tmp_path / "out")
    dockerfile = (created[0] / "environment" / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY --chmod=0755 hud_entrypoint.sh /hud_entrypoint.sh" in dockerfile
    assert 'ENTRYPOINT ["/hud_entrypoint.sh"]' in dockerfile
    assert dockerfile.rindex('CMD ["sh", "-c", "sleep infinity"]') > dockerfile.index(
        'CMD ["hud", "serve", "env:env"]'
    )
    # The env build context is copied so the image can be rebuilt under Harbor.
    assert (created[0] / "environment" / "env.py").exists()


async def test_custom_answer_file(tmp_path: Path) -> None:
    _write_env(tmp_path)
    created = await export(str(tmp_path / "env.py"), tmp_path / "out", answer_file="/app/out.txt")
    assert "/app/out.txt" in (created[0] / "instruction.md").read_text(encoding="utf-8")
    assert "/app/out.txt" in (created[0] / "tests" / "test.sh").read_text(encoding="utf-8")


async def test_export_fails_loudly_when_setup_cannot_run(tmp_path: Path) -> None:
    # An unstarted task must not reach the verifier as a score of 0: broken
    # infrastructure and a failed agent are different outcomes.
    _write_env(tmp_path)
    created = await export(str(tmp_path / "env.py"), tmp_path / "out")

    boot = (created[0] / "environment" / "hud_entrypoint.sh").read_text(encoding="utf-8")

    assert "|| true" not in boot
    assert "exit 1" in boot


async def test_export_survives_a_restrictive_dockerignore(tmp_path: Path) -> None:
    # Same class as the adaptation layer's: re-including a generated file
    # needs its own rule, or COPY fails and no exported image builds.
    _write_env(tmp_path)
    (tmp_path / ".dockerignore").write_text("*\n", encoding="utf-8")

    created = await export(str(tmp_path / "env.py"), tmp_path / "out")

    ignore = (created[0] / "environment" / ".dockerignore").read_text(encoding="utf-8")
    assert "!hud_entrypoint.sh" in ignore


async def test_export_overrides_multiline_startup_directives(tmp_path: Path) -> None:
    _write_env(tmp_path)
    (tmp_path / "Dockerfile").write_text(
        'FROM python:3.12-slim\nCMD ["python", \\\n    "-m", \\\n    "app"]\n', encoding="utf-8"
    )

    created = await export(str(tmp_path / "env.py"), tmp_path / "out")

    dockerfile = (created[0] / "environment" / "Dockerfile").read_text(encoding="utf-8")
    assert dockerfile.rindex('CMD ["sh", "-c", "sleep infinity"]') > dockerfile.index('"app"]')


async def test_export_preserves_a_non_root_runtime_user(tmp_path: Path) -> None:
    _write_env(tmp_path)
    (tmp_path / "Dockerfile").write_text(
        "FROM python:3.12-slim\nRUN useradd -m app\nUSER app\n", encoding="utf-8"
    )

    created = await export(str(tmp_path / "env.py"), tmp_path / "out")

    dockerfile = (created[0] / "environment" / "Dockerfile").read_text(encoding="utf-8")
    assert dockerfile.count("USER ") == 1
    assert "COPY --chmod=0755" in dockerfile


async def test_export_serves_the_resolved_environment(tmp_path: Path) -> None:
    # The env lives in tasks.py under the name ``bench`` — the container must
    # serve that, not a guessed ``env:env``.
    (tmp_path / "tasks.py").write_text(
        textwrap.dedent(_ENV_PY)
        .replace("env = Environment", "bench = Environment")
        .replace("@env.template", "@bench.template"),
        encoding="utf-8",
    )
    (tmp_path / "Dockerfile").write_text(_DOCKERFILE, encoding="utf-8")

    created = await export(str(tmp_path / "tasks.py"), tmp_path / "out")

    boot = (created[0] / "environment" / "hud_entrypoint.sh").read_text(encoding="utf-8")
    assert "hud serve tasks:bench" in boot


async def test_export_slugs_stay_inside_the_output_directory(tmp_path: Path) -> None:
    _write_env(tmp_path)
    taskset = tmp_path / "tasks.json"
    taskset.write_text(
        json.dumps([{"env": "demo", "id": "solve", "args": {"n": 2}, "slug": "suite/fix"}]),
        encoding="utf-8",
    )

    created = await export(str(taskset), tmp_path / "out")

    assert created[0].name == "suite-fix"
    taskset.write_text(
        json.dumps([{"env": "demo", "id": "solve", "args": {"n": 2}, "slug": ".."}]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="usable directory name"):
        await export(str(taskset), tmp_path / "invalid")


async def test_export_never_scores_a_broken_grader_as_zero(tmp_path: Path) -> None:
    # Startup and grading are the same rule: infrastructure that cannot run is
    # an error, not an agent that failed the task.
    _write_env(tmp_path)
    created = await export(str(tmp_path / "env.py"), tmp_path / "out")

    test_sh = (created[0] / "tests" / "test.sh").read_text(encoding="utf-8")

    assert "echo 0 > /logs/verifier/reward.txt" not in test_sh
    assert "exit 1" in test_sh
