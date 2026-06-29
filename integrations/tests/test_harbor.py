"""``integrations.harbor`` — load Harbor task dirs as a Taskset; export HUD tasks."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

import pytest

from integrations.harbor import HarborRuntime, detect, export, load
from integrations.harbor_runtime import (
    _compose_file,
    _compose_overlay,
    _dockerfile_declared_generated_app_files,
    _ensure_dockerfile_created_dirs,
    _ensure_start_script,
    _host_path_for_app_file,
    _preserved_image_paths,
)

from .conftest import make_harbor_task

if TYPE_CHECKING:
    from pathlib import Path

# ─── detect / load: Harbor dirs -> Taskset ─────────────────────────────


def test_detect_recognizes_task_and_dataset_dirs(single_task: Path, tmp_path: Path) -> None:
    assert detect(single_task)
    assert detect(single_task.parent)  # dataset dir containing task dirs
    empty = tmp_path / "empty"
    empty.mkdir()
    assert not detect(empty)
    assert not detect(single_task / "task.toml")  # a file is not a task dir


def test_load_single_task_dir_maps_rows(single_task: Path) -> None:
    taskset = load(single_task)

    assert len(taskset) == 1
    row = taskset["cancel-async-tasks"]
    assert row.id == "cancel-async-tasks"
    assert row.args == {}
    assert row.env == taskset.name


def test_load_dataset_shares_one_env_per_build_context(dataset_same_env: Path) -> None:
    taskset = load(dataset_same_env)

    assert len(taskset) == 3
    # Identical Dockerfiles -> all rows reference one env name.
    assert taskset.environment_names() == {"terminal-bench-sample"}


def test_load_dataset_groups_by_distinct_build_contexts(dataset_multi_env: Path) -> None:
    taskset = load(dataset_multi_env)

    assert len(taskset) == 4
    assert taskset.environment_names() == {"mixed-bench-g1", "mixed-bench-g2"}
    assert taskset["build-pmars"].env == taskset["cancel-async-tasks"].env
    assert taskset["caffe-cifar-10"].env == taskset["sam-cell-seg"].env
    assert taskset["build-pmars"].env != taskset["caffe-cifar-10"].env


def test_load_rejects_dirs_without_harbor_tasks(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no Harbor tasks"):
        load(empty)


def test_load_skips_unparseable_toml_but_keeps_the_rest(tmp_path: Path) -> None:
    dataset = tmp_path / "bench"
    dataset.mkdir()
    make_harbor_task(dataset, "good")
    broken = make_harbor_task(dataset, "broken")
    (broken / "task.toml").write_text("not [valid toml", encoding="utf-8")

    taskset = load(dataset)

    # Unparseable config degrades gracefully; the task itself still loads.
    assert {task.id for task in taskset} == {"good", "broken"}


def test_harbor_runtime_accepts_dataset_dirs(single_task: Path) -> None:
    runtime = HarborRuntime(single_task.parent)

    assert single_task.name in runtime._task_dirs


def test_compose_file_detection_prefers_harbor_names(tmp_path: Path) -> None:
    env = tmp_path / "environment"
    env.mkdir()
    compose = env / "docker-compose.yaml"
    compose.write_text("services: {}\n", encoding="utf-8")

    assert _compose_file(env) == compose


def test_compose_overlay_mounts_main_workspace_tests_and_logs(tmp_path: Path) -> None:
    overlay = _compose_overlay(
        workspace=tmp_path / "workspace",
        tests_dir=tmp_path / "tests",
        logs=tmp_path / "logs",
        preserved_paths=[],
    )

    assert "main:" in overlay
    assert 'entrypoint: ["sleep"]' in overlay
    assert f"{tmp_path / 'workspace'}:/app" in overlay
    assert f"{tmp_path / 'tests'}:/tests:ro" in overlay
    assert f"{tmp_path / 'logs'}:/logs" in overlay


def test_compose_overlay_preserves_image_dependency_subpaths(tmp_path: Path) -> None:
    overlay = _compose_overlay(
        workspace=tmp_path / "workspace",
        tests_dir=tmp_path / "tests",
        logs=tmp_path / "logs",
        preserved_paths=["/app/node_modules"],
    )

    assert '      - "/app/node_modules"' in overlay


def test_preserved_image_paths_detects_node_and_php_dependency_dirs(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text("{}", encoding="utf-8")
    (tmp_path / "composer.json").write_text("{}", encoding="utf-8")

    assert _preserved_image_paths(tmp_path) == ["/app/node_modules", "/app/vendor"]


def test_preserved_image_paths_detects_node_build_output(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text("{}", encoding="utf-8")
    (tmp_path / "Dockerfile").write_text(
        "FROM node:20-slim\nRUN npm ci\nRUN npm run build\n",
        encoding="utf-8",
    )

    assert _preserved_image_paths(tmp_path) == ["/app/node_modules", "/app/dist"]


def test_ensure_start_script_recreates_build_generated_entrypoint(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "docker-entrypoint.sh").write_text("echo start\n", encoding="utf-8")

    _ensure_start_script(workspace)

    start = workspace / "start_app.sh"
    assert start.exists()
    text = start.read_text(encoding="utf-8")
    assert "exec sh /app/docker-entrypoint.sh" in text


def test_ensure_start_script_preserves_dockerfile_generated_command(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "docker-entrypoint.sh").write_text('exec "$@"\n', encoding="utf-8")
    (workspace / "Dockerfile").write_text(
        "FROM python:3.11-slim\n"
        "RUN printf '%s\\n' '#!/usr/bin/env bash' 'set -e' 'cd /app' "
        "'exec /app/docker-entrypoint.sh gunicorn --bind 0.0.0.0:8000 src.main:app' "
        "> /app/start_app.sh && chmod +x /app/start_app.sh\n",
        encoding="utf-8",
    )

    _ensure_start_script(workspace)

    text = (workspace / "start_app.sh").read_text(encoding="utf-8")
    assert "exec /app/docker-entrypoint.sh gunicorn --bind 0.0.0.0:8000 src.main:app" in text
    assert (workspace / "docker-entrypoint.sh").stat().st_mode & 0o111


def test_ensure_start_script_restores_generated_entrypoint(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "Dockerfile").write_text(
        "FROM python:3.11-slim\n"
        "RUN printf '#!/bin/sh\\npython -m src.seed --init\\n"
        "exec uvicorn src.main:app --host 0.0.0.0 --port 8000\\n' "
        "> /app/docker-entrypoint.sh && chmod +x /app/docker-entrypoint.sh\n"
        "RUN printf '%s\\n' '#!/usr/bin/env bash' 'set -e' 'cd /app' "
        "'exec /app/docker-entrypoint.sh' > /app/start_app.sh && chmod +x /app/start_app.sh\n",
        encoding="utf-8",
    )

    _ensure_start_script(workspace)

    entrypoint = workspace / "docker-entrypoint.sh"
    assert entrypoint.exists()
    assert entrypoint.stat().st_mode & 0o111
    assert "python -m src.seed --init" in entrypoint.read_text(encoding="utf-8")
    assert "exec /app/docker-entrypoint.sh" in (workspace / "start_app.sh").read_text(
        encoding="utf-8",
    )


def test_ensure_dockerfile_created_dirs_restores_app_dirs(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "Dockerfile").write_text(
        "FROM node:20-slim\n"
        "RUN mkdir -p static/uploads /app/tmp/cache && mkdir -p /var/lib/ignored\n",
        encoding="utf-8",
    )

    _ensure_dockerfile_created_dirs(workspace)

    assert (workspace / "static" / "uploads").is_dir()
    assert (workspace / "tmp" / "cache").is_dir()
    assert not (workspace / "var" / "lib" / "ignored").exists()


def test_dockerfile_declared_generated_app_files_detects_seeded_sqlite_db(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "Dockerfile").write_text(
        "FROM python:3.11-slim\n"
        "ENV DB_PATH=/app/data/salon_workforce.db\n"
        "RUN python -m src.seed --init\n",
        encoding="utf-8",
    )

    assert _dockerfile_declared_generated_app_files(workspace) == [
        "/app/data/salon_workforce.db",
    ]
    assert _host_path_for_app_file(workspace, "/app/data/salon_workforce.db") == (
        workspace / "data" / "salon_workforce.db"
    )


def test_dockerfile_declared_generated_app_files_ignores_non_app_or_non_db_paths(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "Dockerfile").write_text(
        "FROM python:3.11-slim\n"
        "ENV DB_PATH=/var/lib/app.db CACHE_PATH=/app/cache\n"
        "ENV SOME_DATABASE_PATH=/app/data/app.txt\n",
        encoding="utf-8",
    )

    assert _dockerfile_declared_generated_app_files(workspace) == []
    assert _host_path_for_app_file(workspace, "/tmp/app.db") is None


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
RUN pip install hud-python
COPY env.py ./
CMD ["hud", "serve", "env:env"]
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
    assert "hud serve env:env" in boot
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
    created = await export(str(tmp_path / "env.py"), tmp_path / "out", answer_file="/app/out.txt")
    assert "/app/out.txt" in (created[0] / "instruction.md").read_text(encoding="utf-8")
    assert "/app/out.txt" in (created[0] / "tests" / "test.sh").read_text(encoding="utf-8")
