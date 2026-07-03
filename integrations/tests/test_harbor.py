"""``integrations.harbor`` — load Harbor task dirs as a Taskset; export HUD tasks."""

from __future__ import annotations

import asyncio
import json
import os
import textwrap
from typing import TYPE_CHECKING

import pytest

from hud.eval import Task
from integrations.harbor import HarborRuntime, detect, export, load
from integrations.harbor_runtime import (
    _compose_file,
    _compose_overlay,
    _image_workdir,
    _materialize_workspace,
    _read_harbor_reward,
    _verifier_timeout,
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


async def test_harbor_runtime_builds_unique_images_per_acquisition(
    single_task: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[str, ...], bool]] = []

    async def fake_docker(*args: str, check: bool = True) -> tuple[str, str]:
        calls.append((args, check))
        return "", ""

    monkeypatch.setattr("hud.eval.runtime._docker", fake_docker)
    runtime = HarborRuntime(single_task.parent)

    first = await runtime._build_image(single_task / "environment")
    second = await runtime._build_image(single_task / "environment")

    assert first != second
    assert first.startswith("hud-harbor:")
    assert second.startswith("hud-harbor:")
    assert [args[2] for args, _ in calls] == [first, second]


async def test_compose_container_cleans_up_after_failed_up(
    single_task: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compose = single_task / "environment" / "docker-compose.yaml"
    compose.write_text("services:\n  main:\n    build: .\n", encoding="utf-8")
    calls: list[tuple[tuple[str, ...], bool]] = []

    async def fake_docker(*args: str, check: bool = True) -> tuple[str, str]:
        calls.append((args, check))
        if args[-3:] == ("up", "--detach", "--build"):
            raise RuntimeError("compose failed")
        return "", ""

    monkeypatch.setattr("hud.eval.runtime._docker", fake_docker)
    runtime = HarborRuntime(single_task.parent)

    with pytest.raises(RuntimeError, match="compose failed"):
        async with runtime._compose_container(
            Task(env="bench", id=single_task.name),
            compose,
            tmp_path / "workspace",
            "/app",
            single_task / "tests",
            tmp_path / "logs",
        ):
            raise AssertionError("compose acquisition should not yield")

    assert any(
        args[-5:] == ("down", "--volumes", "--remove-orphans", "--rmi", "local") and check is False
        for args, check in calls
    )


async def test_compose_container_cleans_up_when_main_service_is_missing(
    single_task: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compose = single_task / "environment" / "docker-compose.yaml"
    compose.write_text("services:\n  api:\n    build: .\n", encoding="utf-8")
    calls: list[tuple[tuple[str, ...], bool]] = []

    async def fake_docker(*args: str, check: bool = True) -> tuple[str, str]:
        calls.append((args, check))
        return "", ""

    monkeypatch.setattr("hud.eval.runtime._docker", fake_docker)
    runtime = HarborRuntime(single_task.parent)

    with pytest.raises(RuntimeError, match="did not create a main service"):
        async with runtime._compose_container(
            Task(env="bench", id=single_task.name),
            compose,
            tmp_path / "workspace",
            "/app",
            single_task / "tests",
            tmp_path / "logs",
        ):
            raise AssertionError("compose acquisition should not yield")

    assert any(
        args[-5:] == ("down", "--volumes", "--remove-orphans", "--rmi", "local") and check is False
        for args, check in calls
    )


def test_compose_file_detection_prefers_harbor_names(tmp_path: Path) -> None:
    env = tmp_path / "environment"
    env.mkdir()
    compose = env / "docker-compose.yaml"
    compose.write_text("services: {}\n", encoding="utf-8")

    assert _compose_file(env) == compose


def test_compose_overlay_parks_main_and_mounts_workspace_tests_and_logs(tmp_path: Path) -> None:
    overlay = _compose_overlay(
        workspace=tmp_path / "workspace",
        workdir="/srv/app",
        tests_dir=tmp_path / "tests",
        logs=tmp_path / "logs",
    )

    assert "main:" in overlay
    assert 'entrypoint: ["sleep"]' in overlay
    assert 'working_dir: "/srv/app"' in overlay
    assert f"{tmp_path / 'workspace'}:/srv/app" in overlay
    assert f"{tmp_path / 'tests'}:/tests:ro" in overlay
    assert f"{tmp_path / 'logs'}:/logs" in overlay


async def test_image_workdir_reads_config_working_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_docker(*args: str, check: bool = True) -> tuple[str, str]:
        assert args == ("image", "inspect", "--format", "{{.Config.WorkingDir}}", "img")
        return "/srv/app\n", ""

    monkeypatch.setattr("hud.eval.runtime._docker", fake_docker)

    assert await _image_workdir("img") == "/srv/app"


async def test_image_workdir_defaults_to_app_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_docker(*args: str, check: bool = True) -> tuple[str, str]:
        return "\n", ""

    monkeypatch.setattr("hud.eval.runtime._docker", fake_docker)

    assert await _image_workdir("img") == "/app"


async def test_materialize_workspace_copies_image_workdir_and_owns_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    calls: list[tuple[str, ...]] = []

    async def fake_docker(*args: str, check: bool = True) -> tuple[str, str]:
        calls.append(args)
        if args[0] == "create":
            return "tempcid\n", ""
        return "", ""

    monkeypatch.setattr("hud.eval.runtime._docker", fake_docker)

    await _materialize_workspace("img", workspace, "/app")

    # Contents of the image's workdir are copied out into the host workspace.
    assert ("cp", "tempcid:/app/.", str(workspace)) in calls
    # The throwaway container is removed.
    assert any(a[0] == "rm" for a in calls)
    # On POSIX hosts, ownership is handed to the host user via a chown pass.
    if hasattr(os, "getuid"):
        assert any(a[0] == "run" and "chown" in a and a[-1] == "/app" for a in calls)


def test_read_harbor_reward_prefers_reward_and_score_keys(tmp_path: Path) -> None:
    verifier = tmp_path / "verifier"
    verifier.mkdir()
    (verifier / "reward.json").write_text(json.dumps({"reward": 0.5, "total": 5}), "utf-8")

    reward, info = _read_harbor_reward(verifier)

    assert reward == 0.5
    assert info["reward_json"] == {"reward": 0.5, "total": 5}


def test_read_harbor_reward_rejects_dict_without_reward_or_score(tmp_path: Path) -> None:
    verifier = tmp_path / "verifier"
    verifier.mkdir()
    (verifier / "reward.json").write_text(json.dumps({"passed": 3, "total": 5}), "utf-8")

    reward, info = _read_harbor_reward(verifier)

    assert reward is None
    assert info["reward_parse_error"] == "no numeric reward"


def test_verifier_timeout_reads_task_toml(single_task: Path) -> None:
    assert _verifier_timeout(single_task) == 120.0


def test_verifier_timeout_defaults_when_missing_or_invalid(tmp_path: Path) -> None:
    no_verifier = tmp_path / "no-verifier"
    no_verifier.mkdir()
    (no_verifier / "task.toml").write_text('[metadata]\ncategory = "systems"\n', "utf-8")
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "task.toml").write_text("not toml [", "utf-8")

    assert _verifier_timeout(no_verifier) == 600.0
    assert _verifier_timeout(broken) == 600.0
    assert _verifier_timeout(tmp_path / "missing") == 600.0


async def test_grade_reads_reward_after_verifier_completes(
    single_task: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logs = tmp_path / "logs"
    (logs / "verifier").mkdir(parents=True)
    (logs / "verifier" / "reward.txt").write_text("1.0\n", "utf-8")

    class FakeProc:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"verifier out", b""

    async def fake_exec(*args: str, **kwargs: object) -> FakeProc:
        assert args[:2] == ("docker", "exec")
        return FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    runtime = HarborRuntime(single_task.parent)

    result = await runtime._grade("container", "/app", logs, "done", verifier_timeout=120.0)

    assert result["score"] == 1.0
    assert result["info"]["stdout"] == "verifier out"
    assert (logs / "agent_answer.txt").read_text("utf-8") == "done"


async def test_grade_times_out_when_verifier_hangs(
    single_task: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProc:
        killed = False

        async def communicate(self) -> tuple[bytes, bytes]:
            await asyncio.sleep(3600)
            raise AssertionError("unreachable")

        def kill(self) -> None:
            self.killed = True

        async def wait(self) -> int:
            return -9

    proc = FakeProc()

    async def fake_exec(*args: str, **kwargs: object) -> FakeProc:
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    runtime = HarborRuntime(single_task.parent)

    result = await runtime._grade(
        "container", "/app", tmp_path / "logs", None, verifier_timeout=0.05
    )

    assert result["isError"] is True
    assert "timed out" in result["content"]
    assert proc.killed


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
