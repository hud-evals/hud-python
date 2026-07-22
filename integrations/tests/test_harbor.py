"""``integrations.harbor`` — load Harbor task dirs as a Taskset; export HUD tasks."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import textwrap
import tomllib
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from hud.cli import app
from hud.environment.workspace import Workspace
from hud.eval import Task
from integrations.harbor import HarborRuntime, detect, export, load
from integrations.harbor_runtime import (
    _compose_base,
    _compose_file,
    _compose_overlay,
    _container_workdir,
    _DockerWorkspace,
    _EnvironmentConfig,
    _load_runtime_config,
    _prepare_verifier_staging,
    _read_harbor_reward,
    _shielded_cleanup,
    _should_upload_environment_dir,
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
    assert row.columns == {
        "category": "systems",
        "difficulty": "medium",
        "tags": ["bash", "linux"],
    }


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


async def test_shielded_cleanup_finishes_before_propagating_cancellation() -> None:
    started = asyncio.Event()
    finished = asyncio.Event()

    async def cleanup() -> None:
        started.set()
        await asyncio.sleep(0.01)
        finished.set()

    task = asyncio.create_task(_shielded_cleanup(cleanup()))
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished.is_set()


def test_runtime_config_rejects_invalid_toml(single_task: Path) -> None:
    (single_task / "task.toml").write_text("not [valid toml", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid Harbor task config"):
        _load_runtime_config(single_task)


def test_runtime_config_rejects_unknown_fields(single_task: Path) -> None:
    (single_task / "task.toml").write_text(
        "[environment]\nunknown_runtime_knob = true\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="invalid Harbor task config"):
        _load_runtime_config(single_task)


@pytest.mark.parametrize(
    ("task_toml", "feature"),
    [
        ('[environment]\nos = "windows"\n', "Windows containers"),
        ("steps = [{}]\n", "multi-step tasks"),
        ('[verifier]\nenvironment_mode = "separate"\n', "separate verifier environment"),
        ('[environment]\nnetwork_mode = "no-network"\n', "restricted environment network policy"),
    ],
)
def test_runtime_config_fails_closed_for_unsupported_features(
    tmp_path: Path,
    task_toml: str,
    feature: str,
) -> None:
    task_dir = make_harbor_task(tmp_path, "unsupported", task_toml=task_toml)

    with pytest.raises(ValueError, match=feature):
        _load_runtime_config(task_dir)


def test_runtime_config_accepts_legacy_sizes_and_uppercase_linux(tmp_path: Path) -> None:
    task_dir = make_harbor_task(
        tmp_path,
        "legacy-runtime",
        task_toml=textwrap.dedent("""\
            version = "1.3"

            [environment]
            os = "LINUX"
            memory = "4G"
            storage = "10240M"
        """),
    )

    config = _load_runtime_config(task_dir)

    assert config.schema_version == "1.3"
    assert config.environment.os == "linux"
    assert config.environment.memory_mb == 4096
    assert config.environment.storage_mb == 10240


def test_runtime_config_resolves_environment_templates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HARBOR_REQUIRED", "from-host")
    task_dir = make_harbor_task(
        tmp_path,
        "environment-templates",
        task_toml=textwrap.dedent("""\
            [environment.env]
            REQUIRED = "${HARBOR_REQUIRED}"
            DEFAULTED = "${HARBOR_UNSET:-fallback}"
            LITERAL = "plain"

            [verifier.env]
            VERIFIER_DEFAULT = "${HARBOR_VERIFIER_UNSET:-verifier-fallback}"
        """),
    )

    config = _load_runtime_config(task_dir)

    assert config.environment.env == {
        "REQUIRED": "from-host",
        "DEFAULTED": "fallback",
        "LITERAL": "plain",
    }
    assert config.verifier.env == {"VERIFIER_DEFAULT": "verifier-fallback"}


async def test_compose_container_isolates_task_env_and_cleans_up_after_failed_start(
    single_task: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "secret-value-never-in-docker-argv"
    (single_task / "task.toml").write_text(
        textwrap.dedent("""\
            schema_version = "1.3"

            [environment]
            build_timeout_sec = 12
            cpus = 3
            memory_mb = 2048
            workdir = "/srv/app"

            [environment.env]
            HARBOR_SENTINEL = "visible"
            DOCKER_HOST = "task-controlled-docker-host"
            MAIN_IMAGE_NAME = "task-must-not-override-infrastructure"
        """),
        encoding="utf-8",
    )
    with (single_task / "task.toml").open("a", encoding="utf-8") as config_file:
        config_file.write(f'SECRET = "{secret}"\n')
    compose = single_task / "environment" / "docker-compose.yaml"
    compose.write_text("services:\n  main:\n    build: .\n", encoding="utf-8")
    calls: list[tuple[tuple[str, ...], bool, dict[str, str] | None, tuple[str, ...]]] = []

    async def fake_docker(
        *args: str,
        check: bool = True,
        env: dict[str, str] | None = None,
        unset_env: tuple[str, ...] = (),
    ) -> tuple[str, str]:
        calls.append((args, check, env, unset_env))
        if args[-3:] == ("up", "--detach", "--wait"):
            raise RuntimeError("compose failed")
        return "", ""

    monkeypatch.setattr("hud.eval.runtime._docker", fake_docker)
    runtime = HarborRuntime(single_task.parent)
    config = _load_runtime_config(single_task)
    staged_tests = tmp_path / "staged-tests"
    staged_tests.mkdir()

    with pytest.raises(RuntimeError, match="compose failed"):
        async with runtime._compose_container(
            Task(env="bench", id=single_task.name),
            single_task / "environment",
            compose,
            staged_tests,
            tmp_path / "logs",
            config,
        ):
            raise AssertionError("compose acquisition should not yield")

    up_args, _, up_env, up_unset_env = next(call for call in calls if "up" in call[0])
    assert up_args[up_args.index("--project-directory") + 1] == str(single_task / "environment")
    assert up_args.count("-f") == 3
    assert up_env is None
    assert {
        "HARBOR_SENTINEL",
        "DOCKER_HOST",
        "SECRET",
        "CONTEXT_DIR",
        "MAIN_IMAGE_NAME",
        "HOST_AGENT_LOGS_PATH",
        "ENV_AGENT_LOGS_PATH",
        "HOST_VERIFIER_LOGS_PATH",
        "ENV_VERIFIER_LOGS_PATH",
        "HOST_ARTIFACTS_LOGS_PATH",
        "ENV_ARTIFACTS_LOGS_PATH",
        "CPUS",
        "MEMORY",
    } <= set(up_unset_env)
    assert secret not in " ".join(up_args)

    compose_env_path = tmp_path / "compose.env"
    compose_env = compose_env_path.read_text("utf-8")
    assert f"SECRET='{secret}'" in compose_env
    assert "DOCKER_HOST='task-controlled-docker-host'" in compose_env
    assert f"CONTEXT_DIR='{(single_task / 'environment').resolve()}'" in compose_env
    assert "MAIN_IMAGE_NAME='hud-harbor:" in compose_env
    assert "task-must-not-override-infrastructure" not in compose_env
    assert f"HOST_AGENT_LOGS_PATH='{(tmp_path / 'logs' / 'agent').resolve()}'" in compose_env
    assert "ENV_AGENT_LOGS_PATH='/logs/agent'" in compose_env
    assert f"HOST_VERIFIER_LOGS_PATH='{(tmp_path / 'logs' / 'verifier').resolve()}'" in compose_env
    assert "ENV_VERIFIER_LOGS_PATH='/logs/verifier'" in compose_env
    artifacts_logs = (tmp_path / "logs" / "artifacts").resolve()
    assert f"HOST_ARTIFACTS_LOGS_PATH='{artifacts_logs}'" in compose_env
    assert "ENV_ARTIFACTS_LOGS_PATH='/logs/artifacts'" in compose_env
    assert "CPUS='3'" in compose_env
    assert "MEMORY='2048M'" in compose_env
    assert compose_env_path.stat().st_mode & 0o777 == 0o600

    base = json.loads((tmp_path / "compose.base.json").read_text("utf-8"))["services"]["main"]
    assert base["build"] == {"context": "${CONTEXT_DIR}"}
    assert base["image"] == "${MAIN_IMAGE_NAME}"
    overlay = json.loads((tmp_path / "compose.hud.json").read_text("utf-8"))["services"]["main"]
    # ``environment.workdir`` controls agent/verifier exec cwd only; Compose
    # must retain the task-authored main service working_dir.
    assert "working_dir" not in overlay
    assert overlay["cpus"] == 3.0
    assert overlay["mem_limit"] == "2048m"
    assert overlay["environment"] == {
        "HARBOR_SENTINEL": "visible",
        "DOCKER_HOST": "task-controlled-docker-host",
        "MAIN_IMAGE_NAME": "task-must-not-override-infrastructure",
        "SECRET": secret,
    }
    assert (tmp_path / "compose.hud.json").stat().st_mode & 0o777 == 0o600
    assert any(
        args[-5:] == ("down", "--volumes", "--remove-orphans", "--rmi", "local") and check is False
        for args, check, _, _ in calls
    )


async def test_plain_dockerfile_uses_compose_wait_readiness_and_cleanup(
    single_task: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[str, ...], bool, tuple[str, ...]]] = []

    async def fake_docker(
        *args: str,
        check: bool = True,
        env: dict[str, str] | None = None,
        unset_env: tuple[str, ...] = (),
    ) -> tuple[str, str]:
        assert env is None
        calls.append((args, check, unset_env))
        if args[-3:] == ("ps", "-q", "main"):
            return "container-id\n", ""
        if args[:3] == ("inspect", "--format", "{{.Config.WorkingDir}}"):
            return "/workspace\n", ""
        return "", ""

    monkeypatch.setattr("hud.eval.runtime._docker", fake_docker)
    runtime = HarborRuntime(single_task.parent)
    config = _load_runtime_config(single_task)
    staged_tests = tmp_path / "staged-tests"
    staged_tests.mkdir()

    async with runtime._compose_container(
        Task(env="bench", id=single_task.name),
        single_task / "environment",
        None,
        staged_tests,
        tmp_path / "logs",
        config,
    ) as (container, provider, workdir):
        assert (container, provider, workdir) == (
            "container-id",
            "harbor-compose",
            "/workspace",
        )

    command_tails = [args[-3:] for args, _, _ in calls]
    assert any(args[-1:] == ("build",) for args, _, _ in calls)
    assert ("up", "--detach", "--wait") in command_tails
    up_args = next(args for args, _, _ in calls if args[-3:] == ("up", "--detach", "--wait"))
    assert up_args.count("-f") == 2
    assert any(
        args[-5:] == ("down", "--volumes", "--remove-orphans", "--rmi", "local") and check is False
        for args, check, _ in calls
    )


def test_compose_file_detection_prefers_harbor_names(tmp_path: Path) -> None:
    env = tmp_path / "environment"
    env.mkdir()
    compose = env / "docker-compose.yaml"
    compose.write_text("services: {}\n", encoding="utf-8")

    assert _compose_file(env) == compose


def test_compose_base_builds_or_uses_prebuilt_main() -> None:
    built = json.loads(_compose_base(docker_image=None))["services"]["main"]
    prebuilt = json.loads(_compose_base(docker_image="registry/task:v1"))["services"]["main"]

    assert built["build"] == {"context": "${CONTEXT_DIR}"}
    assert built["pull_policy"] == "build"
    assert built["image"] == "${MAIN_IMAGE_NAME}"
    assert prebuilt["image"] == "${PREBUILT_IMAGE_NAME}"
    assert "build" not in prebuilt
    assert built["command"] == prebuilt["command"] == ["sh", "-c", "sleep infinity"]


def test_compose_overlay_mounts_staged_tests_logs_and_runtime_config(tmp_path: Path) -> None:
    overlay = json.loads(
        _compose_overlay(
            tests_dir=tmp_path / "tests",
            logs=tmp_path / "logs",
            environment=_EnvironmentConfig(
                workdir="/srv/app",
                cpus=4,
                memory_mb=8192,
                env={"SERVICE_TOKEN": "resolved"},
            ),
        )
    )["services"]["main"]

    assert "working_dir" not in overlay
    assert overlay["environment"] == {"SERVICE_TOKEN": "resolved"}
    assert overlay["cpus"] == 4.0
    assert overlay["mem_limit"] == "8192m"
    assert overlay["volumes"] == [
        f"{tmp_path / 'tests'}:/tests:ro",
        f"{tmp_path / 'logs'}:/logs",
    ]


async def test_container_workdir_reads_running_container_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_docker(*args: str, check: bool = True) -> tuple[str, str]:
        assert args == ("inspect", "--format", "{{.Config.WorkingDir}}", "container-id")
        return "/srv/app\n", ""

    monkeypatch.setattr("hud.eval.runtime._docker", fake_docker)

    assert await _container_workdir("container-id") == "/srv/app"


async def test_container_workdir_defaults_to_docker_root_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_docker(*args: str, check: bool = True) -> tuple[str, str]:
        return "\n", ""

    monkeypatch.setattr("hud.eval.runtime._docker", fake_docker)

    assert await _container_workdir("container-id") == "/"


def test_prebuilt_environment_upload_requires_environment_only_files(tmp_path: Path) -> None:
    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    assert not _should_upload_environment_dir(env_dir=env_dir, docker_image="task:v1")

    payload = env_dir / "fixture.txt"
    payload.write_text("fixture", encoding="utf-8")
    assert _should_upload_environment_dir(env_dir=env_dir, docker_image="task:v1")
    assert not _should_upload_environment_dir(env_dir=env_dir, docker_image=None)

    dockerfile = env_dir / "Dockerfile"
    dockerfile.write_text("FROM task:v1\n", encoding="utf-8")
    assert not _should_upload_environment_dir(env_dir=env_dir, docker_image="task:v1")
    dockerfile.unlink()
    (env_dir / "compose.yaml").write_text("services: {}\n", encoding="utf-8")
    assert not _should_upload_environment_dir(env_dir=env_dir, docker_image="task:v1")


def test_docker_workspace_shell_argv_preserves_stdin_user_and_per_call_env(tmp_path: Path) -> None:
    workspace = _DockerWorkspace(
        tmp_path / "control",
        container="container-id",
        guest_path="/workspace",
        exec_user=1001,
    )

    argv = workspace.shell_argv(
        "cat > result.txt",
        cwd="/srv/app",
        env={"OVERRIDE": "new", "PER_CALL": "yes"},
    )

    assert argv[0].endswith("docker")
    assert argv[1:5] == ["exec", "-i", "--workdir", "/srv/app"]
    assert argv[5:7] == ["--user", "1001"]
    assert "OVERRIDE=new" in argv
    assert "PER_CALL=yes" in argv
    assert argv[-4:] == ["container-id", "bash", "-lc", "cat > result.txt"]
    # The current Workspace implementation owns the stdin relay. Keeping this
    # inherited prevents Docker-backed file writes from silently becoming empty.
    assert "_handle_process" not in _DockerWorkspace.__dict__
    assert _DockerWorkspace._handle_process is Workspace._handle_process


def test_read_harbor_reward_prefers_reward_key_and_preserves_named_metrics(tmp_path: Path) -> None:
    verifier = tmp_path / "verifier"
    verifier.mkdir()
    (verifier / "reward.json").write_text(
        json.dumps({"reward": 0.5, "passed": 3, "total": 5}), "utf-8"
    )

    reward, info = _read_harbor_reward(verifier)

    assert reward == 0.5
    assert info["reward_file"] == "reward.json"
    assert info["primary_reward_key"] == "reward"
    assert info["harbor_rewards"] == {"reward": 0.5, "passed": 3.0, "total": 5.0}


def test_read_harbor_reward_selects_explicit_named_metric(tmp_path: Path) -> None:
    verifier = tmp_path / "verifier"
    verifier.mkdir()
    (verifier / "reward.json").write_text(json.dumps({"tests_passed": 0.8, "style": 0.25}), "utf-8")

    reward, info = _read_harbor_reward(verifier, reward_key="tests_passed")

    assert reward == 0.8
    assert info["primary_reward_key"] == "tests_passed"
    assert info["harbor_rewards"] == {"tests_passed": 0.8, "style": 0.25}


def test_read_harbor_reward_selects_a_single_named_metric(tmp_path: Path) -> None:
    verifier = tmp_path / "verifier"
    verifier.mkdir()
    (verifier / "reward.json").write_text(json.dumps({"quality": 0.75}), "utf-8")

    reward, info = _read_harbor_reward(verifier)

    assert reward == 0.75
    assert info["primary_reward_key"] == "quality"


def test_read_harbor_reward_rejects_ambiguous_named_metrics(tmp_path: Path) -> None:
    verifier = tmp_path / "verifier"
    verifier.mkdir()
    (verifier / "reward.json").write_text(json.dumps({"passed": 3, "total": 5}), "utf-8")

    reward, info = _read_harbor_reward(verifier)

    assert reward is None
    assert info["harbor_rewards"] == {"passed": 3.0, "total": 5.0}
    assert "no unambiguous primary reward" in info["reward_parse_error"]


def test_read_harbor_reward_does_not_special_case_score(tmp_path: Path) -> None:
    verifier = tmp_path / "verifier"
    verifier.mkdir()
    (verifier / "reward.json").write_text(
        json.dumps({"score": 1.0, "style": 0.5}), encoding="utf-8"
    )

    reward, info = _read_harbor_reward(verifier)

    assert reward is None
    assert info["reward_file"] == "reward.json"
    assert "no unambiguous primary reward" in info["reward_parse_error"]


@pytest.mark.parametrize(
    "invalid_reward",
    [float("nan"), float("inf"), float("-inf"), 10**1000],
    ids=["nan", "positive-infinity", "negative-infinity", "huge-integer"],
)
def test_read_harbor_reward_rejects_non_finite_numbers(
    tmp_path: Path,
    invalid_reward: float | int,
) -> None:
    verifier = tmp_path / "verifier"
    verifier.mkdir()
    (verifier / "reward.json").write_text(json.dumps({"reward": invalid_reward}), encoding="utf-8")

    reward, info = _read_harbor_reward(verifier)

    assert reward is None
    assert info["reward_file"] == "reward.json"
    assert "non-numeric reward(s): reward" in info["reward_parse_error"]


def test_read_harbor_reward_rejects_oversized_reward_file(tmp_path: Path) -> None:
    verifier = tmp_path / "verifier"
    verifier.mkdir()
    (verifier / "reward.json").write_text(" " * 1_000_001, encoding="utf-8")

    reward, info = _read_harbor_reward(verifier)

    assert reward is None
    assert info == {
        "reward_file": "reward.json",
        "reward_parse_error": "reward.json exceeds 1000000 bytes",
    }


@pytest.mark.parametrize("invalid_reward", ["NaN", "Infinity", "-Infinity", "1e1000000"])
def test_read_harbor_reward_txt_rejects_non_finite_values(
    tmp_path: Path,
    invalid_reward: str,
) -> None:
    verifier = tmp_path / "verifier"
    verifier.mkdir()
    (verifier / "reward.txt").write_text(invalid_reward, encoding="utf-8")

    reward, info = _read_harbor_reward(verifier)

    assert reward is None
    assert info["reward_file"] == "reward.txt"
    assert info["reward_parse_error"] == invalid_reward


def test_read_harbor_reward_reports_invalid_json(tmp_path: Path) -> None:
    verifier = tmp_path / "verifier"
    verifier.mkdir()
    (verifier / "reward.json").write_text('{"reward":', "utf-8")

    reward, info = _read_harbor_reward(verifier)

    assert reward is None
    assert info["reward_parse_error"].startswith("invalid reward.json:")


@pytest.mark.parametrize("reward_name", ["reward.json", "reward.txt"])
def test_read_harbor_reward_rejects_symlinks(tmp_path: Path, reward_name: str) -> None:
    verifier = tmp_path / "verifier"
    verifier.mkdir()
    target = tmp_path / "outside-reward"
    target.write_text('{"reward": 1.0}' if reward_name.endswith("json") else "1.0", "utf-8")
    (verifier / reward_name).symlink_to(target)

    reward, info = _read_harbor_reward(verifier)

    assert reward is None
    assert info["reward_file"] == reward_name
    assert info["reward_parse_error"] == f"{reward_name} is not a regular file"


async def test_grade_stages_hidden_tests_clears_forged_reward_and_reads_fresh_result(
    single_task: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tests_dir = single_task / "tests"
    (tests_dir / "hidden.txt").write_text("secret verifier fixture", encoding="utf-8")
    staged_tests = tmp_path / "staged-tests"
    staged_tests.mkdir()
    assert list(staged_tests.iterdir()) == []

    logs = tmp_path / "logs"
    verifier_logs = logs / "verifier"
    verifier_logs.mkdir(parents=True)
    (verifier_logs / "reward.txt").write_text("1.0\n", encoding="utf-8")
    (verifier_logs / "forged-artifact.txt").write_text("agent supplied", encoding="utf-8")
    secret = "verifier-secret-never-in-argv"
    events: list[str] = []

    async def fake_docker(
        *args: str,
        check: bool = True,
        env: dict[str, str] | None = None,
        unset_env: tuple[str, ...] = (),
    ) -> tuple[str, str]:
        events.append("docker")
        assert check is False
        assert env is None
        assert unset_env == ()
        assert args[:4] == ("exec", "--workdir", "/app", "--user")
        assert args[4] == "1002"
        assert "--env-file" in args
        env_file = staged_tests.parent / "verifier.env"
        assert args[args.index("--env-file") + 1] == str(env_file)
        assert env_file.read_text("utf-8") == (f"SHARED=environment\nVERIFIER_ONLY={secret}\n")
        assert env_file.stat().st_mode & 0o777 == 0o600
        assert secret not in " ".join(args)
        assert args[-2:] == ("container", "/tests/test.sh")
        assert (staged_tests / "hidden.txt").read_text("utf-8") == "secret verifier fixture"
        assert not (verifier_logs / "reward.txt").exists()
        assert not (verifier_logs / "forged-artifact.txt").exists()
        (verifier_logs / "reward.json").write_text(json.dumps({"checks": 0.25}), encoding="utf-8")
        return "verifier out", ""

    async def fake_release_log_permissions(container: str) -> None:
        assert container == "container"
        events.append("release")

    def fake_prepare_verifier_staging(
        source_tests: Path,
        destination_tests: Path,
        destination_logs: Path,
    ) -> None:
        assert events == ["release"]
        events.append("stage")
        _prepare_verifier_staging(source_tests, destination_tests, destination_logs)

    monkeypatch.setattr("hud.eval.runtime._docker", fake_docker)
    monkeypatch.setattr(
        "integrations.harbor_runtime._release_log_permissions",
        fake_release_log_permissions,
    )
    monkeypatch.setattr(
        "integrations.harbor_runtime._prepare_verifier_staging",
        fake_prepare_verifier_staging,
    )
    runtime = HarborRuntime(single_task.parent)

    result = await runtime._grade(
        "container",
        "/app",
        tests_dir,
        staged_tests,
        logs,
        "done",
        verifier_timeout=120.0,
        verifier_user=1002,
        verifier_env={"SHARED": "environment", "VERIFIER_ONLY": secret},
    )

    assert result["score"] == 0.25
    assert result["info"]["stdout"] == "verifier out"
    assert result["info"]["reward_file"] == "reward.json"
    assert events == ["release", "stage", "docker", "release"]
    assert (staged_tests.parent / "agent_answer.txt").read_text("utf-8") == "done"


async def test_grade_times_out_when_verifier_hangs(
    single_task: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancelled = asyncio.Event()

    async def fake_docker(
        *args: str,
        check: bool = True,
        env: dict[str, str] | None = None,
        unset_env: tuple[str, ...] = (),
    ) -> tuple[str, str]:
        try:
            await asyncio.sleep(3600)
            raise AssertionError("unreachable")
        except asyncio.CancelledError:
            cancelled.set()
            raise

    async def fake_release_log_permissions(container: str) -> None:
        assert container == "container"

    monkeypatch.setattr("hud.eval.runtime._docker", fake_docker)
    monkeypatch.setattr(
        "integrations.harbor_runtime._release_log_permissions",
        fake_release_log_permissions,
    )
    runtime = HarborRuntime(single_task.parent)
    staged_tests = tmp_path / "staged-tests"
    staged_tests.mkdir()

    result = await runtime._grade(
        "container",
        "/app",
        single_task / "tests",
        staged_tests,
        tmp_path / "logs",
        None,
        verifier_timeout=0.05,
        verifier_user=None,
        verifier_env={},
    )

    assert result["isError"] is True
    assert "timed out" in result["content"]
    assert cancelled.is_set()


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
    assert tomllib.loads(toml) == {
        "schema_version": "1.4",
        "task": {"name": "hud/solve-99dd84a6"},
        # HUD task/args are metadata, not invalid root-level config fields.
        "metadata": {"hud_task": "solve", "hud_args": '{"n": 2}'},
        "agent": {"timeout_sec": 600.0},
        "verifier": {"timeout_sec": 600.0},
    }


async def test_scripts_drive_hud_task_lifecycle(tmp_path: Path) -> None:
    _write_env(tmp_path)
    created = await export(str(tmp_path / "env.py"), tmp_path / "out")
    boot = (created[0] / "environment" / "hud_entrypoint.sh").read_text(encoding="utf-8")
    test_sh = (created[0] / "tests" / "test.sh").read_text(encoding="utf-8")

    # Boot serves the channel, parks the run via setup, then hands off.
    assert "HUD_ENV_TARGET=env.py:env" in boot
    assert 'hud serve "$HUD_ENV_TARGET"' in boot
    assert '-- "$HUD_TASK"' in boot
    assert boot.index('hud task start --args "$HUD_ARGS"') < boot.index(': > "$HUD_READY_FILE"')
    assert 'exec "$@"' in boot
    # Verifier grades the parked run and writes the Harbor reward.
    assert 'hud task grade --args "$HUD_ARGS"' in test_sh
    assert '-- "$HUD_TASK"' in test_sh
    assert "--answer-file" in test_sh
    assert "/logs/verifier/reward.txt" in test_sh
    assert "echo 0" not in test_sh
    assert 'exit "$status"' in test_sh

    await asyncio.to_thread(
        subprocess.run,
        ["sh", "-n", str(created[0] / "environment" / "hud_entrypoint.sh")],
        check=True,
    )
    await asyncio.to_thread(
        subprocess.run,
        ["sh", "-n", str(created[0] / "tests" / "test.sh")],
        check=True,
    )


async def test_dockerfile_preserves_original_startup_and_appends_boot(tmp_path: Path) -> None:
    _write_env(tmp_path)
    created = await export(str(tmp_path / "env.py"), tmp_path / "out")
    dockerfile = (created[0] / "environment" / "Dockerfile").read_text(encoding="utf-8")
    assert 'CMD ["hud", "serve", "env:env"]' in dockerfile
    assert 'ENTRYPOINT ["/hud_entrypoint.sh"]' in dockerfile
    assert 'CMD ["test", "-f", "/tmp/.hud-harbor-ready"]' in dockerfile
    # The env build context is copied so the image can be rebuilt under Harbor.
    assert (created[0] / "environment" / "env.py").exists()


async def test_custom_answer_file(tmp_path: Path) -> None:
    _write_env(tmp_path)
    created = await export(str(tmp_path / "env.py"), tmp_path / "out", answer_file="/app/out.txt")
    assert "/app/out.txt" in (created[0] / "instruction.md").read_text(encoding="utf-8")
    assert "/app/out.txt" in (created[0] / "tests" / "test.sh").read_text(encoding="utf-8")


async def test_export_preserves_nondefault_module_and_symbol(tmp_path: Path) -> None:
    source = tmp_path / "custom_runtime.py"
    source.write_text(
        textwrap.dedent(_ENV_PY)
        .replace("env = Environment", "project_runtime = Environment")
        .replace("@env.template", "@project_runtime.template"),
        encoding="utf-8",
    )
    (tmp_path / "Dockerfile").write_text(
        'FROM python:3.11-slim\nCOPY custom_runtime.py ./\nCMD ["true"]\n',
        encoding="utf-8",
    )

    created = await export(str(source), tmp_path / "out")

    boot = (created[0] / "environment" / "hud_entrypoint.sh").read_text(encoding="utf-8")
    assert "HUD_ENV_TARGET=custom_runtime.py:project_runtime" in boot


async def test_json_taskset_exclusion_preserves_other_json_build_inputs(tmp_path: Path) -> None:
    _write_env(tmp_path)
    taskset = tmp_path / "tasks.json"
    taskset.write_text(
        json.dumps([{"env": "demo", "id": "solve", "args": {"n": 2}}]),
        encoding="utf-8",
    )
    preserved = {
        "package.json": '{"scripts":{"start":"node index.js"}}\n',
        "tsconfig.json": '{"compilerOptions":{"strict":true}}\n',
        "fixtures.jsonl": '{"input":1}\n',
    }
    for name, content in preserved.items():
        (tmp_path / name).write_text(content, encoding="utf-8")

    created = await export(str(taskset), tmp_path / "out")
    environment = created[0] / "environment"

    assert not (environment / taskset.name).exists()
    for name, content in preserved.items():
        assert (environment / name).read_text(encoding="utf-8") == content


async def test_export_normalizes_traversal_slug_and_rejects_collisions(tmp_path: Path) -> None:
    _write_env(tmp_path)
    taskset = tmp_path / "traversal.json"
    taskset.write_text(
        json.dumps([{"env": "demo", "id": "solve", "args": {"n": 2}, "slug": "../../escape"}]),
        encoding="utf-8",
    )
    out = tmp_path / "out"

    created = await export(str(taskset), out)

    assert created == [out.resolve() / "escape"]
    assert not (tmp_path / "escape").exists()

    taskset.write_text(
        json.dumps(
            [
                {"env": "demo", "id": "solve", "args": {"n": 1}, "slug": "a/b"},
                {"env": "demo", "id": "solve", "args": {"n": 2}, "slug": "ab"},
            ]
        ),
        encoding="utf-8",
    )
    collision_out = tmp_path / "collision-out"
    with pytest.raises(ValueError, match="normalize to Harbor folder 'ab'"):
        await export(str(taskset), collision_out)
    assert not (collision_out / "ab").exists()


async def test_generated_scripts_preserve_hostile_values_as_single_argv(tmp_path: Path) -> None:
    sentinel = tmp_path / "injected"
    task_id = "--help'$(touch injected)\nnext"
    arg_value = "quote' ; touch injected; $(touch injected)\nline"
    source = tmp_path / "custom_runtime.py"
    source.write_text(
        textwrap.dedent(
            f"""\
            from hud import Environment
            from hud.eval import Task

            project_runtime = Environment("demo")

            @project_runtime.template(id={task_id!r})
            async def solve(payload: str):
                yield "hostile"
                yield 1.0

            tasks = [Task(
                env="demo",
                id={task_id!r},
                args={{"payload": {arg_value!r}}},
                slug="hostile-safe",
            )]
            """
        ),
        encoding="utf-8",
    )
    (tmp_path / "Dockerfile").write_text("FROM python:3.11-slim\n", encoding="utf-8")
    answer_file = str(tmp_path / "answer'$(touch injected)\nfile")
    created = await export(str(source), tmp_path / "out", answer_file=answer_file)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_hud = fake_bin / "hud"
    fake_hud.write_text(
        textwrap.dedent("""\
            #!/bin/sh
            if [ "$1" = "serve" ]; then
                exit 0
            fi
            printf '%s\\037' "$@" > "$HUD_CAPTURE"
            printf '0\\n'
        """),
        encoding="utf-8",
    )
    fake_hud.chmod(0o755)
    fake_python = fake_bin / "python3"
    fake_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_python.chmod(0o755)
    command_env = {"PATH": f"{fake_bin}:/usr/bin:/bin"}

    boot_capture = tmp_path / "boot-argv"
    boot = (created[0] / "environment" / "hud_entrypoint.sh").read_text(encoding="utf-8")
    boot_runner = tmp_path / "boot-runner.sh"
    boot_runner.write_text(
        boot.replace("/tmp/.hud-harbor-ready", str(tmp_path / "ready")),
        encoding="utf-8",
    )
    await asyncio.to_thread(
        subprocess.run,
        ["sh", str(boot_runner), "true"],
        check=True,
        cwd=tmp_path,
        env={**command_env, "HUD_CAPTURE": str(boot_capture)},
    )
    boot_argv = boot_capture.read_bytes().split(b"\x1f")[:-1]
    assert [part.decode() for part in boot_argv] == [
        "task",
        "start",
        "--args",
        json.dumps({"payload": arg_value}),
        "--url",
        "tcp://127.0.0.1:8765",
        "--",
        task_id,
    ]

    grade_capture = tmp_path / "grade-argv"
    verifier_logs = tmp_path / "verifier-logs"
    verifier = (created[0] / "tests" / "test.sh").read_text(encoding="utf-8")
    verifier_runner = tmp_path / "verifier-runner.sh"
    verifier_runner.write_text(
        verifier.replace("/logs/verifier", str(verifier_logs)),
        encoding="utf-8",
    )
    await asyncio.to_thread(
        subprocess.run,
        ["sh", str(verifier_runner)],
        check=True,
        cwd=tmp_path,
        env={**command_env, "HUD_CAPTURE": str(grade_capture)},
    )
    grade_argv = grade_capture.read_bytes().split(b"\x1f")[:-1]
    assert [part.decode() for part in grade_argv] == [
        "task",
        "grade",
        "--args",
        json.dumps({"payload": arg_value}),
        "--answer-file",
        answer_file,
        "--url",
        "tcp://127.0.0.1:8765",
        "--",
        task_id,
    ]
    assert not sentinel.exists()


def test_task_cli_option_terminator_keeps_help_like_id_positional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []

    def stop_after_parse(task: str, *_args: object) -> None:
        captured.append(task)
        raise RuntimeError("parsed")

    monkeypatch.setattr("hud.cli.task._resolve", stop_after_parse)

    result = CliRunner().invoke(
        app,
        ["task", "start", "--url", "tcp://127.0.0.1:1", "--", "--help"],
    )

    assert captured == ["--help"]
    assert isinstance(result.exception, RuntimeError)


async def test_entrypoint_fails_closed_when_server_dies(tmp_path: Path) -> None:
    src = _write_env(tmp_path)
    created = await export(str(src), tmp_path / "out")
    fake_bin = tmp_path / "dead-server-bin"
    fake_bin.mkdir()
    start_seen = tmp_path / "start-seen"
    fake_hud = fake_bin / "hud"
    fake_hud.write_text(
        textwrap.dedent(
            f"""\
            #!/bin/sh
            if [ "$1" = "serve" ]; then
                exit 7
            fi
            : > {str(start_seen)!r}
            exit 0
            """
        ),
        encoding="utf-8",
    )
    fake_hud.chmod(0o755)
    final_seen = tmp_path / "final-seen"
    boot = (created[0] / "environment" / "hud_entrypoint.sh").read_text(encoding="utf-8")
    boot_runner = tmp_path / "dead-server-entrypoint.sh"
    boot_runner.write_text(
        boot.replace("/tmp/.hud-harbor-ready", str(tmp_path / "ready")),
        encoding="utf-8",
    )

    result = await asyncio.to_thread(
        subprocess.run,
        ["sh", str(boot_runner), "sh", "-c", f": > {str(final_seen)!r}"],
        check=False,
        timeout=5,
        env={"PATH": f"{fake_bin}:{sys.executable.rpartition('/')[0]}:/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert not start_seen.exists()
    assert not final_seen.exists()


async def test_entrypoint_cleans_up_and_fails_closed_when_setup_fails(tmp_path: Path) -> None:
    src = _write_env(tmp_path)
    created = await export(str(src), tmp_path / "out")
    fake_bin = tmp_path / "setup-failure-bin"
    fake_bin.mkdir()
    server_pid_file = tmp_path / "server-pid"
    fake_hud = fake_bin / "hud"
    fake_hud.write_text(
        textwrap.dedent("""\
            #!/bin/sh
            if [ "$1" = "serve" ]; then
                printf '%s' "$$" > "$SERVER_PID_FILE"
                exec sleep 30
            fi
            echo 'setup failed' >&2
            exit 9
        """),
        encoding="utf-8",
    )
    fake_hud.chmod(0o755)
    fake_python = fake_bin / "python3"
    fake_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_python.chmod(0o755)
    final_seen = tmp_path / "final-seen"
    ready = tmp_path / "ready"
    boot = (created[0] / "environment" / "hud_entrypoint.sh").read_text(encoding="utf-8")
    boot_runner = tmp_path / "setup-failure-entrypoint.sh"
    boot_runner.write_text(boot.replace("/tmp/.hud-harbor-ready", str(ready)), encoding="utf-8")

    result = await asyncio.to_thread(
        subprocess.run,
        ["sh", str(boot_runner), "sh", "-c", f": > {str(final_seen)!r}"],
        check=False,
        timeout=5,
        env={
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "SERVER_PID_FILE": str(server_pid_file),
        },
        capture_output=True,
        text=True,
    )

    assert result.returncode == 9
    assert "setup failed" in result.stderr
    assert not ready.exists()
    assert not final_seen.exists()
    server_pid = server_pid_file.read_text(encoding="utf-8")
    process_check = await asyncio.to_thread(
        subprocess.run,
        ["kill", "-0", server_pid],
        check=False,
    )
    assert process_check.returncode != 0


async def test_verifier_failure_leaves_no_reward_file(tmp_path: Path) -> None:
    src = _write_env(tmp_path)
    created = await export(
        str(src),
        tmp_path / "out",
        answer_file=str(tmp_path / "answer.txt"),
    )
    fake_bin = tmp_path / "grade-failure-bin"
    fake_bin.mkdir()
    fake_hud = fake_bin / "hud"
    fake_hud.write_text(
        "#!/bin/sh\nprintf 'partial reward\\n'\necho 'grade infrastructure failed' >&2\nexit 7\n",
        encoding="utf-8",
    )
    fake_hud.chmod(0o755)
    verifier_logs = tmp_path / "verifier-logs"
    verifier = (created[0] / "tests" / "test.sh").read_text(encoding="utf-8")
    verifier_runner = tmp_path / "grade-failure-verifier.sh"
    verifier_runner.write_text(
        verifier.replace("/logs/verifier", str(verifier_logs)),
        encoding="utf-8",
    )

    result = await asyncio.to_thread(
        subprocess.run,
        ["sh", str(verifier_runner)],
        check=False,
        env={"PATH": f"{fake_bin}:/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 7
    assert "grade infrastructure failed" in result.stderr
    assert not (verifier_logs / "reward.txt").exists()
