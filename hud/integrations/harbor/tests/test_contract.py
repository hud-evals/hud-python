"""Observable contracts for adapting Harbor tasks into HUD images."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from hud.eval import Task
from hud.integrations import harbor

from .conftest import make_harbor_task, make_multi_step_task


def _tree_snapshot(root: Path) -> dict[str, tuple[str, bytes | str]]:
    snapshot: dict[str, tuple[str, bytes | str]] = {}
    for entry in sorted(root.rglob("*")):
        relative = entry.relative_to(root).as_posix()
        if entry.is_symlink():
            snapshot[relative] = ("symlink", os.readlink(entry))
        elif entry.is_file():
            snapshot[relative] = ("file", entry.read_bytes())
        else:
            snapshot[relative] = ("directory", b"")
    return snapshot


def _assert_stock_compose_complete(compose_path: Path) -> dict[str, Any]:
    project = json.loads(compose_path.read_text("utf-8"))
    services = project["services"]
    assert isinstance(services, dict)
    for name, service in services.items():
        assert isinstance(service, dict)
        assert service.get("image") or service.get("build"), name
        for volume in service.get("volumes", []):
            if not isinstance(volume, str):
                continue
            source, separator, target = volume.partition(":")
            if separator and target == "/media/hud/tests:ro":
                tests = (compose_path.parent / source).resolve()
                tests.relative_to(compose_path.parent.resolve())
                assert tests.is_dir(), (name, tests)
        build = service.get("build")
        if build is None:
            continue
        build_config = {"context": build} if isinstance(build, str) else build
        assert isinstance(build_config, dict)
        context_value = build_config.get("context", ".")
        assert isinstance(context_value, str)
        context = (compose_path.parent / context_value).resolve()
        context.relative_to(compose_path.parent.resolve())
        assert context.is_dir(), (name, context)
        dockerfile_value = build_config.get("dockerfile", "Dockerfile")
        assert isinstance(dockerfile_value, str)
        assert (context / dockerfile_value).is_file(), (name, dockerfile_value)
        additional_contexts = build_config.get("additional_contexts", {})
        assert isinstance(additional_contexts, dict)
        for target in additional_contexts.values():
            assert isinstance(target, str)
            if target.startswith("service:"):
                assert target.removeprefix("service:") in services
            else:
                # Local additional-context paths resolve against the project
                # directory (the compose file's directory), not the service
                # build context.
                named_context = (compose_path.parent / target).resolve()
                named_context.relative_to(compose_path.parent.resolve())
                assert named_context.is_dir(), (name, named_context)
    return project


def _environment_config(context: Path) -> dict[str, Any]:
    paths = list((context / "compose-project").glob("*/config.json"))
    assert len(paths) == 1
    config = json.loads(paths[0].read_text("utf-8"))
    assert isinstance(config, dict)
    return config


def test_adapt_packages_an_image_task_as_a_compose_project(tmp_path: Path) -> None:
    task_dir = make_harbor_task(tmp_path, "task-a")
    authored_environment = _tree_snapshot(task_dir / "environment")

    taskset = harbor.adapt(tmp_path)

    (task,) = list(taskset)
    assert task.id == "run"
    assert task.slug == "task-a"
    assert task.args["instruction"] == "Solve the task."
    assert task.args["task"] == {
        "artifacts": [],
        "collect": [],
        "description": "",
        "id": "task-a",
        "separate_verifier": False,
        "verifier_timeout": 120.0,
    }
    assert task.runtime_config is not None
    assert task.runtime_config.image is None

    (context,) = (tmp_path / ".hud-adapt").iterdir()
    assert not (context / "Dockerfile").exists()
    assert not any(path.name == ".hud" for path in context.rglob(".hud"))
    assert {entry.name for entry in context.iterdir()} == {
        "compose-project",
        "compose.yaml",
        "env.py",
        "tasks.json",
    }
    # env.py names the environment as a literal — `hud deploy` resolves the
    # context's identity from source, and refuses a computed name.
    served = (context / "env.py").read_text(encoding="utf-8")
    assert f'Environment("{context.name}")' in served
    assert 'Environment(CONFIG["name"])' not in served
    project_root = context / "compose-project"
    assert _tree_snapshot(project_root / "environment") == authored_environment
    payload = project_root / "hud"
    assert {entry.name for entry in payload.iterdir()} == {
        "config.json",
        "env.py",
        "install.sh",
        "packages",
    }
    build_script = (project_root / "build.sh").read_text("utf-8")
    assert "docker image inspect" in build_script
    assert 'docker tag "$(docker compose' in build_script
    subprocess.run(["sh", "-n", project_root / "build.sh"], check=True)
    assert (project_root / "tests" / "task-a" / "test.sh").is_file()
    assert not any(path.name in {"tasks", "tasks.json"} for path in payload.rglob("*"))
    assert not (context / "compose.json").exists()
    assert task.runtime_config.compose == context / "compose-project" / "compose.json"
    assert task.runtime_config.compose_project == context
    compose_path = context / "compose-project" / "compose.json"
    project = _assert_stock_compose_complete(compose_path)
    assert set(project["services"]) == {"main"}
    main = project["services"]["main"]
    assert main["image"].startswith("hud-harbor:")
    assert main["build"] == {
        "additional_contexts": {"hud": "./hud"},
        "context": "./environment",
        "dockerfile": "../Dockerfile",
    }
    assert main["volumes"] == ["./tests:/media/hud/tests:ro"]
    combined = project_root / "Dockerfile"
    assert combined.read_text("utf-8").startswith("FROM python:3.11-slim AS hud-base\n")
    recipe = _assert_stock_compose_complete(context / "compose.yaml")
    assert recipe["services"]["main"]["build"]["context"] == ("./compose-project/environment")
    assert recipe["services"]["main"]["volumes"] == ["./compose-project/tests:/media/hud/tests:ro"]


def test_task_content_changes_do_not_rebuild_the_environment(tmp_path: Path) -> None:
    task_dir = make_harbor_task(tmp_path, "task-a", instruction="First instruction")

    (before,) = list(harbor.adapt(tmp_path))
    assert before.runtime_config is not None
    assert isinstance(before.runtime_config.compose, Path)
    before_compose = json.loads(before.runtime_config.compose.read_text("utf-8"))
    before_image = before_compose["services"]["main"]["image"]

    (task_dir / "instruction.md").write_text("Second instruction", encoding="utf-8")
    (task_dir / "tests" / "test.sh").write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    (after,) = list(harbor.adapt(tmp_path))
    assert after.runtime_config is not None
    assert isinstance(after.runtime_config.compose, Path)
    after_compose = json.loads(after.runtime_config.compose.read_text("utf-8"))

    assert after_compose["services"]["main"]["image"] == before_image
    assert after.args["instruction"] == "Second instruction"
    assert (after.runtime_config.compose.parent / "tests" / "task-a" / "test.sh").read_text(
        "utf-8"
    ) == "#!/bin/sh\nexit 1\n"


def test_image_task_preserves_a_named_final_stage_verbatim(tmp_path: Path) -> None:
    dockerfile = 'FROM alpine AS build\r\nRUN true\r\nFROM alpine AS final\r\nCMD ["sh"]\r\n'
    make_harbor_task(tmp_path, "task-a", dockerfile=dockerfile)

    harbor.adapt(tmp_path)

    (context,) = (tmp_path / ".hud-adapt").iterdir()
    environment = context / "compose-project" / "environment"
    assert (environment / "Dockerfile").read_bytes() == dockerfile.encode("utf-8")
    combined = (environment.parent / "Dockerfile").read_bytes().decode("utf-8")
    assert combined.startswith(dockerfile + "\nFROM final AS hud-runtime\n")


@pytest.mark.parametrize("stage", ["hud-base", "HUD-RUNTIME"])
def test_image_task_rejects_reserved_user_stage_names(
    tmp_path: Path,
    stage: str,
) -> None:
    make_harbor_task(tmp_path, "task-a", dockerfile=f"FROM alpine AS {stage}\n")

    with pytest.raises(ValueError, match="reserved stage"):
        harbor.adapt(tmp_path)


def test_image_task_preserves_environment_ignored_paths_verbatim(
    tmp_path: Path,
) -> None:
    task = make_harbor_task(tmp_path, "task-a")
    environment = task / "environment"
    (environment / ".dockerignore").write_text("hud/\nDockerfile\n", encoding="utf-8")
    (environment / "ignored.txt").write_bytes(b"unchanged\x00payload")
    authored = _tree_snapshot(environment)

    harbor.adapt(tmp_path)

    (context,) = (tmp_path / ".hud-adapt").iterdir()
    project = context / "compose-project"
    assert _tree_snapshot(project / "environment") == authored
    assert (project / "hud").is_dir()
    assert (project / "Dockerfile").is_file()


def test_adapt_honors_compose_main_build_settings(
    tmp_path: Path,
) -> None:
    task = make_harbor_task(tmp_path, "task-a", dockerfile=None)
    environment = task / "environment"
    environment.mkdir()
    (environment / "compose.yaml").write_text(
        """\
services:
  main:
    build:
      context: .
      dockerfile: Containerfile
      args:
        FLAVOR: compose
""",
        encoding="utf-8",
    )
    (environment / "Containerfile").write_text("FROM python:3.12\n", encoding="utf-8")

    (row,) = list(harbor.adapt(tmp_path))

    assert row.runtime_config is not None
    compose_path = row.runtime_config.compose
    assert isinstance(compose_path, Path)
    project = _assert_stock_compose_complete(compose_path)
    base = project["services"]["hud-base"]
    assert base["build"]["dockerfile"] == "Containerfile"
    assert base["build"]["args"] == {"FLAVOR": "compose"}
    assert base["scale"] == 0


def test_adapt_emits_compose_project_and_peers(
    tmp_path: Path,
) -> None:
    task = make_harbor_task(tmp_path, "task-a")
    (task / "environment" / "compose.yaml").write_text(
        """\
services:
  main:
    environment: {FROM_COMPOSE: "yes"}
    expose: [8080]
    healthcheck:
      test: [CMD-SHELL, curl -f http://localhost:8080/health]
      interval: 2s
      timeout: 3s
      retries: 5
      start_period: 1s
  redis:
    image: redis:7-alpine
    command: [redis-server, --save, ""]
    environment: {SIDE: car}
    expose: [6379]
    healthcheck:
      test: [CMD, redis-cli, ping]
      interval: 2s
      timeout: 3s
      retries: 5
      start_period: 1s
""",
        encoding="utf-8",
    )

    (row,) = list(harbor.adapt(tmp_path))

    assert row.runtime_config is not None
    assert row.runtime_config.image is None
    (context,) = (tmp_path / ".hud-adapt").iterdir()
    assert row.runtime_config.compose == context / "compose-project" / "compose.json"
    assert row.runtime_config.compose_project == context
    assert not (context / "compose.json").exists()
    compose_path = row.runtime_config.compose
    assert isinstance(compose_path, Path)
    project = _assert_stock_compose_complete(compose_path)
    assert project["services"]["redis"]["image"] == "redis:7-alpine"
    assert "build" not in project["services"]["redis"]
    assert project["services"]["main"]["build"]["context"] == "./main"
    assert project["services"]["main"]["build"]["additional_contexts"] == {
        "hud-base": "service:hud-base"
    }
    assert project["services"]["main"]["image"].startswith("hud-harbor:")
    assert project["services"]["hud-base"]["image"].startswith("hud-harbor-base:")
    assert project["services"]["hud-base"]["build"]["context"] == "./environment"
    assert (context / "compose-project" / "environment" / "Dockerfile").is_file()
    assert not any(path.name == ".hud" for path in context.rglob(".hud"))
    recipe = _assert_stock_compose_complete(context / "compose.yaml")
    assert recipe["services"]["main"]["build"]["context"] == "./compose-project/main"
    assert recipe["services"]["redis"]["image"] == "redis:7-alpine"
    redis = project["services"]["redis"]
    assert redis["image"] == "redis:7-alpine"
    assert redis["environment"] == {"SIDE": "car"}
    assert redis["command"] == ["redis-server", "--save", ""]
    assert redis["expose"] == ["6379"]
    assert redis["healthcheck"]["test"] == ["CMD", "redis-cli", "ping"]
    assert "build" not in redis
    manifest = _environment_config(context)
    assert manifest["environment"]["env"] == {"FROM_COMPOSE": "yes"}
    assert manifest["environment"]["healthcheck"] == {
        "command": "curl -f http://localhost:8080/health",
        "interval_sec": 2.0,
        "timeout_sec": 3.0,
        "start_period_sec": 1.0,
        "start_interval_sec": 5.0,
        "retries": 5,
    }
    assert manifest["local_aliases"] == ["main"]
    assert manifest["ports"] == [8080]
    assert manifest["capabilities"] == []
    assert manifest["peers"] == [{"name": "redis", "port": 6379}]
    assert project["services"]["main"]["command"] == [
        "/media/hud/venv/bin/hud",
        "serve",
        "/media/hud/env.py",
        "--host",
        "0.0.0.0",
        "--port",
        "8765",
    ]
    assert "healthcheck" not in project["services"]["main"]


def test_compose_adapt_retains_builds_without_local_docker(
    tmp_path: Path,
) -> None:
    """Hosted adaptation produces a private-build project without a local daemon."""
    task = make_harbor_task(tmp_path, "task-a")
    (task / "environment" / "Dockerfile").write_text(
        'FROM python:3.12\nWORKDIR /app\nENTRYPOINT ["/app/start"]\n',
        encoding="utf-8",
    )
    database = task / "environment" / "database"
    database.mkdir()
    (database / "Dockerfile").write_text("FROM postgres:16\n", encoding="utf-8")
    (database / "db.env").write_text("POSTGRES_DB=test\n", encoding="utf-8")
    (database / "data").mkdir()
    (task / "environment" / "compose.yaml").write_text(
        """\
services:
  main:
    env_file: ./main.env
    volumes: [./main-data:/var/lib/main]
  database:
    build:
      context: ./database
    env_file: ./database/db.env
    volumes: [./database/data:/var/lib/postgresql/data]
    expose: [5432]
  redis:
    image: redis:7-alpine
    expose: [6379]
""",
        encoding="utf-8",
    )
    (task / "environment" / "main.env").write_text("MAIN=true\n", encoding="utf-8")
    (task / "environment" / "main-data").mkdir()

    (row,) = list(harbor.adapt(tmp_path))

    assert row.runtime_config is not None
    compose_path = row.runtime_config.compose
    assert isinstance(compose_path, Path)
    project = json.loads(compose_path.read_text("utf-8"))
    assert project["services"]["database"]["build"]["context"] == ("./environment/database")
    assert project["services"]["database"]["env_file"] == "./environment/database/db.env"
    assert project["services"]["database"]["volumes"] == [
        "./environment/database/data:/var/lib/postgresql/data"
    ]
    assert project["services"]["main"]["env_file"] == "./environment/main.env"
    assert "./environment/main-data:/var/lib/main" in project["services"]["main"]["volumes"]
    assert project["services"]["redis"]["image"] == "redis:7-alpine"
    assert project["services"]["main"]["build"]["additional_contexts"] == {
        "hud-base": "service:hud-base"
    }


def test_adapt_moves_compose_main_process_settings_into_the_workspace(
    tmp_path: Path,
) -> None:
    task = make_harbor_task(tmp_path, "task-a")
    (task / "environment" / "compose.yaml").write_text(
        """\
services:
  main:
    user: 1001:1002
    entrypoint: [/compose-init]
    working_dir: /compose-work
""",
        encoding="utf-8",
    )

    harbor.adapt(tmp_path)

    (context,) = (tmp_path / ".hud-adapt").iterdir()
    manifest = _environment_config(context)
    compose = _assert_stock_compose_complete(context / "compose-project" / "compose.json")
    assert manifest["image_user"] == "1001:1002"
    assert manifest["entrypoint"] == ["/compose-init"]
    assert manifest["workdir"] == "/compose-work"
    assert "user" not in compose["services"]["main"]
    assert compose["services"]["main"]["entrypoint"] == []


def test_adapt_moves_compose_main_healthcheck_into_the_workspace(
    tmp_path: Path,
) -> None:
    task = make_harbor_task(tmp_path, "task-a")
    (task / "environment" / "compose.yaml").write_text(
        """\
services:
  main:
    healthcheck:
      test: [CMD-SHELL, curl -f http://localhost:8080/health]
      interval: 2s
      timeout: 3s
      retries: 5
      start_period: 1s
""",
        encoding="utf-8",
    )

    harbor.adapt(tmp_path)

    (context,) = (tmp_path / ".hud-adapt").iterdir()
    manifest = _environment_config(context)
    compose = json.loads((context / "compose-project" / "compose.json").read_text("utf-8"))
    assert manifest["environment"]["healthcheck"] == {
        "command": "curl -f http://localhost:8080/health",
        "interval_sec": 2.0,
        "timeout_sec": 3.0,
        "start_period_sec": 1.0,
        "start_interval_sec": 5.0,
        "retries": 5,
    }
    assert "healthcheck" not in compose["services"]["main"]


def test_adapt_uses_compose_healthcheck_defaults(tmp_path: Path) -> None:
    task = make_harbor_task(tmp_path, "task-a")
    (task / "environment" / "compose.yaml").write_text(
        "services:\n  main:\n    healthcheck:\n      test: [CMD, 'true']\n",
        encoding="utf-8",
    )

    harbor.adapt(tmp_path)

    (context,) = (tmp_path / ".hud-adapt").iterdir()
    manifest = _environment_config(context)
    assert manifest["environment"]["healthcheck"] == {
        "command": "true",
        "interval_sec": 30.0,
        "timeout_sec": 30.0,
        "start_period_sec": 0.0,
        "start_interval_sec": 5.0,
        "retries": 3,
    }


def test_adapt_requires_peer_port_in_the_compose_project(
    tmp_path: Path,
) -> None:
    task = make_harbor_task(tmp_path, "task-a")
    (task / "environment" / "compose.yaml").write_text(
        "services:\n  main: {}\n  redis:\n    image: redis:7-alpine\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="declares no TCP port"):
        harbor.adapt(tmp_path)


def test_adapt_rejects_sidecar_without_a_tcp_port(
    tmp_path: Path,
) -> None:
    task = make_harbor_task(tmp_path, "task-a")
    (task / "environment" / "compose.yaml").write_text(
        "services:\n  main: {}\n  worker:\n    image: no-ports:latest\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="declares no TCP port"):
        harbor.adapt(tmp_path)


def test_network_mcp_servers_become_named_capabilities(
    tmp_path: Path,
) -> None:
    task = make_harbor_task(tmp_path, "task-a")
    (task / "environment" / "compose.yaml").write_text(
        "services:\n  main: {}\n  redis:\n    image: redis:7-alpine\n    expose: [6379]\n",
        encoding="utf-8",
    )
    (task / "task.toml").write_text(
        """
[[environment.mcp_servers]]
name = "redis-tools"
transport = "streamable-http"
url = "http://redis:6379/mcp"
args = []
""",
        encoding="utf-8",
    )

    harbor.adapt(tmp_path)

    (context,) = (tmp_path / ".hud-adapt").iterdir()
    manifest = _environment_config(context)
    assert manifest["capabilities"] == [
        {
            "name": "redis-tools",
            "params": {"transport": "streamable-http"},
            "protocol": "mcp/2025-11-25",
            "url": "http://redis:6379/mcp",
        }
    ]


@pytest.mark.parametrize("name", ["shell", "filetracking"])
def test_mcp_server_names_cannot_shadow_workspace_capabilities(
    tmp_path: Path,
    name: str,
) -> None:
    task = make_harbor_task(tmp_path, "task-a")
    (task / "task.toml").write_text(
        f"""\
[[environment.mcp_servers]]
name = "{name}"
transport = "streamable-http"
url = "http://server:8000/mcp"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=f"MCP server name {name!r} is reserved"):
        harbor.adapt(tmp_path)


def test_adapt_groups_identical_images_and_keeps_row_metadata(
    dataset_same_env: Path,
) -> None:
    (dataset_same_env / "build-pmars" / "task.toml").write_text(
        """\
artifacts = ["/tmp/result"]

[metadata]
category = "systems"
difficulty = "medium"
tags = ["bash", "linux"]

[agent]
timeout_sec = 45

[verifier]
timeout_sec = 30
""",
        encoding="utf-8",
    )
    taskset = harbor.adapt(dataset_same_env)

    assert len(taskset) == 3
    assert len(taskset.environment_names()) == 1
    assert all(
        task.columns
        == {
            "category": "systems",
            "difficulty": "medium",
            "tags": ["bash", "linux"],
        }
        for task in taskset
    )
    assert all(task.runtime_config is not None for task in taskset)
    assert all(task.runtime_config.compose is not None for task in taskset if task.runtime_config)
    assert taskset["build-pmars"].agent_config == {"timeout_seconds": 45.0}
    assert taskset["build-pmars"].args["task"]["verifier_timeout"] == 30.0
    assert taskset["build-pmars"].args["task"]["artifacts"] == [
        {"service": "main", "source": "/tmp/result"}
    ]


def test_distinct_environments_build_distinct_images(
    dataset_multi_env: Path,
) -> None:
    taskset = harbor.adapt(dataset_multi_env)

    assert len(taskset.environment_names()) == 2
    assert all(task.runtime_config is not None for task in taskset)
    assert all(task.runtime_config.compose is not None for task in taskset if task.runtime_config)


def test_adapt_maps_resources_onto_the_compose_runtime(tmp_path: Path) -> None:
    task = make_harbor_task(tmp_path, "gpu")
    (task / "task.toml").write_text(
        """
[metadata]
difficulty = "hard"

[environment]
cpus = 4
memory_mb = 8192
gpus = 2
gpu_types = ["H100"]
""",
        encoding="utf-8",
    )

    (row,) = list(harbor.adapt(tmp_path))

    assert row.columns == {"difficulty": "hard"}
    assert row.runtime_config is not None
    assert row.runtime_config.image is None
    assert isinstance(row.runtime_config.compose, Path)
    assert row.runtime_config.resources is not None
    assert row.runtime_config.resources.cpu == 4
    assert row.runtime_config.resources.memory_mb == 8192
    assert row.runtime_config.resources.gpu is not None
    assert row.runtime_config.resources.gpu.count == 2
    assert row.runtime_config.resources.gpu.type == "H100"


def test_prebuilt_harbor_image_is_inspected_by_the_project_build(
    tmp_path: Path,
) -> None:
    task = make_harbor_task(tmp_path, "prebuilt", dockerfile=None)
    (task / "task.toml").write_text(
        '[environment]\ndocker_image = "registry.example/base:latest"\n',
        encoding="utf-8",
    )

    harbor.adapt(tmp_path)

    (context,) = (tmp_path / ".hud-adapt").iterdir()
    project = context / "compose-project"
    assert (
        (project / "Dockerfile")
        .read_text("utf-8")
        .startswith("FROM registry.example/base:latest AS hud-base\n")
    )
    script = (project / "build.sh").read_text("utf-8")
    assert "docker pull registry.example/base:latest" in script
    assert "inspect_image registry.example/base:latest" in script


def test_zero_gpus_is_a_valid_harbor_resource_declaration(
    tmp_path: Path,
) -> None:
    task = make_harbor_task(tmp_path, "cpu-only")
    (task / "task.toml").write_text("[environment]\ngpus = 0\n", encoding="utf-8")

    (row,) = list(harbor.adapt(tmp_path))

    assert row.runtime_config is not None
    assert row.runtime_config.resources is None


def test_runtime_configuration_is_data_not_dockerfile_codegen(
    tmp_path: Path,
) -> None:
    task = make_harbor_task(tmp_path, "task-a")
    (task / "task.toml").write_text(
        """
[environment]
workdir = "/app"
network_mode = "allowlist"
allowed_hosts = ["pypi.org"]

[environment.env]
SHARED = "yes"

[environment.healthcheck]
command = "curl -f http://localhost:8080/health"
interval_sec = 2
timeout_sec = 4
start_period_sec = 6
start_interval_sec = 1
retries = 5

[agent]
user = "agent"

[agent.env]
AGENT_ONLY = "yes"

[verifier]
user = 0
network_mode = "no-network"

[verifier.env]
VERIFIER_ONLY = "yes"
""",
        encoding="utf-8",
    )

    harbor.adapt(tmp_path)

    (context,) = (tmp_path / ".hud-adapt").iterdir()
    manifest = _environment_config(context)
    assert manifest["workdir"] == "/app"
    assert manifest["environment"] == {
        "env": {"SHARED": "yes"},
        "network_mode": "allowlist",
        "allowed_hosts": ["pypi.org"],
        "healthcheck": {
            "command": "curl -f http://localhost:8080/health",
            "interval_sec": 2.0,
            "timeout_sec": 4.0,
            "start_period_sec": 6.0,
            "start_interval_sec": 1.0,
            "retries": 5,
        },
    }
    assert manifest["agent"]["user"] == "agent"
    assert manifest["agent"]["env"] == {"AGENT_ONLY": "yes"}
    assert manifest["verifier"]["user"] == 0
    assert manifest["verifier"]["network_mode"] == "no-network"
    assert manifest["verifier"]["env"] == {"VERIFIER_ONLY": "yes"}
    dockerfile = (context / "compose-project" / "Dockerfile").read_text("utf-8")
    assert "SHARED" not in dockerfile
    assert "WORKDIR /app" not in dockerfile


def test_image_entrypoint_is_preserved_as_runtime_data(
    tmp_path: Path,
) -> None:
    task = make_harbor_task(tmp_path, "task-a")
    (task / "environment" / "Dockerfile").write_text(
        """\
FROM python:3.12
USER 1000:2000
WORKDIR /workspace
ENV IMAGE_ONLY=present VALUE_WITH_EQUALS=one=two
ENTRYPOINT ["/usr/local/bin/start-environment"]
CMD ["ignored-by-harbor"]
""",
        encoding="utf-8",
    )

    harbor.adapt(tmp_path)

    (context,) = (tmp_path / ".hud-adapt").iterdir()
    manifest = _environment_config(context)
    assert manifest["image_env"] == {}
    assert manifest["entrypoint"] is None
    script = (context / "compose-project" / "build.sh").read_text("utf-8")
    assert "docker image inspect" in script


@pytest.mark.parametrize(
    ("declaration", "expected"),
    [
        ('[environment]\nos = "windows"\n', "os="),
        ('[environment]\ntpu = {type = "v5", topology = "2x2"}\n', "TPUs"),
        (
            '[environment]\ngpus = 1\ngpu_types = ["H100", "A100"]\n',
            "multiple GPU types",
        ),
        ('[environment]\ngpu_types = ["H100"]\n', "GPU types without GPUs"),
        (
            '[[environment.mcp_servers]]\nname = "db"\ntransport = "stdio"\ncommand = "db-mcp"\n',
            "stdio MCP servers",
        ),
    ],
)
def test_unsupported_harbor_behaviour_fails_before_building(
    tmp_path: Path,
    declaration: str,
    expected: str,
) -> None:
    task = make_harbor_task(tmp_path, "task-a")
    (task / "task.toml").write_text(declaration, encoding="utf-8")

    with pytest.raises(NotImplementedError, match=expected):
        harbor.adapt(tmp_path)


@pytest.mark.parametrize("port", [3128, 3129, 8765])
def test_adapt_rejects_main_ports_reserved_by_hud(
    tmp_path: Path,
    port: int,
) -> None:
    task = make_harbor_task(tmp_path, "task-a")
    (task / "environment" / "compose.yaml").write_text(
        f"services:\n  main:\n    expose: [{port}]\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=f"port {port} conflicts with a HUD reserved port"):
        harbor.adapt(tmp_path)


def test_adapt_builds_a_separate_verifier_and_reuses_the_runtime(
    tmp_path: Path,
) -> None:
    task = make_harbor_task(tmp_path, "separate")
    (task / "environment" / "compose.yaml").write_text(
        "services:\n  main: {}\n  redis:\n    image: redis:7-alpine\n    expose: [6379]\n",
        encoding="utf-8",
    )
    (task / "tests" / "Dockerfile").write_text(
        "FROM python:3.12-alpine\nCOPY . /tests\n",
        encoding="utf-8",
    )
    (task / "task.toml").write_text(
        """
artifacts = ["/tmp/agent.patch"]

[environment]
cpus = 2
memory_mb = 2048

[verifier]
environment_mode = "separate"
timeout_sec = 30

[verifier.environment]
cpus = 4
memory_mb = 1024
workdir = "/judge"
network_mode = "allowlist"
allowed_hosts = ["verifier.example"]

[verifier.environment.env]
NESTED_ONLY = "yes"
SHARED = "nested"

[verifier.env]
SHARED = "phase"

[[verifier.collect]]
service = "redis"
command = "redis-cli save"
timeout_sec = 10
""",
        encoding="utf-8",
    )

    (row,) = list(harbor.adapt(tmp_path))

    assert row.id == "run"
    assert row.slug == "separate"
    assert row.verifier == Task(
        env=row.env,
        id="verify",
        args={"task": row.args["task"]},
        slug="separate:verify",
    )
    assert row.runtime_config is not None
    assert row.runtime_config.resources is not None
    assert row.runtime_config.resources.cpu == 4
    assert row.runtime_config.resources.memory_mb == 2048
    assert row.runtime_config.compose_service_access is True

    (context,) = (tmp_path / ".hud-adapt").iterdir()
    manifest = _environment_config(context)
    assert manifest["verifier_root"] == "/media/hud/verifier"
    assert manifest["verifier_image"]["workdir"] == "/judge"
    assert manifest["verifier"] == {
        "user": None,
        "network_mode": "allowlist",
        "allowed_hosts": ["verifier.example"],
        "env": {"NESTED_ONLY": "yes", "SHARED": "phase"},
    }
    assert "tasks" not in manifest
    assert row.args["task"] == {
        "artifacts": [{"service": "main", "source": "/tmp/agent.patch"}],
        "collect": [{"command": "redis-cli save", "service": "redis", "timeout_sec": 10.0}],
        "description": "",
        "id": "separate",
        "separate_verifier": True,
        "verifier_timeout": 30.0,
    }
    assert not (context / "compose-project" / "main" / "tasks").exists()
    compose = json.loads((context / "compose-project" / "compose.json").read_text("utf-8"))
    assert compose["services"]["main"].get("volumes", []) == []
    assert compose["services"]["main"]["build"]["target"] == "verifier"
    assert compose["services"]["main"]["build"]["additional_contexts"] == {
        "hud-base": "service:hud-base",
        "hud-verifier": "service:hud-verifier",
    }
    assert compose["services"]["hud-verifier"]["scale"] == 0
    assert compose["services"]["hud-verifier"]["image"].startswith("hud-harbor-verifier:")


def test_image_task_keeps_only_the_verifier_as_a_build_service(
    tmp_path: Path,
) -> None:
    task = make_harbor_task(tmp_path, "separate-image")
    (task / "tests" / "Dockerfile").write_text(
        "FROM python:3.12-alpine\nCOPY . /tests\n",
        encoding="utf-8",
    )
    (task / "task.toml").write_text(
        '[verifier]\nenvironment_mode = "separate"\n',
        encoding="utf-8",
    )

    (row,) = list(harbor.adapt(tmp_path))

    assert row.runtime_config is not None
    assert row.runtime_config.compose_service_access is True
    assert isinstance(row.runtime_config.compose, Path)
    project = _assert_stock_compose_complete(row.runtime_config.compose)
    assert set(project["services"]) == {"main", "hud-verifier"}
    assert project["services"]["main"]["build"] == {
        "additional_contexts": {
            "hud": "./hud",
            "hud-verifier": "service:hud-verifier",
        },
        "context": "./environment",
        "dockerfile": "../Dockerfile",
    }
    assert project["services"]["hud-verifier"]["scale"] == 0
    combined = (row.runtime_config.compose.parent / "Dockerfile").read_text("utf-8")
    assert "FROM hud-verifier AS hud-verifier-root" in combined
    assert "COPY --from=hud-verifier-root / /media/hud/verifier" in combined


def test_separate_verifier_groups_have_distinct_environment_names(
    tmp_path: Path,
) -> None:
    declaration = '[verifier]\nenvironment_mode = "separate"\n'
    compose = "services:\n  main: {}\n  redis:\n    image: redis:7-alpine\n    expose: [6379]\n"
    verifier = "FROM python:3.12-alpine\nCOPY . /tests\n"
    for name in ("task-a", "task-b"):
        task = make_harbor_task(tmp_path, name)
        (task / "task.toml").write_text(declaration, encoding="utf-8")
        (task / "environment" / "compose.yaml").write_text(compose, encoding="utf-8")
        (task / "tests" / "Dockerfile").write_text(verifier, encoding="utf-8")

    rows = list(harbor.adapt(tmp_path))

    assert len({row.env for row in rows}) == 2
    compose_paths = {
        compose
        for row in rows
        if row.runtime_config is not None
        and isinstance((compose := row.runtime_config.compose), Path)
    }
    assert len(compose_paths) == 2
    assert all(path.is_file() for path in compose_paths)
    task_files = [
        json.loads((path.parents[1] / "tasks.json").read_text("utf-8")) for path in compose_paths
    ]
    assert {rows[0]["slug"] for rows in task_files} == {"task-a", "task-b"}
    assert {rows[0]["args"]["task"]["id"] for rows in task_files} == {"task-a", "task-b"}


def test_multi_step_tasks_are_refused_directly(tmp_path: Path) -> None:
    make_multi_step_task(tmp_path, "multi")

    with pytest.raises(NotImplementedError, match="multi-step"):
        harbor.adapt(tmp_path)


def test_invalid_task_config_is_not_silently_defaulted(
    tmp_path: Path,
) -> None:
    task = make_harbor_task(tmp_path, "task-a")
    (task / "task.toml").write_text("[environment]\ncpus = 'many'\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not a valid Harbor task"):
        harbor.adapt(tmp_path)


@pytest.mark.parametrize("source", ["/", "//", "/workspace/../secret"])
def test_artifacts_must_name_normalized_paths_beneath_root(
    tmp_path: Path,
    source: str,
) -> None:
    task = make_harbor_task(tmp_path, "task-a")
    (task / "task.toml").write_text(f'artifacts = ["{source}"]\n', encoding="utf-8")

    with pytest.raises(ValueError, match="artifact source must name a path beneath /"):
        harbor.adapt(tmp_path)


def test_agent_timeout_becomes_per_task_agent_policy(
    tmp_path: Path,
) -> None:
    task = make_harbor_task(tmp_path, "task-a")
    (task / "task.toml").write_text("[agent]\ntimeout_sec = 60\n", encoding="utf-8")

    taskset = harbor.adapt(tmp_path)

    (row,) = list(taskset)
    assert row.agent_config == {"timeout_seconds": 60.0}


def test_task_symlinks_are_copied_without_reading_host_files(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("host secret", encoding="utf-8")
    task = make_harbor_task(tmp_path / "dataset", "task-a")
    (task / "tests" / "link").symlink_to(outside)

    harbor.adapt(task.parent)

    (context,) = (task.parent / ".hud-adapt").iterdir()
    copied = context / "compose-project" / "tests" / "task-a" / "link"
    assert copied.is_symlink()
    assert os.readlink(copied) == str(outside)


def test_adapt_hashes_links_not_their_targets(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("first", encoding="utf-8")
    task = make_harbor_task(tmp_path / "dataset", "task-a")
    (task / "environment" / "link").symlink_to(outside)

    (before,) = list(harbor.adapt(task.parent))
    outside.write_text("changed", encoding="utf-8")
    (after,) = list(harbor.adapt(task.parent))

    assert before.runtime_config == after.runtime_config


def test_authored_runtime_assets_are_valid_source() -> None:
    integration = Path(__file__).parents[1]
    compile((integration / "env.py").read_text("utf-8"), "env.py", "exec")
    installer = (integration / "install.sh").read_text("utf-8")
    assert "python_version=3.12" in installer
    assert "sys.version_info[:2] < (3, 13)" in installer
    assert 'uv python install "$python_version"' in installer
    assert 'python="$root/bin/python$python_version"' in installer
    assert 'uv venv "$root/venv" --python "$python"' in installer
    result = subprocess.run(
        ["sh", "-n", integration / "install.sh"],
        check=False,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr.decode()


def test_public_surface_is_only_the_two_real_operations() -> None:
    assert harbor.__all__ == ["adapt", "export"]
