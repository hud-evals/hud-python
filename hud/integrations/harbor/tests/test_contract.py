"""Observable contracts for adapting Harbor tasks into HUD images."""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from hud.eval import Task
from hud.integrations import harbor

from .conftest import make_harbor_task, make_multi_step_task


@pytest.fixture(autouse=True)
def fake_docker(monkeypatch):
    calls: list[tuple[str, ...]] = []

    async def run(*args: str, **_kwargs):
        calls.append(args)
        if args[:4] == ("image", "inspect", "--format", "{{json .Config}}"):
            if args[-1] == "no-ports:latest":
                return json.dumps(
                    {
                        "User": "",
                        "WorkingDir": "/workspace",
                        "Entrypoint": None,
                        "Cmd": ["serve"],
                    }
                ), ""
            if args[-1].startswith("hud-harbor:"):
                return json.dumps(
                    {
                        "User": "",
                        "WorkingDir": "/workspace",
                        "Entrypoint": [],
                        "Cmd": ["/media/hud/venv/bin/hud", "serve", "/media/hud/env.py"],
                    }
                ), ""
            if args[-1].startswith("hud-harbor-base:"):
                return json.dumps(
                    {
                        "User": "",
                        "WorkingDir": "/workspace",
                        "Entrypoint": None,
                        "Cmd": None,
                    }
                ), ""
            return json.dumps(
                {
                    "User": "",
                    "WorkingDir": "/workspace",
                    "Entrypoint": None,
                    "Cmd": None,
                    "ExposedPorts": {"6379/tcp": {}},
                }
            ), ""
        if args[:4] == ("image", "inspect", "--format", "{{.Id}}"):
            return "sha256:0123456789abcdef0123456789abcdef\n", ""
        if args[0] == "compose" and args[-3:] == ("config", "--format", "json"):
            project = {
                "name": "task",
                "services": {
                    "main": {
                        "image": "hud-main",
                        "environment": {"FROM_COMPOSE": "yes"},
                        "expose": ["8080/tcp"],
                        "healthcheck": {
                            "test": ["CMD-SHELL", "curl -f http://localhost:8080/health"],
                            "interval": "2s",
                            "timeout": "3s",
                            "retries": 5,
                            "start_period": "1s",
                        },
                        "depends_on": {
                            "redis": {
                                "condition": "service_healthy",
                                "required": True,
                            }
                        },
                        "networks": {"default": None},
                    },
                    "redis": {
                        "image": "redis:7-alpine",
                        "entrypoint": None,
                        "command": ["redis-server", "--save", ""],
                        "environment": {"SIDE": "car"},
                        "expose": ["6379/tcp"],
                        "healthcheck": {
                            "test": ["CMD", "redis-cli", "ping"],
                            "interval": "2s",
                            "timeout": "3s",
                            "retries": 5,
                            "start_period": "1s",
                        },
                        "networks": {"default": None},
                    },
                },
                "networks": {"default": {"name": "task_default"}},
            }
            authored_file = Path(args[-4])
            authored = await asyncio.to_thread(authored_file.read_text, "utf-8")
            if "dockerfile: Containerfile" in authored:
                project["services"] = {
                    "main": {
                        "image": "hud-main",
                        "build": {
                            "context": str(authored_file.parent),
                            "dockerfile": "Containerfile",
                            "args": {"FLAVOR": "compose"},
                        },
                        "environment": {"FROM_COMPOSE": "yes"},
                        "networks": {"default": None},
                    }
                }
            elif "no-ports:latest" in authored:
                project["services"]["main"].pop("depends_on")
                project["services"].pop("redis")
                project["services"]["worker"] = {
                    "image": "no-ports:latest",
                    "networks": {"default": None},
                }
            elif "main-healthcheck" in authored:
                project["services"]["main"]["healthcheck"] = {
                    "test": ["CMD-SHELL", "curl -f http://localhost:8080/health"],
                    "interval": "2s",
                    "timeout": "3s",
                    "retries": 5,
                    "start_period": "1s",
                }
            elif "healthcheck-defaults" in authored:
                project["services"]["main"]["healthcheck"] = {
                    "test": ["CMD", "true"],
                }
            for port in (3128, 3129, 8765):
                if f"expose: [{port}]" in authored:
                    project["services"]["main"]["expose"] = [f"{port}/tcp"]
            if "expose:" not in authored and "redis" in project["services"]:
                project["services"]["redis"].pop("expose")
            if "user: 1001:1002" in authored:
                project["services"]["main"]["user"] = "1001:1002"
                project["services"]["main"]["entrypoint"] = ["/compose-init"]
                project["services"]["main"]["working_dir"] = "/compose-work"
            return json.dumps(project), ""
        if args[0] == "compose" and args[-2] == "build":
            compose_file = Path(args[args.index("--file") + 1])
            compose = await asyncio.to_thread(compose_file.read_text, "utf-8")
            calls.append(("compose-build-config", args[-1], compose))
        return "", ""

    module = importlib.import_module("hud.integrations.harbor.adapt")
    monkeypatch.setattr(module, "docker", run)
    return calls


async def test_adapt_builds_the_source_then_an_authored_hud_environment(
    tmp_path: Path,
    fake_docker,
) -> None:
    make_harbor_task(tmp_path, "task-a")

    taskset = await harbor.adapt(tmp_path)

    (task,) = list(taskset)
    assert task.id == "task-a"
    assert task.runtime_config is not None
    assert task.runtime_config.image is not None
    assert task.runtime_config.image.startswith("hud-harbor:")

    builds = [call for call in fake_docker if call[0] == "build"]
    assert len(builds) == 2
    assert "BASE_IMAGE=hud-harbor-base:" in " ".join(builds[1])

    (context,) = (tmp_path / ".hud-adapt").iterdir()
    integration = Path(__file__).parents[1]
    for asset in ("Dockerfile", "install.sh"):
        assert (context / asset).read_bytes() == (integration / asset).read_bytes()
    # env.py names the environment as a literal — `hud deploy` resolves the
    # context's identity from source, and refuses a computed name.
    served = (context / "env.py").read_text(encoding="utf-8")
    assert f'Environment("{context.name}")' in served
    assert 'Environment(CONFIG["name"])' not in served
    assert (context / "tasks" / "task-a" / "instruction.md").is_file()
    assert (context / "tasks" / "task-a" / "tests" / "test.sh").is_file()
    assert not (context / "compose.json").exists()


async def test_adapt_honors_compose_main_build_settings(
    tmp_path: Path,
    fake_docker,
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

    await harbor.adapt(tmp_path)

    compose_build = next(call for call in fake_docker if call[:2] == ("compose", "--file"))
    assert compose_build[-2:] == ("build", "main")
    _, _, serialized = next(call for call in fake_docker if call[0] == "compose-build-config")
    main = json.loads(serialized)["services"]["main"]
    assert main["build"]["dockerfile"] == "Containerfile"
    assert main["build"]["args"] == {"FLAVOR": "compose"}
    assert main["image"].startswith("hud-harbor-base:")
    wrapper_build = next(call for call in fake_docker if call[0] == "build")
    assert any(value.startswith("BASE_IMAGE=hud-harbor-base:") for value in wrapper_build)


async def test_adapt_emits_compose_with_pinned_sidecars_and_peers(
    tmp_path: Path,
    fake_docker,
) -> None:
    task = make_harbor_task(tmp_path, "task-a")
    (task / "environment" / "compose.yaml").write_text(
        "services:\n  main: {}\n  redis:\n    image: redis:7-alpine\n    expose: [6379]\n",
        encoding="utf-8",
    )

    (row,) = list(await harbor.adapt(tmp_path))

    assert row.runtime_config is not None
    assert row.runtime_config.image is None
    (context,) = (tmp_path / ".hud-adapt").iterdir()
    assert row.runtime_config.compose == context / "compose.json"
    compose = json.loads((context / "compose.json").read_text("utf-8"))
    redis = compose["services"]["redis"]
    assert redis["image"].startswith("hud-harbor-sidecar:")
    assert redis["image"].endswith("-0123456789abcdef")
    assert redis["environment"] == {"SIDE": "car"}
    assert redis["command"] == ["redis-server", "--save", ""]
    assert redis["expose"] == ["6379/tcp"]
    assert redis["healthcheck"]["test"] == ["CMD", "redis-cli", "ping"]
    assert redis["entrypoint"] == []
    assert redis["working_dir"] == "/workspace"
    assert redis["networks"] == {"default": None}
    assert "build" not in redis
    manifest = json.loads((context / "tasks.json").read_text("utf-8"))
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
    assert compose["services"]["main"]["command"] == [
        "/media/hud/venv/bin/hud",
        "serve",
        "/media/hud/env.py",
    ]
    assert "healthcheck" not in compose["services"]["main"]
    assert ("pull", "redis:7-alpine") in fake_docker
    assert any(call[:2] == ("tag", "redis:7-alpine") for call in fake_docker)


async def test_adapt_moves_compose_main_process_settings_into_the_workspace(
    tmp_path: Path,
    fake_docker,
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

    await harbor.adapt(tmp_path)

    (context,) = (tmp_path / ".hud-adapt").iterdir()
    manifest = json.loads((context / "tasks.json").read_text("utf-8"))
    compose = json.loads((context / "compose.json").read_text("utf-8"))
    assert manifest["image_user"] == "1001:1002"
    assert manifest["entrypoint"] == ["/compose-init"]
    assert manifest["workdir"] == "/compose-work"
    assert "user" not in compose["services"]["main"]
    assert compose["services"]["main"]["entrypoint"] == []


async def test_adapt_moves_compose_main_healthcheck_into_the_workspace(
    tmp_path: Path,
    fake_docker,
) -> None:
    task = make_harbor_task(tmp_path, "task-a")
    (task / "environment" / "compose.yaml").write_text(
        "# main-healthcheck\nservices:\n  main: {}\n",
        encoding="utf-8",
    )

    await harbor.adapt(tmp_path)

    (context,) = (tmp_path / ".hud-adapt").iterdir()
    manifest = json.loads((context / "tasks.json").read_text("utf-8"))
    compose = json.loads((context / "compose.json").read_text("utf-8"))
    assert manifest["environment"]["healthcheck"] == {
        "command": "curl -f http://localhost:8080/health",
        "interval_sec": 2.0,
        "timeout_sec": 3.0,
        "start_period_sec": 1.0,
        "start_interval_sec": 5.0,
        "retries": 5,
    }
    assert "healthcheck" not in compose["services"]["main"]


async def test_adapt_uses_compose_healthcheck_defaults(tmp_path: Path) -> None:
    task = make_harbor_task(tmp_path, "task-a")
    (task / "environment" / "compose.yaml").write_text(
        "# healthcheck-defaults\nservices:\n  main: {}\n",
        encoding="utf-8",
    )

    await harbor.adapt(tmp_path)

    (context,) = (tmp_path / ".hud-adapt").iterdir()
    manifest = json.loads((context / "tasks.json").read_text("utf-8"))
    assert manifest["environment"]["healthcheck"] == {
        "command": "true",
        "interval_sec": 30.0,
        "timeout_sec": 30.0,
        "start_period_sec": 0.0,
        "start_interval_sec": 5.0,
        "retries": 3,
    }


async def test_adapt_derives_implicit_peer_port_from_image(
    tmp_path: Path,
    fake_docker,
) -> None:
    task = make_harbor_task(tmp_path, "task-a")
    (task / "environment" / "compose.yaml").write_text(
        "services:\n  main: {}\n  redis:\n    image: redis:7-alpine\n",
        encoding="utf-8",
    )

    await harbor.adapt(tmp_path)

    (context,) = (tmp_path / ".hud-adapt").iterdir()
    manifest = json.loads((context / "tasks.json").read_text("utf-8"))
    assert manifest["peers"] == [{"name": "redis", "port": 6379}]


async def test_adapt_rejects_sidecar_without_a_tcp_port(
    tmp_path: Path,
    fake_docker,
) -> None:
    task = make_harbor_task(tmp_path, "task-a")
    (task / "environment" / "compose.yaml").write_text(
        "services:\n  main: {}\n  worker:\n    image: no-ports:latest\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="declares no TCP port"):
        await harbor.adapt(tmp_path)


async def test_network_mcp_servers_become_named_capabilities(
    tmp_path: Path,
    fake_docker,
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

    await harbor.adapt(tmp_path)

    (context,) = (tmp_path / ".hud-adapt").iterdir()
    manifest = json.loads((context / "tasks.json").read_text("utf-8"))
    assert manifest["capabilities"] == [
        {
            "name": "redis-tools",
            "params": {"transport": "streamable-http"},
            "protocol": "mcp/2025-11-25",
            "url": "http://redis:6379/mcp",
        }
    ]


@pytest.mark.parametrize("name", ["shell", "filetracking"])
async def test_mcp_server_names_cannot_shadow_workspace_capabilities(
    tmp_path: Path,
    fake_docker,
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
        await harbor.adapt(tmp_path)

    assert fake_docker == []


async def test_adapt_groups_identical_images_and_keeps_row_metadata(
    dataset_same_env: Path,
    fake_docker,
) -> None:
    taskset = await harbor.adapt(dataset_same_env)

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
    assert len([call for call in fake_docker if call[0] == "build"]) == 2


async def test_distinct_environments_build_distinct_images(
    dataset_multi_env: Path,
    fake_docker,
) -> None:
    taskset = await harbor.adapt(dataset_multi_env)

    assert len(taskset.environment_names()) == 2
    assert len([call for call in fake_docker if call[0] == "build"]) == 4


async def test_adapt_maps_resources_and_pushes_the_images(tmp_path: Path, fake_docker) -> None:
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

    (row,) = list(await harbor.adapt(tmp_path, push="registry.example/hud"))

    assert row.columns == {"difficulty": "hard"}
    assert row.runtime_config is not None
    assert row.runtime_config.image is not None
    assert row.runtime_config.image.startswith("registry.example/hud/")
    assert row.runtime_config.resources is not None
    assert row.runtime_config.resources.cpu == 4
    assert row.runtime_config.resources.memory_mb == 8192
    assert row.runtime_config.resources.gpu is not None
    assert row.runtime_config.resources.gpu.count == 2
    assert row.runtime_config.resources.gpu.type == "H100"
    assert any(call[0] == "push" for call in fake_docker)


async def test_prebuilt_harbor_image_skips_the_source_build(tmp_path: Path, fake_docker) -> None:
    task = make_harbor_task(tmp_path, "prebuilt", dockerfile=None)
    (task / "task.toml").write_text(
        '[environment]\ndocker_image = "registry.example/base:latest"\n',
        encoding="utf-8",
    )

    await harbor.adapt(tmp_path)

    builds = [call for call in fake_docker if call[0] == "build"]
    assert len(builds) == 1
    assert "BASE_IMAGE=registry.example/base:latest" in builds[0]
    assert ("pull", "registry.example/base:latest") in fake_docker


async def test_zero_gpus_is_a_valid_harbor_resource_declaration(
    tmp_path: Path,
    fake_docker,
) -> None:
    task = make_harbor_task(tmp_path, "cpu-only")
    (task / "task.toml").write_text("[environment]\ngpus = 0\n", encoding="utf-8")

    (row,) = list(await harbor.adapt(tmp_path))

    assert row.runtime_config is not None
    assert row.runtime_config.resources is None


async def test_runtime_configuration_is_data_not_dockerfile_codegen(
    tmp_path: Path,
    fake_docker,
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

    await harbor.adapt(tmp_path)

    (context,) = (tmp_path / ".hud-adapt").iterdir()
    manifest = json.loads((context / "tasks.json").read_text("utf-8"))
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
    dockerfile = (context / "Dockerfile").read_text("utf-8")
    assert "SHARED" not in dockerfile
    assert "WORKDIR /app" not in dockerfile


async def test_image_entrypoint_is_preserved_as_runtime_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    make_harbor_task(tmp_path, "task-a")

    async def docker(*args: str, **_kwargs):
        if args[:3] == ("image", "inspect", "--format"):
            return (
                json.dumps(
                    {
                        "User": "1000:2000",
                        "WorkingDir": "/workspace",
                        "Env": ["IMAGE_ONLY=present", "VALUE_WITH_EQUALS=one=two"],
                        "Entrypoint": ["/usr/local/bin/start-environment"],
                        "Cmd": ["ignored-by-harbor"],
                    }
                ),
                "",
            )
        return "", ""

    module = importlib.import_module("hud.integrations.harbor.adapt")
    monkeypatch.setattr(module, "docker", docker)

    await harbor.adapt(tmp_path)

    (context,) = (tmp_path / ".hud-adapt").iterdir()
    manifest = json.loads((context / "tasks.json").read_text("utf-8"))
    assert manifest["image_env"] == {
        "IMAGE_ONLY": "present",
        "VALUE_WITH_EQUALS": "one=two",
    }
    assert manifest["entrypoint"] == ["/usr/local/bin/start-environment"]
    assert "ignored-by-harbor" not in json.dumps(manifest)


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
async def test_unsupported_harbor_behaviour_fails_before_building(
    tmp_path: Path,
    fake_docker,
    declaration: str,
    expected: str,
) -> None:
    task = make_harbor_task(tmp_path, "task-a")
    (task / "task.toml").write_text(declaration, encoding="utf-8")

    with pytest.raises(NotImplementedError, match=expected):
        await harbor.adapt(tmp_path)

    assert fake_docker == []


@pytest.mark.parametrize("port", [3128, 3129, 8765])
async def test_adapt_rejects_main_ports_reserved_by_hud(
    tmp_path: Path,
    fake_docker,
    port: int,
) -> None:
    task = make_harbor_task(tmp_path, "task-a")
    (task / "environment" / "compose.yaml").write_text(
        f"services:\n  main:\n    expose: [{port}]\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=f"port {port} conflicts with a HUD reserved port"):
        await harbor.adapt(tmp_path)


async def test_adapt_builds_a_separate_verifier_and_reuses_the_runtime(
    tmp_path: Path,
    fake_docker,
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

    (row,) = list(await harbor.adapt(tmp_path))

    assert row.verifier == Task(env=row.env, id="separate:verify")
    assert row.runtime_config is not None
    assert row.runtime_config.resources is not None
    assert row.runtime_config.resources.cpu == 4
    assert row.runtime_config.resources.memory_mb == 2048
    assert row.runtime_config.compose_service_access is True

    (context,) = (tmp_path / ".hud-adapt").iterdir()
    manifest = json.loads((context / "tasks.json").read_text("utf-8"))
    assert manifest["verifier_root"] == "/media/hud/verifier"
    assert manifest["verifier_image"]["workdir"] == "/judge"
    assert manifest["verifier"] == {
        "user": None,
        "network_mode": "allowlist",
        "allowed_hosts": ["verifier.example"],
        "env": {"NESTED_ONLY": "yes", "SHARED": "phase"},
    }
    assert manifest["tasks"] == [
        {
            "artifacts": [{"service": "main", "source": "/tmp/agent.patch"}],
            "collect": [{"command": "redis-cli save", "service": "redis", "timeout_sec": 10.0}],
            "description": "",
            "id": "separate",
            "separate_verifier": True,
            "verifier_timeout": 30.0,
        }
    ]
    assert not (context / "tasks" / "separate" / "tests").exists()
    compose = json.loads((context / "compose.json").read_text("utf-8"))
    assert compose["services"]["main"].get("volumes", []) == []
    wrapper = [call for call in fake_docker if call[0] == "build"][-1]
    assert wrapper[1:3] == ("--target", "verifier")
    assert any(value.startswith("VERIFIER_IMAGE=hud-harbor-verifier:") for value in wrapper)


async def test_separate_verifier_groups_have_distinct_environment_names(
    tmp_path: Path,
    fake_docker,
) -> None:
    declaration = '[verifier]\nenvironment_mode = "separate"\n'
    compose = "services:\n  main: {}\n  redis:\n    image: redis:7-alpine\n    expose: [6379]\n"
    verifier = "FROM python:3.12-alpine\nCOPY . /tests\n"
    for name in ("task-a", "task-b"):
        task = make_harbor_task(tmp_path, name)
        (task / "task.toml").write_text(declaration, encoding="utf-8")
        (task / "environment" / "compose.yaml").write_text(compose, encoding="utf-8")
        (task / "tests" / "Dockerfile").write_text(verifier, encoding="utf-8")

    rows = list(await harbor.adapt(tmp_path))

    assert len({row.env for row in rows}) == 2
    compose_paths = {
        row.runtime_config.compose
        for row in rows
        if row.runtime_config is not None and row.runtime_config.compose is not None
    }
    assert len(compose_paths) == 2
    assert all(path.is_file() for path in compose_paths)
    manifests = [
        json.loads((path.parent / "tasks.json").read_text("utf-8")) for path in compose_paths
    ]
    assert {manifest["tasks"][0]["id"] for manifest in manifests} == {"task-a", "task-b"}


async def test_multi_step_tasks_are_refused_directly(tmp_path: Path, fake_docker) -> None:
    make_multi_step_task(tmp_path, "multi")

    with pytest.raises(NotImplementedError, match="multi-step"):
        await harbor.adapt(tmp_path)

    assert fake_docker == []


async def test_invalid_task_config_is_not_silently_defaulted(
    tmp_path: Path,
    fake_docker,
) -> None:
    task = make_harbor_task(tmp_path, "task-a")
    (task / "task.toml").write_text("[environment]\ncpus = 'many'\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not a valid Harbor task"):
        await harbor.adapt(tmp_path)

    assert fake_docker == []


@pytest.mark.parametrize("source", ["/", "//", "/workspace/../secret"])
async def test_artifacts_must_name_normalized_paths_beneath_root(
    tmp_path: Path,
    fake_docker,
    source: str,
) -> None:
    task = make_harbor_task(tmp_path, "task-a")
    (task / "task.toml").write_text(f'artifacts = ["{source}"]\n', encoding="utf-8")

    with pytest.raises(ValueError, match="artifact source must name a path beneath /"):
        await harbor.adapt(tmp_path)

    assert fake_docker == []


async def test_agent_timeout_becomes_per_task_agent_policy(
    tmp_path: Path,
    fake_docker,
) -> None:
    task = make_harbor_task(tmp_path, "task-a")
    (task / "task.toml").write_text("[agent]\ntimeout_sec = 60\n", encoding="utf-8")

    taskset = await harbor.adapt(tmp_path)

    (row,) = list(taskset)
    assert row.agent_config == {"timeout_seconds": 60.0}


async def test_task_symlinks_are_copied_without_reading_host_files(
    tmp_path: Path,
    fake_docker,
) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("host secret", encoding="utf-8")
    task = make_harbor_task(tmp_path / "dataset", "task-a")
    (task / "tests" / "link").symlink_to(outside)

    await harbor.adapt(task.parent)

    (context,) = (task.parent / ".hud-adapt").iterdir()
    copied = context / "tasks" / "task-a" / "tests" / "link"
    assert copied.is_symlink()
    assert os.readlink(copied) == str(outside)


async def test_adapt_hashes_links_not_their_targets(
    tmp_path: Path,
    fake_docker,
) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("first", encoding="utf-8")
    task = make_harbor_task(tmp_path / "dataset", "task-a")
    (task / "environment" / "link").symlink_to(outside)

    (before,) = list(await harbor.adapt(task.parent))
    outside.write_text("changed", encoding="utf-8")
    (after,) = list(await harbor.adapt(task.parent))

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
