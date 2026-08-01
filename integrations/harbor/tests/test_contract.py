"""Observable contracts for adapting Harbor tasks into HUD images."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from integrations import harbor

from .conftest import make_harbor_task, make_multi_step_task


@pytest.fixture(autouse=True)
def fake_docker(monkeypatch):
    calls: list[tuple[str, ...]] = []

    async def run(*args: str, **_kwargs):
        calls.append(args)
        if args[:3] == ("image", "inspect", "--format"):
            return json.dumps({"User": "", "WorkingDir": "/workspace"}), ""
        return "", ""

    module = importlib.import_module("integrations.harbor.adapt")
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
    }
    assert manifest["agent"]["user"] == "agent"
    assert manifest["agent"]["env"] == {"AGENT_ONLY": "yes"}
    assert manifest["verifier"]["user"] == 0
    assert manifest["verifier"]["network_mode"] == "no-network"
    assert manifest["verifier"]["env"] == {"VERIFIER_ONLY": "yes"}
    dockerfile = (context / "Dockerfile").read_text("utf-8")
    assert "SHARED" not in dockerfile
    assert "WORKDIR /app" not in dockerfile


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
        ('[environment.healthcheck]\ncommand = "curl localhost"\n', "healthcheck"),
        ('[[environment.mcp_servers]]\nname = "db"\n', "MCP servers"),
        ('[verifier]\nenvironment_mode = "separate"\n', "separate verifier"),
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
    result = subprocess.run(
        ["sh", "-n", integration / "install.sh"],
        check=False,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr.decode()


def test_public_surface_is_only_the_two_real_operations() -> None:
    assert harbor.__all__ == ["adapt", "export"]
