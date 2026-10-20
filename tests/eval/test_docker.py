"""DockerRuntime: a row runs in a confined container or Compose project, removed after.

The scenarios run real containers from the fixture image (this checkout's SDK
serving a fixture environment) on the local Docker daemon, and check what the
daemon reports about them. What every runtime does (outcomes, deadlines, cancel,
cleanup, session files for a verifier, refusals) runs on Docker as a row of the
rollout tables; this module holds what only Docker does.
"""

from __future__ import annotations

import io
import json
import subprocess
import tarfile
import textwrap
import uuid
from typing import TYPE_CHECKING, Any

import pytest

from hud.eval import (
    ComposeProject,
    DockerRuntime,
    RuntimeConfig,
    RuntimeLimits,
    RuntimeResources,
    Task,
    rollout,
)
from hud.eval.runtime import DockerBindMount
from hud.eval.runtime.docker import DockerEndpoint
from tests.fixtures.agents import ScriptedAgent
from tests.support import eventually

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from tests.support import HudEnv

ECHO = Task(env="echo", id="echo", args={"word": "tangerine"})


def inspect(container: str) -> dict[str, Any]:
    out = subprocess.run(
        ["docker", "inspect", container], capture_output=True, text=True, check=True
    ).stdout
    return json.loads(out)[0]


def containers(**labels: str) -> list[str]:
    filters = [arg for key, value in labels.items() for arg in ("--filter", f"label={key}={value}")]
    out = subprocess.run(
        ["docker", "ps", "--all", "--quiet", *filters], capture_output=True, text=True, check=True
    ).stdout
    return out.split()


def volume_exists(name: str) -> bool:
    probe = subprocess.run(["docker", "volume", "inspect", name], capture_output=True, check=False)
    return probe.returncode == 0


def labelled() -> tuple[str, list[str]]:
    """A label unique to this test and the ``docker run`` flags that set it."""
    label = uuid.uuid4().hex
    return label, ["--label", f"hud-test={label}"]


@pytest.mark.e2e
@pytest.mark.docker
async def test_an_image_row_runs_in_a_confined_container_that_is_removed_after(
    fixture_image: Callable[[str], str],
) -> None:
    label, flags = labelled()

    async with DockerRuntime(fixture_image("echo"), run_args=flags)(ECHO) as runtime:
        assert isinstance(runtime, DockerEndpoint)
        held = inspect(runtime.container)
        run = await rollout(ECHO, ScriptedAgent("tangerine"), runtime=runtime)

    sessions = [mount["Name"] for mount in held["Mounts"] if mount["Type"] == "volume"]
    assert run.reward == 1.0
    assert any(option.startswith("seccomp=") for option in held["HostConfig"]["SecurityOpt"])
    assert held["HostConfig"]["Privileged"] is False
    assert containers(**{"hud-test": label}) == []
    assert sessions and not any(volume_exists(name) for name in sessions)


@pytest.mark.e2e
@pytest.mark.docker
async def test_the_rows_settings_win_over_the_providers_and_reach_the_container(
    fixture_image: Callable[[str], str], tmp_path: Path
) -> None:
    (tmp_path / "data").mkdir()
    _, flags = labelled()
    provider = DockerRuntime(
        "hud-test-fixture:missing",
        run_args=flags,
        env_vars={"MODE": "eval"},
        bind_mounts=[DockerBindMount(str(tmp_path / "data"), "/data")],
        runtime_config=RuntimeConfig(resources=RuntimeResources(cpu=2, memory_mb=1024)),
    )
    row = ECHO.model_copy(
        update={
            "runtime_config": RuntimeConfig(
                image=fixture_image("echo"), resources=RuntimeResources(cpu=1, memory_mb=512)
            )
        }
    )

    async with provider(row) as runtime:
        assert isinstance(runtime, DockerEndpoint)
        held = inspect(runtime.container)
        run = await rollout(row, ScriptedAgent("tangerine"), runtime=runtime)

    (data,) = [mount for mount in held["Mounts"] if mount["Destination"] == "/data"]
    assert run.reward == 1.0
    assert (held["HostConfig"]["NanoCpus"], held["HostConfig"]["Memory"]) == (10**9, 512 * 2**20)
    assert "MODE=eval" in held["Config"]["Env"]
    assert data["RW"] is False


@pytest.mark.e2e
@pytest.mark.docker
async def test_a_row_that_names_an_image_runs_there_without_a_runtime(
    fixture_image: Callable[[str], str],
) -> None:
    row = ECHO.model_copy(update={"runtime_config": RuntimeConfig(image=fixture_image("echo"))})

    job = await row.run(ScriptedAgent("tangerine"))

    assert job.reward == 1.0


@pytest.mark.e2e
@pytest.mark.docker
async def test_more_disk_than_the_container_has_is_refused_and_the_container_removed(
    fixture_image: Callable[[str], str],
) -> None:
    label, flags = labelled()
    row = ECHO.model_copy(
        update={"runtime_config": RuntimeConfig(resources=RuntimeResources(storage_mb=10**9))}
    )

    with pytest.raises(RuntimeError, match=r"requires 1000000000 MB of free disk"):
        async with DockerRuntime(fixture_image("echo"), run_args=flags)(row):
            pytest.fail("the request should be refused")

    assert containers(**{"hud-test": label}) == []


@pytest.mark.e2e
@pytest.mark.docker
async def test_a_container_past_its_run_limit_is_removed_while_held(
    fixture_image: Callable[[str], str],
) -> None:
    label, flags = labelled()
    row = ECHO.model_copy(
        update={"runtime_config": RuntimeConfig(limits=RuntimeLimits(run_timeout_s=2))}
    )

    async with DockerRuntime(fixture_image("echo"), run_args=flags)(row):
        await eventually(lambda: containers(**{"hud-test": label}) == [])


@pytest.mark.e2e
@pytest.mark.docker
async def test_a_restored_session_archive_may_hold_only_files_and_directories(
    fixture_image: Callable[[str], str], tmp_path: Path
) -> None:
    archive = tmp_path / "session.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        link = tarfile.TarInfo("escape")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        tar.addfile(link, io.BytesIO())

    async with DockerRuntime(fixture_image("echo"))(ECHO) as runtime:
        with pytest.raises(
            ValueError, match="runtime session archive contains an unsupported entry"
        ):
            await runtime.restore_session("sess-12345678", archive)


def compose_file(tmp_path: Path, document: str, env_file: str = "") -> ComposeProject:
    (tmp_path / ".env").write_text(env_file)
    path = tmp_path / "compose.yaml"
    path.write_text(textwrap.dedent(document))
    return ComposeProject(document=path)


@pytest.mark.e2e
@pytest.mark.docker
async def test_a_compose_project_serves_main_confined_through_its_network_owner(
    fixture_image: Callable[[str], str], tmp_path: Path, hud_env: HudEnv
) -> None:
    hud_env.set(HOST_ONLY="from-the-host")
    project = compose_file(
        tmp_path,
        f"""
        services:
          main:
            image: {fixture_image("echo")}
            network_mode: service:gateway
            environment:
              MODE: ${{MODE}}
              LITERAL: $$HOST_ONLY
            depends_on: [gateway]
          gateway:
            image: python:3.12-slim
            command: sleep infinity
            healthcheck: {{test: [CMD, "true"], interval: 1s}}
        """,
        env_file="MODE=eval\n",
    )
    row = ECHO.model_copy(update={"runtime_config": RuntimeConfig(compose=project)})

    async with DockerRuntime()(row) as runtime:
        assert isinstance(runtime, DockerEndpoint)
        held = inspect(runtime.container)
        run = await rollout(row, ScriptedAgent("tangerine"), runtime=runtime)

    compose = held["Config"]["Labels"]["com.docker.compose.project"]
    assert run.reward == 1.0
    assert any(option.startswith("seccomp=") for option in held["HostConfig"]["SecurityOpt"])
    assert {"MODE=eval", "LITERAL=$HOST_ONLY"} <= set(held["Config"]["Env"])
    assert containers(**{"com.docker.compose.project": compose}) == []


@pytest.mark.e2e
@pytest.mark.docker
async def test_a_compose_project_whose_network_owner_dies_is_taken_down(
    fixture_image: Callable[[str], str], tmp_path: Path
) -> None:
    project = compose_file(
        tmp_path,
        f"""
        services:
          main:
            image: {fixture_image("echo")}
            network_mode: service:gateway
          gateway:
            image: python:3.12-slim
            command: python -c "print('gateway gave up'); raise SystemExit(1)"
        """,
    )
    row = ECHO.model_copy(update={"runtime_config": RuntimeConfig(compose=project)})

    with pytest.raises(RuntimeError):
        async with DockerRuntime()(row):
            pytest.fail("the project should not serve")

    working_dir = {"com.docker.compose.project.working_dir": str(tmp_path)}
    assert containers(**working_dir) == []
