"""The in-image Harbor controller, adapted from fixture tasks and run in Docker.

Each directory under ``tasks/`` is a Harbor task with a reference
``solution/solve.sh`` and an optional ``expected.toml``. The scenario adapts
the task against this checkout's wheel and runs it under ``DockerRuntime`` with
an oracle that only runs the solution, so the verifier's reward proves the
controller kept each phase's contract. ``expected.toml`` keys, all optional,
at the top level or per ``[[case]]`` (a case adds to the top level):

- ``reward``: the reward every run must get (default ``1.0``);
- ``error``: text the run's error or evaluation must contain;
- ``repeat``: run the solution in this many SSH sessions;
- ``group`` and ``shared_width``: rollouts per task, sharing placements;
- ``capability_probe``: ``{name, tools}`` an MCP capability opened host-side lists;
- ``runtime.env_vars`` and ``runtime.bind_mounts`` (``source`` relative to the
  fixture): what ``DockerRuntime`` gives the container;
- ``startup_error``: the controller must refuse to serve, with this message;
- ``base``: another fixture this one starts from (``assemble_case``), so a
  variant holds only what it changes.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from hud.agents.base import Agent
from hud.capabilities import MCPClient, SSHClient, find_capability, open_client
from hud.integrations import harbor
from hud.runtime import DockerBindMount, DockerRuntime, Shared
from tests.integrations.harbor.cases import assemble_case, read_expected

if TYPE_CHECKING:
    from hud import Run, Trace
    from hud.runtime import Provider

TASKS = Path(__file__).parent / "tasks"

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.docker,
    pytest.mark.skipif(sys.platform == "win32", reason="adapted images are Linux containers"),
]


@dataclass(frozen=True)
class Case:
    fixture: str
    name: str
    reward: float
    error: str | None
    repeat: int
    group: int | None
    shared_width: int | None
    probe: dict[str, Any] | None
    env_vars: dict[str, str]
    bind_mounts: tuple[DockerBindMount, ...]
    startup_error: str | None

    @property
    def id(self) -> str:
        return f"{self.fixture}[{self.name}]" if self.name else self.fixture


def load_cases() -> list[Case]:
    cases = []
    fixtures = (path for path in TASKS.iterdir() if path.is_dir() and path.name != "__pycache__")
    for root in sorted(fixtures):
        spec = read_expected(root)
        spec.pop("base", None)
        for variant in spec.pop("case", None) or [{}]:
            merged = {**spec, **variant}
            runtime = {**spec.get("runtime", {}), **variant.get("runtime", {})}
            cases.append(
                Case(
                    fixture=root.name,
                    name=merged.get("name", ""),
                    reward=merged.get("reward", 1.0),
                    error=merged.get("error"),
                    repeat=merged.get("repeat", 1),
                    group=merged.get("group"),
                    shared_width=merged.get("shared_width"),
                    probe=merged.get("capability_probe"),
                    env_vars=runtime.get("env_vars", {}),
                    bind_mounts=tuple(
                        DockerBindMount(root / mount["source"], mount["target"])
                        for mount in runtime.get("bind_mounts", [])
                    ),
                    startup_error=merged.get("startup_error"),
                )
            )
    return cases


CASES = load_cases()
RUNNABLE = [case for case in CASES if case.startup_error is None]
REFUSED = [case for case in CASES if case.startup_error is not None]


class Oracle(Agent):
    """Run the fixture's reference solution; its stdout is the answer."""

    def __init__(self, solution: str, *, repeat: int, probe: str | None) -> None:
        super().__init__()
        self.solution = solution
        self.repeat = repeat
        self.probe = probe
        self.probed: list[list[str]] = []

    async def __call__(self, run: Run) -> None:
        if self.probe is not None:
            async with open_client(find_capability(run.capabilities, self.probe)) as mcp:
                assert isinstance(mcp, MCPClient)
                self.probed.append(sorted(tool.name for tool in await mcp.list_tools()))
        stdout = ""
        async with open_client(find_capability(run.capabilities, "ssh")) as ssh:
            assert isinstance(ssh, SSHClient)
            for session in range(self.repeat):
                result = await ssh.run(self.solution, check=False)
                if result.exit_status != 0:
                    raise RuntimeError(
                        f"solution session {session} exited with {result.exit_status}:\n"
                        f"{result.stdout}\n{result.stderr}"
                    )
                stdout = str(result.stdout)
        run.trace.content = stdout


def adapted(case: Case, sdk_wheel: Path, tmp_path: Path) -> harbor.AdaptResult:
    dataset = tmp_path / "harbor-fixtures"
    assemble_case(TASKS / case.fixture, dataset / case.fixture, "managed")
    result = harbor.adapt(dataset, hud_requirement=str(sdk_wheel))
    assert result.failures == ()
    return result


def report(trace: Trace) -> str:
    """Everything a failed trace says about why, for assertions and their messages."""
    evaluation = trace.evaluation
    info = evaluation.get("info") or {}
    return "\n".join(
        str(part)
        for part in (
            trace.content,
            trace.error,
            evaluation.get("content"),
            info.get("stdout"),
            info.get("stderr"),
        )
        if part
    )


@pytest.mark.parametrize("case", RUNNABLE, ids=[case.id for case in RUNNABLE])
async def test_a_harbor_task_grades_as_its_fixture_expects(
    case: Case, sdk_wheel: Path, tmp_path: Path
) -> None:
    taskset = adapted(case, sdk_wheel, tmp_path).taskset
    docker = DockerRuntime(env_vars=case.env_vars, bind_mounts=case.bind_mounts)
    runtime: Provider = Shared(docker, width=case.shared_width) if case.shared_width else docker
    oracle = Oracle(
        (tmp_path / "harbor-fixtures" / case.fixture / "solution" / "solve.sh").read_text("utf-8"),
        repeat=case.repeat,
        probe=None if case.probe is None else case.probe["name"],
    )

    job = await taskset.run(oracle, runtime=runtime, group=case.group, max_concurrent=1)

    assert len(job.traces) == (case.group or 1)
    for trace in job.traces:
        detail = report(trace)
        assert trace.reward == case.reward, detail
        if case.error is not None:
            assert case.error in detail
    if case.probe is not None:
        assert oracle.probed == [sorted(case.probe["tools"])] * len(job.traces)


@pytest.mark.parametrize("case", REFUSED, ids=[case.id for case in REFUSED])
def test_a_controller_with_an_unusable_configuration_refuses_to_serve(
    case: Case, sdk_wheel: Path, tmp_path: Path
) -> None:
    # Providers yield once the control port is published, before the serve
    # process proves itself, so the refusal is observed by running the main
    # service of the adapted project in the foreground.
    (row,) = adapted(case, sdk_wheel, tmp_path).taskset
    assert row.runtime_config is not None and row.runtime_config.compose is not None
    compose = row.runtime_config.compose.document
    assert isinstance(compose, Path)

    def compose_command(*args: str, timeout: float) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["docker", "compose", "--file", str(compose), *args],
            cwd=compose.parent,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

    try:
        built = compose_command("build", timeout=1200)
        assert built.returncode == 0, built.stderr[-4000:]
        served = compose_command("run", "--rm", "main", timeout=120)
    finally:
        compose_command("down", "--volumes", "--remove-orphans", timeout=120)

    assert served.returncode != 0
    assert case.startup_error is not None
    assert case.startup_error in served.stdout + served.stderr
