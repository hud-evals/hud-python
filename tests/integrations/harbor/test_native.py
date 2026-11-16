"""Harbor tasks run natively: each in its authored image, with Harbor's lifecycle run here.

The Docker tests need a daemon and network (``pytest -m e2e``).
"""

from __future__ import annotations

import asyncio
import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import ANY

import pytest

from hud.agents.base import Agent
from hud.capabilities import find_capability, open_client
from hud.eval import file_tracking
from hud.eval.run import connect
from hud.integrations import harbor
from hud.integrations.harbor.load import KEEP_ALIVE
from hud.runtime import (
    EnvRef,
    Runtime,
    RuntimeBuild,
    RuntimeConfig,
    RuntimeHealthcheck,
    RuntimeLimits,
    RuntimePort,
    RuntimeProcess,
    RuntimeResources,
)
from hud.runtime.compose import ComposeConfig
from hud.settings import settings
from tests.integrations.harbor.cases import assemble_case

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Mapping, Sequence

    from hud.capabilities import SSHClient
    from hud.eval.task import Task
    from hud.types import Trace

TASK_TOML = """\
[environment]
build_timeout_sec = 900
cpus = 2
memory_mb = 4096

[environment.env]
SHARED = "baseline"
TOKEN = "${HOST_TOKEN:-none}"

[environment.healthcheck]
command = "test -f /ready"
interval_sec = 0.5
timeout_sec = 5
retries = 25

[agent]
user = "agent"
timeout_sec = 30

[agent.env]
AGENT_ONLY = "secret"

[verifier.env]
JUDGE = "${JUDGE_KEY}"
"""

MCP_TOML = """\
[[environment.mcp_servers]]
name = "tools"
transport = "sse"
url = "http://tools:8000/sse"
"""


def make_harbor_task(
    parent: Path,
    name: str,
    *,
    instruction: str = "Solve the task.",
    task_toml: str = "",
    dockerfile: str | None = "FROM python:3.12-slim\nWORKDIR /workspace\n",
) -> Path:
    """Write a Harbor task directory under ``parent`` whose test always passes; return it."""
    task = parent / name
    (task / "tests").mkdir(parents=True)
    (task / "instruction.md").write_text(instruction, encoding="utf-8")
    (task / "task.toml").write_text(task_toml, encoding="utf-8")
    if dockerfile is not None:
        (task / "environment").mkdir()
        (task / "environment" / "Dockerfile").write_text(dockerfile, encoding="utf-8")
    (task / "tests" / "test.sh").write_text(
        "#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n", encoding="utf-8"
    )
    return task


def _files(root: Path) -> dict[str, bytes]:
    return {
        entry.relative_to(root).as_posix(): entry.read_bytes() if entry.is_file() else b""
        for entry in sorted(root.rglob("*"))
    }


def test_load_reads_each_task_into_its_placement_and_reports_the_rest(tmp_path: Path) -> None:
    dataset = tmp_path / "harbor-native"
    plain = make_harbor_task(dataset, "plain", task_toml=TASK_TOML)
    make_harbor_task(
        dataset,
        "prebuilt",
        task_toml='[environment]\ndocker_image = "python:3.12-slim"\n',
        dockerfile=None,
    )
    composed = make_harbor_task(dataset, "composed", task_toml=MCP_TOML)
    (composed / "environment" / "docker-compose.yaml").write_text(
        "services:\n  main: {}\n  tools:\n    image: tools:1\n", encoding="utf-8"
    )
    separate = make_harbor_task(
        dataset, "separate", task_toml='[verifier]\nenvironment_mode = "separate"\n'
    )
    (separate / "tests" / "Dockerfile").write_text("FROM python:3.11-slim\n", encoding="utf-8")
    make_harbor_task(dataset, "networked", task_toml='[agent]\nnetwork_mode = "no-network"\n')
    before = _files(dataset)

    result = harbor.load(dataset)

    assert _files(dataset) == before
    rows = {row.id: row for row in result.taskset}
    assert rows.keys() == {"plain", "prebuilt", "composed", "separate"}
    row = rows["plain"]
    assert isinstance(row, harbor.HarborTask)
    assert row.path == plain
    assert row.env.startswith("harbor-native-")
    assert row.agent_config == {"timeout_seconds": 30.0}
    assert row.verifier is None
    assert row.runtime_config == RuntimeConfig(
        image=RuntimeBuild(context=plain / "environment"),
        process=RuntimeProcess(
            command=KEEP_ALIVE,
            env={"SHARED": "baseline", "TOKEN": EnvRef(name="HOST_TOKEN", default="none")},
            healthcheck=RuntimeHealthcheck(
                command="test -f /ready", interval_s=0.5, timeout_s=5, retries=25
            ),
        ),
        resources=RuntimeResources(cpu=2, memory_mb=4096),
        limits=RuntimeLimits(startup_timeout_s=900),
    )
    prebuilt = rows["prebuilt"].runtime_config
    assert prebuilt is not None
    assert prebuilt.image == "python:3.12-slim"

    # A Compose task runs its authored project, main built from the environment's Dockerfile,
    # with its MCP server published from the service that serves it.
    composed_row = rows["composed"]
    assert isinstance(composed_row, harbor.HarborTask)
    placement = composed_row.runtime_config
    assert placement is not None
    assert placement.image is None
    assert placement.compose is not None
    assert placement.compose.root == composed / "environment"
    recipe = placement.compose.document
    assert isinstance(recipe, ComposeConfig)
    assert recipe.services["main"].build == {"context": "."}
    assert recipe.services["tools"].image == "tools:1"
    assert placement.process is not None
    assert placement.process.command == KEEP_ALIVE
    assert composed_row.mcp_ports == {"tools": RuntimePort(port=8000, service="tools")}
    assert placement.ports == [RuntimePort(port=8000, service="tools")]

    # A separate verifier is graded in a box of its own, built from tests/.
    verifier = rows["separate"].verifier
    assert verifier is not None
    assert verifier.runtime_config == RuntimeConfig(
        image=RuntimeBuild(context=separate / "tests"),
        process=RuntimeProcess(command=KEEP_ALIVE),
    )
    assert {
        failure.task: [finding.code for finding in failure.findings] for failure in result.failures
    } == {"networked": ["harbor.unsupported.native_network_policy"]}


def test_a_harbor_task_needs_a_runtime_it_can_run_commands_in(tmp_path: Path) -> None:
    (task,) = harbor.load(make_harbor_task(tmp_path, "plain")).taskset

    with pytest.raises(TypeError, match="needs a runtime it can run commands in"):
        connect(task, Runtime("tcp://127.0.0.1:8765"))


def _is_session(argv: Sequence[str]) -> bool:
    """Whether ``argv`` is an agent session: a command under the box's login shell."""
    return argv[:2] == ["sh", "-c"] and "exec bash -l" in argv[2] and argv[-2] == "-c"


class _Box:
    """A stand-in box: the agent's shells run on this machine, the verifier runs ``verifier``,
    and every other command only collects what it is sent."""

    url = "box://stand-in"

    def __init__(self, root: Path, *, verifier: str, writes: Mapping[str, str]) -> None:
        self.received = root / "received"
        self.received.touch()
        self.verifier = verifier
        self.writes = dict(writes)
        self.calls: list[dict[str, Any]] = []
        self.puts: list[tuple[Path, str]] = []

    def exec_argv(
        self,
        argv: Sequence[str],
        *,
        user: str | None = None,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        tty: bool = False,
    ) -> list[str]:
        self.calls.append({"argv": list(argv), "user": user, "cwd": cwd, "env": dict(env or {})})
        if _is_session(argv):
            # Run the session's command here, outside any login profile.
            variables = [f"{key}={value}" for key, value in (env or {}).items()]
            return ["env", *variables, "sh", "-c", argv[-1]]
        if argv[-1] == "/tests/test.sh":
            return ["sh", "-c", self.verifier]
        return ["sh", "-c", 'cat >> "$1"', "sh", str(self.received)]

    async def put(self, source: Path, target: str) -> None:
        self.puts.append((source, target))

    async def get(self, source: str, target: Path) -> None:
        assert source == "/logs/verifier"
        await asyncio.to_thread(self._write_logs, target)

    def _write_logs(self, target: Path) -> None:
        target.mkdir(parents=True)
        for name, content in self.writes.items():
            (target / name).write_text(content, encoding="utf-8")

    def service(self, name: str) -> _Box:
        assert name == "main"
        return self

    def address(self, port: int) -> str:
        raise AssertionError(f"the stand-in publishes no ports, not {port}")


class _Solver(Agent):
    """Run one shell script through the task's shell and answer with its output."""

    def __init__(self, script: str) -> None:
        super().__init__()
        self.script = script

    async def __call__(self, run: Any) -> None:
        async with open_client(find_capability(run.capabilities, "ssh/2")) as client:
            result = await cast("SSHClient", client).run(self.script)
        run.trace.content = (
            result.stdout
            if result.returncode == 0
            else f"solution failed:\n{result.stdout}\n{result.stderr}"
        )


async def _run_on(box: _Box, task_dir: Path, agent: Agent) -> Trace:
    @asynccontextmanager
    async def place(task: Task) -> AsyncGenerator[Runtime]:
        yield cast("Runtime", box)

    loaded = harbor.load(task_dir)
    assert loaded.failures == ()
    job = await loaded.taskset.run(agent, runtime=place)
    (trace,) = job.traces
    return trace


@pytest.mark.parametrize(
    ("verifier", "writes", "reward", "error"),
    [
        ("echo verified", {"reward.txt": "1\n"}, 1.0, None),
        ("echo verified", {"reward.json": '{"reward": 0.5}', "reward.txt": "1"}, 0.5, None),
        ("echo verified", {}, 0.0, "did not write a numeric reward"),
        ("sleep 5", {"reward.txt": "1"}, 0.0, "timed out after 1s"),
    ],
)
async def test_a_rollout_serves_the_agent_a_shell_then_grades_with_the_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    verifier: str,
    writes: dict[str, str],
    reward: float,
    error: str | None,
) -> None:
    monkeypatch.setenv("AGENT_KEY", "agent-secret")
    monkeypatch.setenv("JUDGE_KEY", "judge-secret")
    task = make_harbor_task(
        tmp_path,
        "plain",
        instruction="Serve /app on port 9000.",
        task_toml=(
            '[environment]\nworkdir = "/app"\n\n'
            '[agent]\nuser = "agent"\n\n[agent.env]\nAGENT_ONLY = "${AGENT_KEY}"\n\n'
            '[verifier]\nuser = "judge"\ntimeout_sec = 1\n\n'
            '[verifier.env]\nJUDGE = "${JUDGE_KEY}"\n'
        ),
    )
    (task / "tests" / "test.sh").write_text("#!/bin/bash -eu\necho graded\n", encoding="utf-8")
    box = _Box(tmp_path, verifier=verifier, writes=writes)

    trace = await _run_on(box, task, _Solver('printf %s "$AGENT_ONLY"'))

    assert trace.prompt_text == "Serve /app on port 9000."
    (shell,) = [call for call in box.calls if _is_session(call["argv"])]
    assert (shell["user"], shell["cwd"]) == ("agent", "/app")
    assert shell["env"]["AGENT_ONLY"] == "agent-secret"
    assert box.received.read_text("utf-8") == "agent-secret"
    assert box.puts == [(task / "tests", "/tests")]
    (verified,) = [call for call in box.calls if call["argv"][-1] == "/tests/test.sh"]
    assert verified == {
        "argv": ["/bin/bash", "-eu", "/tests/test.sh"],
        "user": "judge",
        "cwd": "/app",
        "env": {"JUDGE": "judge-secret"},
    }
    assert trace.reward == reward
    assert trace.grade.is_error is (error is not None)
    if error is not None:
        assert error in (trace.grade.content or "")


async def test_a_missing_verifier_reference_fails_the_task_before_the_agent_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("JUDGE_KEY", raising=False)
    task = make_harbor_task(tmp_path, "plain", task_toml='[verifier.env]\nJUDGE = "${JUDGE_KEY}"\n')
    box = _Box(tmp_path, verifier="echo verified", writes={"reward.txt": "1"})

    trace = await _run_on(box, task, _Solver("true"))

    assert trace.status == "error"
    assert "env 'JUDGE' reads 'JUDGE_KEY', which is not set here" in (trace.error or "")
    assert box.calls == []


DOCKERFILE = """\
FROM python:3.11-slim
RUN groupadd --gid 2000 worker && useradd --create-home --uid 1000 --gid 2000 agent
WORKDIR /app
ENV IMAGE_ONLY=present
RUN chown agent:worker /app
COPY entrypoint.sh /usr/local/bin/start-environment
RUN chmod +x /usr/local/bin/start-environment
ENTRYPOINT ["/usr/local/bin/start-environment"]
"""

ENTRYPOINT = """\
#!/bin/sh
set -eu
mkdir -p /tmp/startup
printf '%s\\n' "${SHARED:-missing}" > /tmp/startup/status
python3 -m http.server 8080 --directory /tmp/startup >/tmp/startup.log 2>&1 &
exec "$@"
"""

DOCKER_TASK_TOML = """\
[environment.env]
SHARED = "baseline"

[environment.healthcheck]
command = "python3 -c \\"import urllib.request as u; u.urlopen('http://127.0.0.1:8080/status')\\""
interval_sec = 0.2
timeout_sec = 5
retries = 25

[agent]
user = "agent:worker"

[agent.env]
AGENT_ONLY = "secret"

[verifier]
timeout_sec = 60

[verifier.env]
JUDGE = "${JUDGE_KEY}"
"""

SOLUTION = """\
set -eu
id -u > identity.txt
id -g >> identity.txt
printf '%s %s %s %s\\n' "$IMAGE_ONLY" "$SHARED" "$AGENT_ONLY" "${JUDGE-unset}" > env.txt
python3 -m http.server 9000 --directory /app >/dev/null 2>&1 &
sleep 1
echo solution completed
"""

TEST_SH = """\
#!/bin/bash
set -u
mkdir -p /logs/verifier
fail() { echo "$1"; echo 0 > /logs/verifier/reward.txt; exit 0; }
[ "$PWD" = /app ] || fail "the verifier ran in $PWD"
[ "$(cat /app/identity.txt)" = "$(printf '1000\\n2000')" ] \\
  || fail "agent identity: $(cat /app/identity.txt)"
[ "$(cat /app/env.txt)" = "present baseline secret unset" ] || fail "agent env: $(cat /app/env.txt)"
[ "${JUDGE:-}" = "judge-secret" ] || fail "the verifier lost its secret"
[ -z "${AGENT_ONLY+x}" ] || fail "the verifier inherited the agent environment"
[ "$(cat /logs/agent_answer.txt)" = "solution completed" ] || fail "the verifier lost the answer"
python3 -c "import urllib.request as u; u.urlopen('http://127.0.0.1:9000/env.txt')" \\
  || fail "the agent's service did not survive into grading"
echo 1 > /logs/verifier/reward.txt
"""


@pytest.mark.e2e
@pytest.mark.docker
@pytest.mark.skipif(sys.platform == "win32", reason="Harbor images are Linux containers")
async def test_a_harbor_task_runs_in_its_authored_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("JUDGE_KEY", "judge-secret")
    monkeypatch.setattr(settings, "telemetry_enabled", True)
    monkeypatch.setattr(settings, "file_tracking_enabled", True)
    spans: list[tuple[str, dict[str, Any]]] = []

    def record(name: str, payload: dict[str, Any], **_: Any) -> bool:
        spans.append((name, payload))
        return True

    monkeypatch.setattr(file_tracking, "_emit_file_tracking", record)
    dataset = tmp_path / "harbor-native"
    task = make_harbor_task(dataset, "native", task_toml=DOCKER_TASK_TOML, dockerfile=DOCKERFILE)
    (task / "environment" / "entrypoint.sh").write_text(ENTRYPOINT, encoding="utf-8")
    (task / "tests" / "test.sh").write_text(TEST_SH, encoding="utf-8")
    loaded = harbor.load(dataset)
    assert loaded.failures == ()

    job = await loaded.taskset.run(_Solver(SOLUTION), max_concurrent=1)

    (trace,) = job.traces
    info = trace.evaluation.get("info") or {}
    detail = "\n".join(
        filter(None, (trace.content, trace.error, info.get("stdout"), info.get("stderr")))
    )
    assert trace.reward == 1.0, detail
    assert ("filetracking.snapshot", {"root": ANY, "files": [], "files_scanned": 0}) in spans
    changes = {
        patch["path"]: patch["status"]
        for name, payload in spans
        if name == "filetracking.diff"
        for patch in payload["patches"]
    }
    assert changes == {"identity.txt": "added", "env.txt": "added"}


TASKS = Path(__file__).parent / "tasks"


@pytest.mark.e2e
@pytest.mark.docker
@pytest.mark.skipif(sys.platform == "win32", reason="Harbor images are Linux containers")
@pytest.mark.parametrize(
    "fixture",
    [
        "agent-lifecycle",
        "artifacts-dir-separate",
        "artifacts-dir-shared",
        "hello-mcp",
        "sidecar-reachability",
    ],
)
async def test_the_adapted_fixtures_pass_natively(tmp_path: Path, fixture: str) -> None:
    dataset = tmp_path / "harbor-native"
    task = dataset / fixture
    assemble_case(TASKS / fixture, task)
    if fixture == "agent-lifecycle":
        # Its solution and test use tools the adapted image installs; a native box has only
        # what the task's own image does.
        dockerfile = task / "environment" / "Dockerfile"
        dockerfile.write_text(
            dockerfile.read_text("utf-8").replace(
                "FROM debian:bookworm-slim\n",
                "FROM debian:bookworm-slim\n"
                "RUN apt-get update && apt-get install -y --no-install-recommends curl python3\n",
            ),
            encoding="utf-8",
        )
    solve = task / "solution" / "solve.sh"
    # The adapted network's egress proxy sets NO_PROXY; a public native box has no proxy.
    solution = re.sub(r'case ",\$\{NO_PROXY.*?esac\n', "", solve.read_text("utf-8"), flags=re.S)
    loaded = harbor.load(dataset)
    assert loaded.failures == ()

    job = await loaded.taskset.run(_Solver(solution), max_concurrent=1)

    (trace,) = job.traces
    info = trace.evaluation.get("info") or {}
    detail = "\n".join(
        filter(None, (trace.content, trace.error, info.get("stdout"), info.get("stderr")))
    )
    assert trace.reward == 1.0, detail
