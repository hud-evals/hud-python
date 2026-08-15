"""Docker coverage for Harbor phase behavior.

The tests build adapted images and require network access::

    uv run pytest -m integration
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from hud.agents.base import Agent
from hud.eval import DockerRuntime, Shared, Taskset
from hud.integrations import harbor

from .conftest import make_harbor_task

if TYPE_CHECKING:
    from hud.capabilities import MCPClient, SSHClient
    from hud.eval.run import Run

TASKS = Path(__file__).parent / "tasks"
REPO = Path(__file__).resolve().parents[4]

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(sys.platform == "win32", reason="adapted images are Linux containers"),
]


def _adapt(path: Path, *, hud_requirement: str = "hud") -> Taskset:
    result = harbor.adapt(path, hud_requirement=hud_requirement)
    assert result.failures == ()
    return result.taskset


@pytest.fixture(scope="module", autouse=True)
def docker_daemon() -> None:
    if shutil.which("docker") is None:
        pytest.skip("needs a running Docker daemon")
    try:
        probe = subprocess.run(["docker", "info"], capture_output=True, check=False, timeout=20)
    except subprocess.TimeoutExpired:
        pytest.skip("Docker daemon did not answer")
    if probe.returncode != 0:
        pytest.skip("needs a running Docker daemon")


class Oracle(Agent):
    """Run each fixture's reference solution."""

    def __init__(self, solutions: dict[str, str]) -> None:
        self.solutions = solutions

    async def __call__(self, run: Run) -> None:
        setup = next(
            step.task_call
            for step in run.trace.steps
            if step.task_call is not None and step.task_call.phase == "setup"
        )
        assert isinstance(setup.arguments, dict)
        task = setup.arguments.get("task")
        assert isinstance(task, dict)
        task_id = task.get("id")
        assert isinstance(task_id, str)
        ssh = cast("SSHClient", await run.client.open("ssh/2"))
        if task_id == "agent-lifecycle":
            for index in range(40):
                result = await ssh.conn.run(f"printf '%s' {index}", check=False)
                assert result.exit_status == 0, f"command session {index} failed: {result.stderr!r}"
                assert result.stdout == str(index), f"command session {index} lost output"
            await ssh.conn.run("cat > session-input", input="written", check=True)
            written = await ssh.conn.run("cat session-input", check=True)
            assert written.stdout == "written"
        if task_id == "hello-mcp":
            mcp = cast("MCPClient", await run.client.open("mcp-server"))
            assert {tool.name for tool in await mcp.list_tools()} == {"get_secret"}
            assert run.client.manifest is not None
            capability = next(
                cap for cap in run.client.manifest.bindings if cap.name == "mcp-server"
            )
            script = f"""
import json
from pathlib import Path
from urllib.request import Request, urlopen

url = {capability.url!r}
session_id = None

def post(payload):
    global session_id
    headers = {{
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }}
    if session_id is not None:
        headers["Mcp-Session-Id"] = session_id
    request = Request(url, data=json.dumps(payload).encode(), headers=headers)
    with urlopen(request) as response:
        body = response.read().decode()
        returned_session = response.headers.get("Mcp-Session-Id")
        if returned_session:
            session_id = returned_session
    data = [line[6:] for line in body.splitlines() if line.startswith("data: ")]
    return json.loads(data[-1] if data else body) if body else None

post({{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {{
        "protocolVersion": "2025-06-18",
        "capabilities": {{}},
        "clientInfo": {{"name": "hud-test", "version": "1"}},
    }},
}})
post({{"jsonrpc": "2.0", "method": "notifications/initialized"}})
result = post({{
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {{"name": "get_secret", "arguments": {{}}}},
}})
Path("/app/secret.txt").write_text(result["result"]["content"][0]["text"])
"""
            result = await ssh.conn.run("python3 -", input=script, check=True)
            if result.exit_status is None:
                raise RuntimeError("workspace MCP client closed without an exit status")
            run.trace.content = "called get_secret from inside the workspace"
            return
        result = await ssh.conn.run(self.solutions[task_id], check=False)
        if result.exit_status is None:
            run.trace.content = (
                "solution SSH channel closed without an exit status:\n"
                f"{result.stdout}\n{result.stderr}"
            )
            raise RuntimeError("solution SSH channel closed without an exit status")
        if result.exit_status != 0:
            run.trace.content = f"solution failed:\n{result.stdout}\n{result.stderr}"
            raise RuntimeError(f"solution exited with status {result.exit_status}")
        run.trace.content = "solution completed"


async def _grade_every_task(dataset: Path, wheel: Path) -> dict[str, Run]:
    def load_solutions() -> dict[str, str]:
        return {
            task.name: (task / "solution" / "solve.sh").read_text("utf-8")
            for task in sorted(dataset.iterdir())
            if (task / "task.toml").is_file()
        }

    solutions = await asyncio.to_thread(load_solutions)
    taskset = _adapt(dataset, hud_requirement=str(wheel))
    job = await taskset.run(
        Oracle(solutions),
        runtime=DockerRuntime(),
        max_concurrent=1,
    )
    return {cast("str", run.slug): run for run in job.runs}


@pytest.fixture(scope="module")
def wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    workdir = tmp_path_factory.mktemp("harbor")
    wheels = workdir / "wheels"
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(wheels)],
        cwd=REPO,
        check=True,
        capture_output=True,
    )
    return next(iter(wheels.glob("*.whl")))


@pytest.fixture(scope="module")
def graded(tmp_path_factory: pytest.TempPathFactory, wheel: Path) -> dict[str, Run]:
    """Adapt and grade the inline-verifier fixtures once for this module."""
    dataset = tmp_path_factory.mktemp("harbor-inline") / "harbor-harness"
    shutil.copytree(TASKS, dataset, ignore=shutil.ignore_patterns("sidecar-reachability"))
    return asyncio.run(_grade_every_task(dataset, wheel))


@pytest.fixture(scope="module")
def separately_graded(tmp_path_factory: pytest.TempPathFactory, wheel: Path) -> dict[str, Run]:
    """Adapt and grade the separate-verifier fixture."""
    dataset = tmp_path_factory.mktemp("harbor-separate") / "harbor-harness"
    dataset.mkdir()
    task = dataset / "sidecar-reachability"
    shutil.copytree(TASKS / "sidecar-reachability", task)
    declaration = task / "task.toml"
    declaration.write_text(
        declaration.read_text("utf-8") + "\n[verifier.environment]\ncpus = 1\n",
        encoding="utf-8",
    )
    return asyncio.run(_grade_every_task(dataset, wheel))


@pytest.mark.parametrize(
    "task_id",
    [
        "phase-boundary",
        "agent-lifecycle",
        "verifier-lifecycle",
        "hello-mcp",
    ],
)
def test_harbor_phase_behavior(graded: dict[str, Run], task_id: str) -> None:
    run = graded[task_id]
    evaluation = run.evaluation
    detail = evaluation.get("content") or ""
    info = evaluation.get("info") or {}
    detail = "\n".join(
        filter(
            None,
            (
                run.trace.content,
                run.trace.error,
                detail,
                info.get("stdout"),
                info.get("stderr"),
            ),
        )
    )
    assert run.reward == 1.0, f"{task_id} scored {run.reward}; the verifier reported:\n{detail}"


def test_separate_verifier_phase_behavior(separately_graded: dict[str, Run]) -> None:
    test_harbor_phase_behavior(separately_graded, "sidecar-reachability")


def test_sidecar_ports_are_discovered_from_the_built_image(
    tmp_path_factory: pytest.TempPathFactory,
    wheel: Path,
) -> None:
    dataset = tmp_path_factory.mktemp("harbor-sidecar-ports") / "harbor-harness"
    task = dataset / "sidecar-reachability"
    shutil.copytree(TASKS / "sidecar-reachability", task)
    compose = task / "environment/docker-compose.yaml"
    authored = compose.read_text("utf-8")
    inferred = authored.replace(
        "    image: ${SIDECAR_IMAGE:-python:3.11-alpine}\n",
        "    build: ./workspace\n",
    ).replace('    expose: ["5678", "5679"]\n', "")
    assert inferred != authored
    compose.write_text(inferred, encoding="utf-8")
    sidecar = task / "environment/workspace"
    sidecar.mkdir()
    (sidecar / "Dockerfile").write_text(
        "FROM python:3.11-alpine\nEXPOSE 5678 5679\n",
        encoding="utf-8",
    )

    runs = asyncio.run(_grade_every_task(dataset, wheel))

    test_harbor_phase_behavior(runs, "sidecar-reachability")


def test_separate_verifier_artifact_materialization_is_repeatable(
    tmp_path_factory: pytest.TempPathFactory, wheel: Path
) -> None:
    dataset = tmp_path_factory.mktemp("harbor-verifier-artifacts") / "harbor-harness"
    task = dataset / "sidecar-reachability"
    shutil.copytree(TASKS / "sidecar-reachability", task)

    async def grade_twice() -> list[Run]:
        taskset = _adapt(dataset, hud_requirement=str(wheel))
        job = await taskset.run(
            Oracle({"sidecar-reachability": (task / "solution/solve.sh").read_text("utf-8")}),
            runtime=Shared(DockerRuntime(), width=1),
            group=2,
            max_concurrent=1,
        )
        return job.runs

    runs = asyncio.run(grade_twice())

    assert len(runs) == 2
    assert all(run.reward == 1.0 for run in runs)


def test_separate_verifier_artifacts_are_accessible_to_child_user(
    tmp_path_factory: pytest.TempPathFactory, wheel: Path
) -> None:
    dataset = tmp_path_factory.mktemp("harbor-child-user") / "harbor-harness"
    task = dataset / "sidecar-reachability"
    shutil.copytree(TASKS / "sidecar-reachability", task)
    dockerfile = task / "tests/Dockerfile"
    dockerfile.write_text(
        dockerfile.read_text("utf-8").replace("USER verifier\n", "USER root\n"),
        encoding="utf-8",
    )
    (task / "tests/test.sh").write_text(
        "#!/bin/sh\n"
        "set -eu\n"
        "mkdir -p /logs/verifier\n"
        "if su verifier -s /bin/sh -c 'test \"$(cat /root/agent-output.txt)\" = private'; then\n"
        "  echo 1 > /logs/verifier/reward.txt\n"
        "else\n"
        "  echo 0 > /logs/verifier/reward.txt\n"
        "fi\n",
        encoding="utf-8",
    )

    run = asyncio.run(_grade_every_task(dataset, wheel))["sidecar-reachability"]

    assert run.reward == 1.0


def test_fedora_environment_bootstrap(
    tmp_path_factory: pytest.TempPathFactory, wheel: Path
) -> None:
    dataset = tmp_path_factory.mktemp("harbor-fedora") / "harbor-harness"
    task = dataset / "fedora-bootstrap"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "solution").mkdir()
    (task / "task.toml").write_text(
        '[task]\nname = "fedora-bootstrap"\n\n[verifier]\ntimeout_sec = 30\n',
        encoding="utf-8",
    )
    (task / "instruction.md").write_text("Verify Fedora bootstrap.\n", encoding="utf-8")
    (task / "environment/Dockerfile").write_text(
        "FROM fedora:42\nWORKDIR /app\n",
        encoding="utf-8",
    )
    (task / "tests/test.sh").write_text(
        "#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n",
        encoding="utf-8",
    )
    (task / "solution/solve.sh").write_text("true\n", encoding="utf-8")

    run = asyncio.run(_grade_every_task(dataset, wheel))["fedora-bootstrap"]

    assert run.reward == 1.0


def test_separate_verifier_rejects_artifact_symlinks(
    tmp_path_factory: pytest.TempPathFactory, wheel: Path
) -> None:
    dataset = tmp_path_factory.mktemp("harbor-symlink") / "harbor-harness"
    task = dataset / "sidecar-reachability"
    shutil.copytree(TASKS / "sidecar-reachability", task)
    solution = task / "solution" / "solve.sh"
    solution.write_text(
        solution.read_text("utf-8")
        + "\nrm -f /app/main.html\n"
        + "ln -s /media/hud/verifier/tests /app/main.html\n",
        encoding="utf-8",
    )

    run = asyncio.run(_grade_every_task(dataset, wheel))["sidecar-reachability"]

    assert run.reward == 0.0
    assert "artifact /app/main.html is a symbolic link" in (run.trace.error or "")


def test_separate_verifier_rejects_sidecar_symlink_artifact_roots(
    tmp_path_factory: pytest.TempPathFactory, wheel: Path
) -> None:
    dataset = tmp_path_factory.mktemp("harbor-sidecar-symlink") / "harbor-harness"
    task = dataset / "sidecar-reachability"
    shutil.copytree(TASKS / "sidecar-reachability", task)
    (task / "task.toml").write_text(
        """\
artifacts = [{ source = "/link", service = "web", exclude = ["main.html"] }]

[task]
name = "sidecar-reachability"

[verifier]
environment_mode = "separate"
timeout_sec = 30

[[verifier.collect]]
service = "web"
command = "ln -sfn /app /link"
timeout_sec = 10
""",
        encoding="utf-8",
    )
    (task / "solution" / "solve.sh").write_text("true\n", encoding="utf-8")

    run = asyncio.run(_grade_every_task(dataset, wheel))["sidecar-reachability"]

    assert run.reward == 0.0
    assert "artifact /link contains a symbolic link" in (run.trace.error or "")


def test_separate_verifier_sees_directory_artifacts_without_excluded_entries(
    tmp_path_factory: pytest.TempPathFactory, wheel: Path
) -> None:
    dataset = tmp_path_factory.mktemp("harbor-artifact-exclude") / "harbor-harness"
    task = dataset / "sidecar-reachability"
    shutil.copytree(TASKS / "sidecar-reachability", task)
    (task / "task.toml").write_text(
        """\
artifacts = [
  { source = "/app/outputs", destination = "results", exclude = ["*.tmp", "cache"] },
]

[task]
name = "sidecar-reachability"

[verifier]
environment_mode = "separate"
timeout_sec = 30
""",
        encoding="utf-8",
    )
    (task / "solution" / "solve.sh").write_text(
        """\
#!/bin/sh
set -eu
mkdir -p /app/outputs/cache/nested /app/outputs/logs
echo keep > /app/outputs/keep.txt
echo junk > /app/outputs/junk.tmp
echo junk > /app/outputs/logs/nested.tmp
echo junk > /app/outputs/cache/nested/blob
ln -s /etc/passwd /app/outputs/cache/link
ln -s missing /app/outputs/logs/dangling.tmp
""",
        encoding="utf-8",
    )
    (task / "tests" / "test.sh").write_text(
        """\
#!/bin/sh
set -u
mkdir -p /logs/verifier
if [ "$(cat /app/outputs/keep.txt 2>/dev/null)" = "keep" ] \\
  && [ -d /app/outputs/logs ] \\
  && [ ! -e /app/outputs/junk.tmp ] \\
  && [ ! -e /app/outputs/logs/nested.tmp ] \\
  && [ ! -L /app/outputs/logs/dangling.tmp ] \\
  && [ ! -e /app/outputs/cache ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo "excluded artifact entries leaked into the verifier" >&2
  echo 0 > /logs/verifier/reward.txt
fi
""",
        encoding="utf-8",
    )

    run = asyncio.run(_grade_every_task(dataset, wheel))["sidecar-reachability"]

    assert run.reward == 1.0, run.trace.error


def test_separate_verifier_rejects_artifacts_beneath_symlinks(
    tmp_path_factory: pytest.TempPathFactory, wheel: Path
) -> None:
    dataset = tmp_path_factory.mktemp("harbor-ancestor-symlink") / "harbor-harness"
    task = dataset / "sidecar-reachability"
    shutil.copytree(TASKS / "sidecar-reachability", task)
    config = task / "task.toml"
    config.write_text(
        """\
artifacts = [{ source = "/app/linked/tests", service = "main" }]

[task]
name = "sidecar-reachability"

[verifier]
environment_mode = "separate"
timeout_sec = 30
""",
        encoding="utf-8",
    )
    solution = task / "solution" / "solve.sh"
    solution.write_text(
        "#!/bin/sh\nset -eu\nln -s /media/hud/verifier /app/linked\n",
        encoding="utf-8",
    )

    run = asyncio.run(_grade_every_task(dataset, wheel))["sidecar-reachability"]

    assert run.reward == 0.0
    assert "artifact /app/linked/tests has a symbolic link in its path" in (run.trace.error or "")


def test_separate_verifier_restores_image_paths_between_shared_rollouts(
    tmp_path_factory: pytest.TempPathFactory, wheel: Path
) -> None:
    dataset = tmp_path_factory.mktemp("harbor-verifier-restore") / "harbor-harness"
    task = dataset / "sidecar-reachability"
    shutil.copytree(TASKS / "sidecar-reachability", task)
    (task / "task.toml").write_text(
        """\
artifacts = [{ source = "/tests/test.sh", service = "main" }]

[task]
name = "sidecar-reachability"

[verifier]
environment_mode = "separate"
timeout_sec = 30
""",
        encoding="utf-8",
    )
    (task / "solution" / "solve.sh").write_text("true\n", encoding="utf-8")

    async def grade_twice() -> list[Run]:
        taskset = _adapt(dataset, hud_requirement=str(wheel))
        job = await taskset.run(
            Oracle({"sidecar-reachability": "true"}),
            runtime=Shared(DockerRuntime(), width=1),
            group=2,
            max_concurrent=1,
        )
        return job.runs

    runs = asyncio.run(grade_twice())

    assert len(runs) == 2
    for run in runs:
        assert any(
            step.task_call is not None
            and step.task_call.phase == "evaluate"
            and step.task_call.name == "verify"
            for step in run.trace.steps
        )


def test_separate_verifier_honors_the_test_script_shebang(
    tmp_path_factory: pytest.TempPathFactory, wheel: Path
) -> None:
    dataset = tmp_path_factory.mktemp("harbor-verifier-shebang") / "harbor-harness"
    task = dataset / "sidecar-reachability"
    shutil.copytree(TASKS / "sidecar-reachability", task)
    (task / "task.toml").write_text(
        """\
[task]
name = "sidecar-reachability"

[verifier]
environment_mode = "separate"
timeout_sec = 30
""",
        encoding="utf-8",
    )
    (task / "solution" / "solve.sh").write_text("true\n", encoding="utf-8")
    (task / "tests" / "test.sh").write_text(
        """\
#!/usr/bin/env python3
from pathlib import Path

Path("/logs/verifier/reward.txt").write_text("1")
""",
        encoding="utf-8",
    )

    run = asyncio.run(_grade_every_task(dataset, wheel))["sidecar-reachability"]

    assert run.reward == 1.0


def test_env_templates_resolve_from_runtime_env_vars(
    tmp_path_factory: pytest.TempPathFactory, wheel: Path
) -> None:
    """``${VAR}``/``${VAR:-default}`` env values resolve exactly like Harbor's
    host-side resolution, sourced from the runtime's ``env_vars``."""
    dataset = tmp_path_factory.mktemp("harbor-env-templates") / "harbor-harness"
    task = make_harbor_task(
        dataset,
        "env-templates",
        task_toml="""\
[metadata]
category = "systems"

[environment.env]
GREETING = "${HARBOR_GREETING:-hello}"
JUDGE_KEY = "${HARBOR_JUDGE_KEY}"
EMBEDDED = "Bearer ${HARBOR_JUDGE_KEY}"

[verifier]
timeout_sec = 120

[verifier.env]
VERIFIER_KEY = "${HARBOR_JUDGE_KEY}"
EMPTY_DEFAULT = "${HARBOR_UNSET:-}"
""",
    )
    (task / "tests" / "test.sh").write_text(
        """\
#!/bin/bash
fail() { echo "unexpected $1"; echo "0.0" > /logs/verifier/reward.txt; exit 0; }
[ "$GREETING" = "hello" ] || fail "GREETING=$GREETING"
[ "$JUDGE_KEY" = "judge-secret" ] || fail "JUDGE_KEY=$JUDGE_KEY"
[ "$EMBEDDED" = 'Bearer ${HARBOR_JUDGE_KEY}' ] || fail "EMBEDDED=$EMBEDDED"
[ "$VERIFIER_KEY" = "judge-secret" ] || fail "VERIFIER_KEY=$VERIFIER_KEY"
[ "${EMPTY_DEFAULT-unset}" = "" ] || fail "EMPTY_DEFAULT=${EMPTY_DEFAULT-unset}"
echo "1.0" > /logs/verifier/reward.txt
""",
        encoding="utf-8",
    )
    solution = '[ "$JUDGE_KEY" = "judge-secret" ] && [ "$GREETING" = "hello" ]'

    async def grade() -> Run:
        taskset = harbor.adapt(dataset, hud_requirement=str(wheel)).taskset
        job = await taskset.run(
            Oracle({"env-templates": solution}),
            runtime=DockerRuntime(env_vars={"HARBOR_JUDGE_KEY": "judge-secret"}),
            max_concurrent=1,
        )
        (run,) = job.runs
        return run

    run = asyncio.run(grade())

    evaluation = run.evaluation
    info = evaluation.get("info") or {}
    detail = "\n".join(
        filter(
            None,
            (
                run.trace.content,
                run.trace.error,
                evaluation.get("content") or "",
                info.get("stdout"),
                info.get("stderr"),
            ),
        )
    )
    assert run.reward == 1.0, f"env-templates scored {run.reward}; the verifier reported:\n{detail}"


def test_missing_env_template_aborts_startup(
    tmp_path_factory: pytest.TempPathFactory, wheel: Path
) -> None:
    """A required ``${VAR}`` with no default and no runtime value must abort
    environment startup with an error naming the variable, like Harbor."""
    dataset = tmp_path_factory.mktemp("harbor-env-missing") / "harbor-harness"
    make_harbor_task(
        dataset,
        "env-missing",
        task_toml="""\
[metadata]
category = "systems"

[environment.env]
API_KEY = "${HARBOR_MISSING_KEY}"

[verifier]
timeout_sec = 120
""",
    )

    taskset = harbor.adapt(dataset, hud_requirement=str(wheel)).taskset
    task = next(iter(taskset))
    assert task.runtime_config is not None
    source = task.runtime_config.compose_source()
    assert source is not None
    compose = source.runnable_path("test")
    # Providers yield as soon as the published port exists, before the serve
    # process proves itself, so an early abort is only observable from the
    # adapted artifact: run its main service in the foreground.
    subprocess.run(
        ["sh", "build.sh"], cwd=compose.parent, check=True, capture_output=True, timeout=600
    )
    command = ["docker", "compose", "--file", str(compose), "run", "--rm", "main"]
    try:
        serve = subprocess.run(command, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        pytest.fail("main service kept serving despite an unresolvable env template")
    finally:
        subprocess.run(
            ["docker", "compose", "--file", str(compose), "down", "--volumes", "--remove-orphans"],
            capture_output=True,
            check=False,
        )
    assert serve.returncode != 0
    assert "HARBOR_MISSING_KEY" in serve.stdout + serve.stderr
