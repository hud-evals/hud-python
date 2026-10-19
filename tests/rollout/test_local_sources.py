"""Local providers serve an environment from the source a user has: a file, a directory,
a package, a live instance, or a constructor, in this process or in a child process."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import textwrap
from typing import TYPE_CHECKING, Any

import pytest

from hud.environment import Environment, load_environment
from hud.eval import LocalRuntime, SubprocessRuntime, Task, Taskset, rollout
from tests.support import ScriptedAgent, eventually, lab, minted, solve

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from hud.eval import Provider

LOGGED_ENV = """
import asyncio
from pathlib import Path

from hud import Environment

LOG = Path({log!r})
with LOG.open("a") as log:
    log.write("import\\n")

env = Environment("{name}")


@env.initialize
async def start():
    with LOG.open("a") as log:
        log.write("start\\n")
    await asyncio.sleep(0)


@env.shutdown
async def stop():
    with LOG.open("a") as log:
        log.write("stop\\n")


@env.template()
async def add(a: int, b: int):
    answer = yield f"add {{a}} {{b}}"
    yield 1.0 if answer == str(a + b) else 0.0
"""


class Project:
    """A source tree on disk whose environment logs its imports and lifecycle."""

    def __init__(self, root: Path, request: pytest.FixtureRequest) -> None:
        self.root = root
        self.log = root / "lifecycle.log"
        self.package = root / f"project_{abs(hash(root)):x}"
        self.package.mkdir()
        request.addfinalizer(self.forget)

    def module(self, name: str = "sums", *, env: str = "lab") -> Path:
        path = self.package / f"{self.package.name}_{name}.py"
        path.write_text(LOGGED_ENV.format(log=str(self.log), name=env))
        return path

    def entry(self, text: str) -> Path:
        path = self.package / "env.py"
        path.write_text(text)
        return path

    def lines(self) -> list[str]:
        return self.log.read_text().splitlines() if self.log.exists() else []

    def forget(self) -> None:
        for name, module in sys.modules.copy().items():
            file = getattr(module, "__file__", None)
            if file and str(file).startswith(str(self.package)):
                del sys.modules[name]


def file_source(project: Project) -> Provider:
    return LocalRuntime(project.module())


def directory_source(project: Project) -> Provider:
    project.module()
    return LocalRuntime(project.package)


def package_source(project: Project) -> Provider:
    module = project.module().stem
    (project.package / "__init__.py").write_text("")
    return LocalRuntime(project.entry(f"from .{module} import env\n"))


def shared_constructor(project: Project) -> Provider:
    env = load_environment(project.module())
    return LocalRuntime(lambda task: env)


def subprocess_live_env(project: Project) -> Provider:
    return SubprocessRuntime(load_environment(project.module()))


SOURCES = {
    "a file is imported per acquisition": (file_source, 1, ["import"] * 2),
    "a directory is imported once": (directory_source, 2, ["import"]),
    "a package entry with a relative import shares its env": (package_source, 2, ["import"]),
    "a constructor returning one instance shares it": (shared_constructor, 2, ["import"]),
    "a child process serves a live env from its template file": (
        subprocess_live_env,
        1,
        ["import"] * 3,
    ),
}


@pytest.mark.parametrize(("source", "concurrency", "imports"), SOURCES.values(), ids=SOURCES.keys())
async def test_a_source_serves_its_rows_one_acquisition_at_a_time_per_instance(
    source: Callable[[Project], Provider],
    concurrency: int,
    imports: list[str],
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    project = Project(tmp_path, request)

    job = await Task(env="lab", id="add", args={"a": 2, "b": 3}).run(
        ScriptedAgent(solve), runtime=source(project), group=2, max_concurrent=concurrency
    )

    assert [run.reward for run in job.runs] == [1.0, 1.0]
    assert [line for line in project.lines() if line == "import"] == imports
    assert [line for line in project.lines() if line != "import"] == ["start", "stop"] * 2


WHERE_ENV = """
from pathlib import Path

from hud import Environment

env = Environment("{name}")


@env.template()
async def where():
    answer = yield f"{name} serves from {{Path.cwd()}}"
    yield 1.0
"""


@pytest.mark.parametrize("provider", [LocalRuntime, SubprocessRuntime])
async def test_env_picks_one_of_several_environments_a_source_defines(
    provider: Callable[..., Provider], tmp_path: Path
) -> None:
    source = tmp_path / "project"
    source.mkdir()
    (source / "alpha.py").write_text(WHERE_ENV.format(name="alpha"))
    (source / "beta.py").write_text(WHERE_ENV.format(name="beta"))
    (source / "env.py").write_text("from alpha import env as alpha\nfrom beta import env as beta\n")
    agent = ScriptedAgent("ok")

    job = await Task(env="alpha", id="where").run(
        agent, runtime=provider(source / "env.py", env="beta")
    )

    assert job.runs[0].reward == 1.0
    assert agent.prompts == [
        f"beta serves from {source if provider is SubprocessRuntime else os.getcwd()}"
    ]


async def test_distinct_instances_from_a_constructor_run_concurrently() -> None:
    built: list[str] = []
    started = 0
    all_started = asyncio.Event()

    def build(task: Task) -> Environment:
        built.append(task.slug)
        env = lab()

        @env.initialize
        async def barrier() -> None:
            nonlocal started
            started += 1
            if started == 3:
                all_started.set()
            await all_started.wait()

        return env

    row = Task(env="lab", id="add", args={"a": 1, "b": 2})
    job = await row.run(ScriptedAgent(solve), runtime=LocalRuntime(build, ready_timeout=5), group=3)

    assert [run.reward for run in job.runs] == [1.0, 1.0, 1.0]
    assert built == [row.slug] * 3


def rows_of_two_envs(tmp_path: Path, request: pytest.FixtureRequest) -> Any:
    project = Project(tmp_path, request)
    alpha, beta = project.module("alpha", env="alpha"), project.module("beta", env="beta")
    source = project.entry(
        f"from {alpha.stem} import env as alpha\nfrom {beta.stem} import env as beta\n"
    )
    rows = [
        Task(env="alpha", id="add", args={"a": 1, "b": 2}),
        Task(env="beta", id="add", args={"a": 3, "b": 4}),
    ]
    return Taskset("mixed", rows), LocalRuntime(source)


def a_template_importing_a_sibling(tmp_path: Path, request: pytest.FixtureRequest) -> Any:
    (tmp_path / "lazy_helper.py").write_text("SUFFIX = ''\n")
    source = write(
        tmp_path / "env.py",
        """
        from hud import Environment

        env = Environment("lab")


        @env.template()
        async def add(a: int, b: int):
            import lazy_helper

            answer = yield f"add {a} {b}{lazy_helper.SUFFIX}"
            yield 1.0 if answer == str(a + b) else 0.0
        """,
    )
    return Taskset("lazy", [Task(env="lab", id="add", args={"a": 2, "b": 3})]), LocalRuntime(source)


def rows_a_live_env_minted(tmp_path: Path, request: pytest.FixtureRequest) -> Any:
    return Taskset("live", [minted()]), None


def a_module_taskset(tmp_path: Path, request: pytest.FixtureRequest) -> Any:
    source = write(
        tmp_path / "tasks.py",
        """
        from hud import Environment

        env = Environment("lab")


        @env.template()
        async def add(a: int, b: int):
            answer = yield f"add {a} {b}"
            yield 1.0 if answer == str(a + b) else 0.0


        tasks = [add(a=2, b=3), add(a=4, b=5)]
        """,
    )
    return Taskset.from_module(source), None


TASKSETS: dict[str, Callable[[Path, pytest.FixtureRequest], Any]] = {
    "one source serves each row's own env": rows_of_two_envs,
    "a template imports a sibling module while it runs": a_template_importing_a_sibling,
    "rows a live env minted run against it": rows_a_live_env_minted,
    "a module's tasks run against the envs it defines": a_module_taskset,
}


@pytest.mark.parametrize("taskset", TASKSETS.values(), ids=TASKSETS.keys())
async def test_a_taskset_runs_each_row_in_the_env_it_names(
    taskset: Callable[[Path, pytest.FixtureRequest], Any],
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    rows, runtime = taskset(tmp_path, request)

    job = await rows.run(ScriptedAgent(solve), runtime=runtime, group=2, max_concurrent=2)

    assert [run.reward for run in job.runs] == [1.0] * len(job.runs)
    assert len(job.runs) == 2 * len(list(rows))


def test_a_live_environment_serves_across_event_loops() -> None:
    env = lab()
    row = Task(env="lab", id="add", args={"a": 2, "b": 3})

    rewards = [
        asyncio.run(row.run(ScriptedAgent(solve), runtime=LocalRuntime(env), group=2)).reward
        for _ in range(2)
    ]

    assert rewards == [1.0, 1.0]


def write(path: Path, source: str) -> Path:
    path.write_text(textwrap.dedent(source))
    return path


ADD = Task(env="lab", id="add", args={"a": 2, "b": 3})


def failing_initialize(events: list[str]) -> Environment:
    env = lab(events)

    @env.initialize
    async def explode() -> None:
        raise RuntimeError("daemon failed to start")

    return env


def slow_initialize(events: list[str]) -> Environment:
    env = lab(events)

    @env.initialize
    async def hang() -> None:
        await asyncio.Event().wait()

    return env


CLOSES_STDOUT = """
import os
import sys
import time

print("stdout closed", file=sys.stderr, flush=True)
os.close(1)
time.sleep(60)
"""
ACQUISITION_ERRORS: dict[
    str, tuple[Callable[[Path, list[str]], Provider], Task, Any, tuple[str, ...], list[str]]
] = {
    "a source without the row's env": (
        lambda tmp, events: LocalRuntime(write(tmp / "env.py", "from hud import Environment\n")),
        Task(env="other", id="add"),
        ValueError,
        (),
        [],
    ),
    "a source running an event loop at import": (
        lambda tmp, events: LocalRuntime(
            write(tmp / "env.py", "import asyncio\n\nasyncio.run(asyncio.sleep(0))\n")
        ),
        ADD,
        RuntimeError,
        ("guard top-level run calls",),
        [],
    ),
    "an initialize hook that raises still shuts down": (
        lambda tmp, events: LocalRuntime(failing_initialize(events)),
        ADD,
        RuntimeError,
        (),
        ["initialize", "shutdown"],
    ),
    "an initialize hook slower than ready_timeout still shuts down": (
        lambda tmp, events: LocalRuntime(slow_initialize(events), ready_timeout=0.1),
        ADD,
        TimeoutError,
        (),
        ["initialize", "shutdown"],
    ),
    "a child that closes stdout before serving": (
        lambda tmp, events: SubprocessRuntime(write(tmp / "env.py", CLOSES_STDOUT)),
        ADD,
        RuntimeError,
        ("spawned env closed stdout before serving", "stdout closed"),
        [],
    ),
}


@pytest.mark.filterwarnings("ignore:coroutine 'sleep' was never awaited:RuntimeWarning")
@pytest.mark.parametrize(
    ("provider", "row", "error", "messages", "lifecycle"),
    ACQUISITION_ERRORS.values(),
    ids=ACQUISITION_ERRORS.keys(),
)
async def test_a_source_that_cannot_serve_fails_the_run_naming_the_mistake(
    provider: Callable[[Path, list[str]], Provider],
    row: Task,
    error: type[Exception],
    messages: tuple[str, ...],
    lifecycle: list[str],
    tmp_path: Path,
) -> None:
    events: list[str] = []

    run = await rollout(row, ScriptedAgent("x"), runtime=provider(tmp_path, events))

    assert (run.trace.status, run.prompt) == ("error", None)
    assert f"{error.__name__}" in (run.trace.error or "")
    assert [message for message in messages if message not in (run.trace.error or "")] == []
    assert events == lifecycle


def process_is_gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    status = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, check=False
    )
    return status.stdout.strip().startswith("Z")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
async def test_a_child_that_dies_before_serving_reports_its_output_and_takes_its_children(
    tmp_path: Path,
) -> None:
    source = write(
        tmp_path / "env.py",
        """
        import asyncio
        import sys
        from pathlib import Path

        from hud import Environment

        env = Environment("leaky")


        @env.initialize
        async def start():
            child = await asyncio.create_subprocess_exec("sleep", "120")
            Path("child.pid").write_text(str(child.pid))
            print("daemon starting", file=sys.stderr, flush=True)
            raise SystemExit(3)
        """,
    )

    run = await rollout(
        Task(env="leaky", id="noop"), ScriptedAgent("x"), runtime=SubprocessRuntime(source)
    )

    error = run.trace.error or ""
    assert f"spawned env exited with code 3 before serving (source: {source}):\n" in error
    assert "daemon starting" in error
    pid = int((tmp_path / "child.pid").read_text())
    try:
        await eventually(lambda: process_is_gone(pid), within=5)
    finally:
        if not process_is_gone(pid):
            os.kill(pid, signal.SIGKILL)


async def test_a_child_streams_its_output_to_the_terminal(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    source = write(
        tmp_path / "env.py",
        """
        import sys

        from hud import Environment

        env = Environment("lab")


        @env.initialize
        async def start():
            print("x" * 100_000, flush=True)
            print("environment booted", flush=True)
            print("y" * 100_000, file=sys.stderr, flush=True)
            print("environment warning", file=sys.stderr, flush=True)
        """,
    )

    await rollout(ADD, ScriptedAgent(solve), runtime=SubprocessRuntime(source))

    out, err = capfd.readouterr()
    assert "x" * 100_000 + "\nenvironment booted\n" in out
    assert "y" * 100_000 + "\nenvironment warning\n" in err
