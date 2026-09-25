"""LocalRuntime serves environments from live instances and source recipes."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncGenerator  # noqa: TC003 - env.template resolves at runtime
from pathlib import Path
from typing import Any, cast

import pytest

import hud.eval.runtime.local as local_runtime_module
from hud.agents.base import Agent
from hud.environment import Environment, load_environment
from hud.eval import LocalRuntime, RuntimeConfig, SubprocessRuntime, Task, Taskset

_SUMS_ENV = """\
from hud import Environment

env = Environment("{name}")


@env.template(id="add")
async def add(a: int, b: int):
    answer = yield f"add:{{a}}:{{b}}"
    yield 1.0 if answer == str(a + b) else 0.0
"""


def _sums_env(name: str = "sums") -> Environment:
    env = Environment(name)

    @env.template(id="add")
    async def add(a: int, b: int) -> AsyncGenerator[Any, Any]:
        answer = yield f"add:{a}:{b}"
        yield 1.0 if answer == str(a + b) else 0.0

    return env


class _FnAgent(Agent):
    """Stateless agent: answers each run by applying ``fn`` to ``run.prompt``."""

    def __init__(self, fn: Any) -> None:
        super().__init__()
        self._fn = fn

    async def __call__(self, run: Any) -> None:
        run.trace.content = self._fn(run.prompt)


def _solve_add(prompt: str) -> str:
    _, a, b = prompt.split(":")
    return str(int(a) + int(b))


# ─── LocalRuntime sources ──────────────────────────────────────────────


async def test_source_path_serves_a_fresh_env_per_rollout(tmp_path) -> None:
    # LOADS is module state: a fresh throwaway import per acquisition means
    # every rollout sees exactly one load.
    (tmp_path / "env.py").write_text(
        "from hud import Environment\n\n"
        "LOADS = []\n"
        'env = Environment("sums")\n\n\n'
        '@env.template(id="add")\nasync def add(a: int, b: int):\n'
        "    LOADS.append(1)\n"
        '    answer = yield f"add:{a}:{b}:{len(LOADS)}"\n'
        "    yield 1.0 if answer == str(a + b) else 0.0\n",
        encoding="utf-8",
    )

    def _solve(prompt: str) -> str:
        _, a, b, loads = prompt.split(":")
        assert loads == "1"
        return str(int(a) + int(b))

    job = await Task(env="sums", id="add", args={"a": 2, "b": 3}).run(
        _FnAgent(_solve),
        runtime=LocalRuntime(tmp_path / "env.py"),
        group=2,
    )

    assert [run.reward for run in job.runs] == [1.0, 1.0]


async def test_subprocess_runtime_streams_environment_output(
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "env.py"
    source.write_text(
        "import sys\n\n"
        "from hud import Environment\n\n"
        'env = Environment("sums")\n\n'
        "@env.initialize\n"
        "async def start():\n"
        '    print("x" * 100_000, flush=True)\n'
        '    print("environment booted", flush=True)\n'
        '    print("y" * 100_000, file=sys.stderr, flush=True)\n'
        '    print("environment warning", file=sys.stderr, flush=True)\n',
        encoding="utf-8",
    )

    async with SubprocessRuntime(source)(Task(env="sums", id="add")):
        pass

    captured = capsys.readouterr()
    assert "x" * 100_000 in captured.out
    assert "environment booted" in captured.out
    assert "y" * 100_000 in captured.err
    assert "environment warning" in captured.err


def test_subprocess_runtime_serves_a_live_env_from_its_template_file(tmp_path, request) -> None:
    module_name = f"sums_env_{request.node.name}"
    env_py = tmp_path / f"{module_name}.py"
    env_py.write_text(_SUMS_ENV.format(name="sums"), encoding="utf-8")
    tasks_py = tmp_path / "tasks.py"
    tasks_py.write_text(f"from {module_name} import add\n\ntasks = [add(a=2, b=3)]\n")
    request.addfinalizer(lambda: sys.modules.pop(module_name, None))

    task = next(iter(Taskset.from_module(tasks_py)))
    assert task._env is not None
    provider = SubprocessRuntime(task._env)

    assert provider.source == env_py.resolve()
    assert provider.env == "sums"


def test_subprocess_runtime_rejects_env_without_templates() -> None:
    with pytest.raises(ValueError, match="exactly one source file"):
        SubprocessRuntime(Environment("demo"))


def test_subprocess_runtime_rejects_env_pin_for_live_env() -> None:
    with pytest.raises(TypeError, match="env= applies only to source paths"):
        SubprocessRuntime(_sums_env(), env="sums")


async def test_subprocess_runtime_fails_when_stdout_closes_before_serving(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "env.py"
    source.write_text("", encoding="utf-8")
    output = asyncio.StreamReader()
    output.feed_eof()
    error = asyncio.StreamReader()
    error.feed_data(b"stdout closed\n")
    error.feed_eof()

    class Process:
        stdout = output
        stderr = error
        returncode = None
        terminated = False

        async def wait(self) -> int:
            await asyncio.Event().wait()
            return 0

        async def terminate(self) -> None:
            self.terminated = True

    process = Process()

    async def create_process(*args: Any, **kwargs: Any) -> Any:
        return process

    monkeypatch.setattr(local_runtime_module, "create_process_group_exec", create_process)

    async def run() -> None:
        async with SubprocessRuntime(source, ready_timeout=30)(Task(env="closed", id="noop")):
            pass

    with pytest.raises(RuntimeError, match=r"(?s)closed stdout.*stdout closed"):
        await asyncio.wait_for(run(), timeout=1)
    assert process.terminated is True


async def test_constructor_builds_fresh_per_rollout_from_the_row() -> None:
    built: list[str] = []
    started = 0
    all_started = asyncio.Event()

    def env_for(task: Task) -> Environment:
        built.append(task.env)
        env = _sums_env(task.env)

        @env.initialize
        async def start() -> None:
            nonlocal started
            started += 1
            if started == 3:
                all_started.set()
            await all_started.wait()

        return env

    job = await Task(env="sums", id="add", args={"a": 1, "b": 2}).run(
        _FnAgent(_solve_add),
        runtime=LocalRuntime(env_for, ready_timeout=2),
        group=3,
        max_concurrent=3,
    )

    assert all(run.reward == 1.0 for run in job.runs)
    assert built == ["sums", "sums", "sums"]


async def test_live_environment_is_served_serially() -> None:
    env = Environment("sums")
    active = 0

    @env.template(id="add")
    async def add(a: int, b: int):
        answer = yield f"add:{a}:{b}"
        yield 1.0 if answer == str(a + b) else 0.0

    @env.initialize
    async def _start() -> None:
        nonlocal active
        active += 1
        assert active == 1
        await asyncio.sleep(0)

    @env.shutdown
    async def _stop() -> None:
        nonlocal active
        active -= 1

    job = await add(a=2, b=3).run(
        _FnAgent(_solve_add),
        group=2,
        max_concurrent=2,
    )

    assert [run.reward for run in job.runs] == [1.0, 1.0]
    assert active == 0


@pytest.mark.parametrize(
    "source", ["directory", "split_file", "package", "factory", "separate_providers"]
)
async def test_cached_environment_lifecycle_is_serialized(tmp_path, request, source) -> None:
    module_name = f"cached_lifecycle_{source}"
    project = tmp_path / module_name
    project.mkdir()
    if source == "package":
        (project / "__init__.py").write_text("from .env import env\n")
    events = tmp_path / "lifecycle.txt"
    (project / f"{module_name}.py").write_text(
        _SUMS_ENV.format(name="sums")
        + f"""
import asyncio
from pathlib import Path
events = Path({str(events)!r})

@env.initialize
async def start():
    with events.open("a") as log:
        log.write("start\\n")
    await asyncio.sleep(0)

@env.shutdown
async def stop():
    with events.open("a") as log:
        log.write("stop\\n")
"""
    )
    prefix = "." if source == "package" else ""
    entrypoint = project / "env.py"
    entrypoint.write_text(f"from {prefix}{module_name} import env\n")

    def cleanup() -> None:
        for name, module in tuple(sys.modules.items()):
            file = getattr(module, "__file__", None)
            if file and Path(file).is_relative_to(project):
                del sys.modules[name]

    request.addfinalizer(cleanup)
    if source in {"factory", "separate_providers"}:
        env = load_environment(entrypoint)
        runtime = LocalRuntime(lambda _task: env)
        if source == "separate_providers":
            runtime = lambda task: LocalRuntime(env)(task)
    else:
        runtime = LocalRuntime(project if source == "directory" else entrypoint)
    job = await Task(env="sums", id="add", args={"a": 2, "b": 3}).run(
        _FnAgent(_solve_add), runtime=runtime, group=2, max_concurrent=2
    )

    assert [run.reward for run in job.runs] == [1.0, 1.0]
    assert events.read_text().splitlines() == ["start", "stop", "start", "stop"]


async def test_cancelling_a_waiting_acquisition_preserves_the_running_environment() -> None:
    env = _sums_env()
    active = False

    @env.initialize
    async def start() -> None:
        nonlocal active
        active = True

    @env.shutdown
    async def stop() -> None:
        nonlocal active
        active = False

    runtime = LocalRuntime(lambda _task: env)
    task = Task(env="sums", id="add")

    async def acquire() -> None:
        async with runtime(task):
            pytest.fail("the environment already has an active acquisition")

    async with runtime(task):
        waiter = asyncio.create_task(acquire())
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert active

    assert not active


def test_local_environment_can_run_in_successive_event_loops() -> None:
    env = _sums_env()
    task = Task(env="sums", id="add", args={"a": 2, "b": 3})

    for _ in range(2):
        job = asyncio.run(
            task.run(_FnAgent(_solve_add), runtime=LocalRuntime(env), group=2, max_concurrent=2)
        )
        assert [run.reward for run in job.runs] == [1.0, 1.0]


async def test_serialized_task_does_not_retain_local_placement() -> None:
    env = Environment("sums")

    @env.template(id="add")
    async def add(a: int, b: int):
        yield f"add:{a}:{b}"
        yield 1.0

    task = add(a=2, b=3)
    portable = Task.model_validate(task.model_dump())

    with pytest.raises(ValueError, match="no placement: pass runtime="):
        await portable.run(_FnAgent(_solve_add))


async def test_source_missing_env_name_fails_loudly(tmp_path) -> None:
    (tmp_path / "env.py").write_text(
        'from hud import Environment\n\nenv = Environment("sums")\n',
        encoding="utf-8",
    )
    provider = LocalRuntime(tmp_path / "env.py")

    with pytest.raises(ValueError, match="no Environment named 'other'"):
        async with provider(Task(env="other", id="add")):
            pass


async def test_module_taskset_uses_factory_environments(tmp_path) -> None:
    source = tmp_path / "tasks.py"
    source.write_text(
        _SUMS_ENV.format(name="sums") + "\ntasks = [add(a=2, b=3), add(a=4, b=5)]\n",
        encoding="utf-8",
    )

    job = await Taskset.from_module(source).run(_FnAgent(_solve_add))

    assert [run.reward for run in job.runs] == [1.0, 1.0]


async def test_tasks_module_uses_factory_environment(tmp_path, request) -> None:
    module_name = f"sums_env_{request.node.name}"
    (tmp_path / f"{module_name}.py").write_text(
        _SUMS_ENV.format(name="sums"),
        encoding="utf-8",
    )
    tasks = tmp_path / "tasks.py"
    tasks.write_text(
        f"from {module_name} import add\n\ntasks = [add(a=2, b=3)]\n",
        encoding="utf-8",
    )
    request.addfinalizer(lambda: sys.modules.pop(module_name, None))

    job = await Taskset.from_module(tasks).run(_FnAgent(_solve_add))

    assert job.reward == 1.0


async def test_ad_hoc_taskset_requires_explicit_placement() -> None:
    with pytest.raises(ValueError, match="no placement: pass runtime="):
        await Taskset("sums", [Task(env="sums", id="add")]).run(_FnAgent(_solve_add))


async def test_container_rows_start_their_image_by_default(monkeypatch) -> None:
    """A row that names an image is placeable without ``runtime=``: it gets DockerRuntime."""
    import hud.eval.taskset as taskset_module

    live = LocalRuntime(_sums_env())
    monkeypatch.setattr(
        taskset_module,
        "DockerRuntime",
        lambda: lambda task: live(task.model_copy(update={"runtime_config": None})),
    )
    portable = Task(env="sums", id="add", args={"a": 2, "b": 3})
    container = Task(
        env="sums", id="add", args={"a": 4, "b": 5}, runtime_config=RuntimeConfig(image="sums")
    )

    job = await Taskset("sums", [container]).run(_FnAgent(_solve_add))
    assert [run.reward for run in job.runs] == [1.0]

    with pytest.raises(ValueError, match="no placement"):
        await Taskset("sums", [container, portable]).run(_FnAgent(_solve_add))


async def test_container_row_with_shared_verifier_is_placeable(monkeypatch) -> None:
    """A verifier on the same env with no runtime of its own rides the actor's container."""
    import hud.eval.taskset as taskset_module

    env = _sums_env()

    @env.template(id="verify")
    async def verify() -> AsyncGenerator[Any, Any]:
        actor_grade = yield ""
        yield 1.0 if actor_grade["score"] == 1.0 else 0.0

    live = LocalRuntime(env)
    starts: list[str] = []

    def docker(task: Task) -> Any:
        starts.append(task.id)
        return live(task.model_copy(update={"runtime_config": None}))

    monkeypatch.setattr(taskset_module, "DockerRuntime", lambda: docker)
    container = Task(
        env="sums",
        id="add",
        args={"a": 4, "b": 5},
        runtime_config=RuntimeConfig(image="sums"),
        verifier=Task(env="sums", id="verify"),
    )

    job = await Taskset("sums", [container]).run(_FnAgent(_solve_add))
    assert [run.reward for run in job.runs] == [1.0]
    assert starts == ["add"]  # one container for actor and verifier


async def test_empty_taskset_needs_no_placement() -> None:
    job = await Taskset("empty", []).run(_FnAgent(_solve_add))

    assert job.runs == []


def test_rejects_a_non_pointer_argument() -> None:
    with pytest.raises(TypeError, match="expected a source path"):
        LocalRuntime(cast("Any", 42))


async def test_failed_startup_still_runs_shutdown_hooks() -> None:
    env = _sums_env()
    lifecycle: list[str] = []

    @env.initialize
    async def _up() -> None:
        lifecycle.append("up")

    @env.shutdown
    async def _down() -> None:
        lifecycle.append("down")

    @env.initialize
    async def _boom() -> None:
        raise RuntimeError("daemon failed to start")

    with pytest.raises(RuntimeError, match="daemon failed to start"):
        async with LocalRuntime(env)(Task(env="sums", id="add")):
            pass

    assert lifecycle == ["up", "down"]


async def test_template_can_lazily_import_a_sibling_module(tmp_path) -> None:
    (tmp_path / "lazy_helper.py").write_text("ANSWER_SUFFIX = ':ok'\n", encoding="utf-8")
    (tmp_path / "env.py").write_text(
        "from hud import Environment\n\n"
        'env = Environment("sums")\n\n\n'
        '@env.template(id="add")\nasync def add(a: int, b: int):\n'
        "    import lazy_helper\n"
        '    answer = yield f"add:{a}:{b}{lazy_helper.ANSWER_SUFFIX}"\n'
        "    yield 1.0 if answer == str(a + b) else 0.0\n",
        encoding="utf-8",
    )

    def _solve(prompt: str) -> str:
        _, a, b, ok = prompt.split(":")
        assert ok == "ok"
        return str(int(a) + int(b))

    job = await Task(env="sums", id="add", args={"a": 2, "b": 3}).run(
        _FnAgent(_solve),
        runtime=LocalRuntime(tmp_path / "env.py"),
        group=2,
        max_concurrent=2,
    )

    assert [run.reward for run in job.runs] == [1.0, 1.0]


# ─── seam defenses ─────────────────────────────────────────────────────


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # the broken source leaks its coroutine
async def test_unguarded_run_call_in_source_names_the_mistake(tmp_path) -> None:
    (tmp_path / "env.py").write_text(
        "import asyncio\n\nfrom hud import Environment\n\n"
        'env = Environment("sums")\n\n'
        "asyncio.run(asyncio.sleep(0))\n",
        encoding="utf-8",
    )
    provider = LocalRuntime(tmp_path / "env.py")

    with pytest.raises(RuntimeError, match='if __name__ == "__main__"'):
        async with provider(Task(env="sums", id="add")):
            pass
