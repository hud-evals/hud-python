"""Local placement: LocalRuntime and the no-runtime resolution ladder.

LocalRuntime serves a fresh env per rollout from any pointer to it — a source
path (throwaway import), a live module-level env (its declaring file is the
recipe), or a ``(task) -> Environment`` constructor. With no runtime, a run
uses what is already known: taskset origin, then envs declared in imported
modules, else a loud error. Everything crosses the real control channel —
these tests drive the rollout engine end to end.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import AsyncGenerator  # noqa: TC003 - env.template resolves at runtime
from typing import Any, cast

import pytest

from hud.agents.base import Agent
from hud.environment import Environment
from hud.eval import LocalRuntime, Task, Taskset
from hud.eval.run import rollout

_SUMS_ENV = """\
from hud import Environment

env = Environment("{name}")


@env.template(id="add")
async def add(a: int, b: int):
    answer = yield f"add:{{a}}:{{b}}"
    yield 1.0 if answer == str(a + b) else 0.0
"""


@pytest.fixture
def imported_env(tmp_path, request):
    """Write an env module, import it for real, and clean it up after.

    The module stays in ``sys.modules`` for the test's duration — the state a
    user's ``from env import add`` leaves behind.
    """
    module_name = f"_sums_mod_{request.node.name}"
    file = tmp_path / f"{module_name}.py"
    file.write_text(_SUMS_ENV.format(name="sums"), encoding="utf-8")
    spec = importlib.util.spec_from_file_location(module_name, file)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    yield module
    del sys.modules[module_name]


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
        self._fn = fn

    async def __call__(self, run: Any) -> None:
        run.trace.content = self._fn(run.prompt)


def _solve_add(prompt: str) -> str:
    _, a, b = prompt.split(":")
    return str(int(a) + int(b))


# ─── LocalRuntime: the three pointer forms ─────────────────────────────


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


async def test_live_env_pointer_resolves_to_its_declaring_file(imported_env) -> None:
    run = await rollout(
        Task(env="sums", id="add", args={"a": 2, "b": 3}),
        _FnAgent(_solve_add),
        runtime=LocalRuntime(imported_env.env),
    )

    assert run.reward == 1.0


def test_live_env_without_a_declaring_file_is_rejected() -> None:
    with pytest.raises(TypeError, match="constructor instead"):
        LocalRuntime(_sums_env())


async def test_constructor_builds_fresh_per_rollout_from_the_row() -> None:
    built: list[str] = []

    def env_for(task: Task) -> Environment:
        built.append(task.env)
        return _sums_env(task.env)

    job = await Task(env="sums", id="add", args={"a": 1, "b": 2}).run(
        _FnAgent(_solve_add),
        runtime=LocalRuntime(env_for),
        group=3,
        max_concurrent=3,
    )

    assert all(run.reward == 1.0 for run in job.runs)
    assert built == ["sums", "sums", "sums"]


async def test_source_missing_env_name_fails_loudly(tmp_path) -> None:
    (tmp_path / "env.py").write_text(
        'from hud import Environment\n\nenv = Environment("sums")\n',
        encoding="utf-8",
    )
    provider = LocalRuntime(tmp_path / "env.py")

    with pytest.raises(ValueError, match="no Environment named 'other'"):
        async with provider(Task(env="other", id="add")):
            pass


# ─── the no-runtime resolution ladder ──────────────────────────────────


async def test_module_loaded_taskset_serves_its_source_by_default(tmp_path, request) -> None:
    (tmp_path / "env.py").write_text(_SUMS_ENV.format(name="sums"), encoding="utf-8")
    (tmp_path / "tasks.py").write_text(
        "from env import add\n\ntasks = [add(a=2, b=3), add(a=4, b=5)]\n",
        encoding="utf-8",
    )
    # tasks.py's `from env import add` imports env.py normally, so it outlives
    # the throwaway tasks module.
    request.addfinalizer(lambda: sys.modules.pop("env", None))

    taskset = Taskset.from_module(tmp_path / "tasks.py")
    job = await taskset.run(_FnAgent(_solve_add))

    assert len(job.runs) == 2
    assert all(run.reward == 1.0 for run in job.runs)


async def test_minted_tasks_resolve_a_declared_env_by_name(imported_env) -> None:
    job = await Taskset("sums", [imported_env.add(a=2, b=3), imported_env.add(a=4, b=5)]).run(
        _FnAgent(_solve_add)
    )

    assert len(job.runs) == 2
    assert all(run.reward == 1.0 for run in job.runs)


async def test_no_placement_fails_with_the_forms_to_pass() -> None:
    with pytest.raises(ValueError, match="no placement for env"):
        await Task(env="ghost", id="add").run(_FnAgent(_solve_add))


async def test_ambiguous_env_name_fails_loudly(imported_env, tmp_path, request) -> None:
    module_name = f"_sums_rival_{request.node.name}"
    file = tmp_path / f"{module_name}.py"
    file.write_text(_SUMS_ENV.format(name="sums"), encoding="utf-8")
    spec = importlib.util.spec_from_file_location(module_name, file)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        with pytest.raises(ValueError, match="LocalRuntime\\(env\\)"):
            await Task(env="sums", id="add").run(_FnAgent(_solve_add))
    finally:
        del sys.modules[module_name]


def test_rejects_a_non_pointer_argument() -> None:
    with pytest.raises(TypeError, match="expected a source path"):
        LocalRuntime(cast("Any", 42))


async def test_failed_startup_still_runs_shutdown_hooks() -> None:
    from hud.eval.runtime import _local

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
        async with _local(env):
            pass

    assert lifecycle == ["up", "down"]


async def test_tasks_only_module_resolves_envs_imported_from_elsewhere(
    tmp_path, monkeypatch, request
) -> None:
    # The env lives in a separate importable package, not next to tasks.py:
    # the origin declares no envs, so resolution falls through to the live
    # env the tasks module imported.
    env_dir = tmp_path / "pkg"
    env_dir.mkdir()
    (env_dir / "sums_envmod.py").write_text(_SUMS_ENV.format(name="sums"), encoding="utf-8")
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    (tasks_dir / "tasks.py").write_text(
        "from sums_envmod import add\n\ntasks = [add(a=2, b=3)]\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(env_dir))
    request.addfinalizer(lambda: sys.modules.pop("sums_envmod", None))

    taskset = Taskset.from_module(tasks_dir / "tasks.py")
    job = await taskset.run(_FnAgent(_solve_add))

    assert [run.reward for run in job.runs] == [1.0]


async def test_single_file_taskset_never_drags_in_a_same_named_sibling(tmp_path) -> None:
    (tmp_path / "env_a.py").write_text(
        _SUMS_ENV.format(name="sums") + "\ntasks = [add(a=2, b=3)]\n", encoding="utf-8"
    )
    (tmp_path / "env_b.py").write_text(_SUMS_ENV.format(name="sums"), encoding="utf-8")

    taskset = Taskset.from_module(tmp_path / "env_a.py")
    job = await taskset.run(_FnAgent(_solve_add))

    assert [run.reward for run in job.runs] == [1.0]


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


async def test_live_env_mutated_after_import_fails_with_the_cause(imported_env) -> None:
    @imported_env.env.template(id="patched")
    async def patched() -> Any:
        answer = yield "noop"
        yield 1.0 if answer else 0.0

    provider = LocalRuntime(imported_env.env)

    with pytest.raises(ValueError, match="modified after import"):
        async with provider(Task(env="sums", id="patched")):
            pass
