"""``LocalRuntime`` and ``SubprocessRuntime`` serve an environment from the source a user
has: a file, a directory, a package, a live instance, or a constructor."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import textwrap
from typing import TYPE_CHECKING, Any

import pytest

from hud import Environment
from hud.agents.base import Agent
from hud.environment import load_environment
from hud.eval import LocalRuntime, SubprocessRuntime, Task, Taskset, rollout
from tests.conftest import eventually
from tests.fixtures.agents import ScriptedAgent

if TYPE_CHECKING:
    from pathlib import Path

    from hud.eval import Run


def write(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source))
    return path


@pytest.mark.parametrize(
    ("source", "imports"),
    [("counted/sums.py", 2), ("counted", 1)],
    ids=["a-file-is-imported-for-each-rollout", "a-directory-is-imported-once"],
)
async def test_a_source_path_is_imported_to_serve_rollouts(
    source: str, imports: int, tmp_path: Path
) -> None:
    log = tmp_path / "imports.log"
    write(
        tmp_path / "counted" / "sums.py",
        f"""
        from pathlib import Path

        from hud import Environment

        with Path({str(log)!r}).open("a") as log:
            log.write("import\\n")

        env = Environment("lab")


        @env.template()
        async def solve():
            yield "go"
            yield 1.0
        """,
    )

    job = await Task(env="lab", id="solve").run(
        ScriptedAgent("x"), runtime=LocalRuntime(tmp_path / source), group=2
    )

    assert [run.reward for run in job.runs] == [1.0, 1.0]
    assert log.read_text().splitlines() == ["import"] * imports


async def test_a_package_entry_serves_the_env_its_relative_import_brings_in(
    tmp_path: Path,
) -> None:
    package = tmp_path / f"envs_{abs(hash(tmp_path)):x}"
    write(package / "__init__.py", "")
    write(
        package / "sums.py",
        """
        from hud import Environment

        env = Environment("lab")


        @env.template()
        async def solve():
            yield "go"
            yield 1.0
        """,
    )
    entry = write(package / "env.py", "from .sums import env\n")

    job = await Task(env="lab", id="solve").run(ScriptedAgent("x"), runtime=LocalRuntime(entry))

    assert job.reward == 1.0


async def test_a_second_copy_of_a_package_from_another_directory_is_refused(
    tmp_path: Path,
) -> None:
    name = f"envs_{abs(hash(tmp_path)):x}"
    for copy in ("first", "second"):
        write(tmp_path / copy / name / "__init__.py", "")
        write(
            tmp_path / copy / name / "env.py",
            """
            from hud import Environment

            env = Environment("lab")


            @env.template()
            async def solve():
                yield "go"
                yield 1.0
            """,
        )
    row = Task(env="lab", id="solve")

    first = await row.run(
        ScriptedAgent("x"), runtime=LocalRuntime(tmp_path / "first" / name / "env.py")
    )
    second = await row.run(
        ScriptedAgent("x"), runtime=LocalRuntime(tmp_path / "second" / name / "env.py")
    )

    assert first.reward == 1.0
    assert f"package {name!r} is already imported from a different source root" in (
        second.runs[0].trace.error or ""
    )


WHERE = """
from pathlib import Path

from hud import Environment

env = Environment("{name}")


@env.template()
async def where():
    yield f"{name} serves from {{Path.cwd()}}"
    yield 1.0
"""


@pytest.mark.parametrize("serve", [LocalRuntime, SubprocessRuntime])
async def test_one_source_serves_each_rows_env_unless_env_pins_one(
    serve: type[LocalRuntime | SubprocessRuntime], tmp_path: Path
) -> None:
    root = tmp_path / "project"
    write(root / "alpha.py", WHERE.format(name="alpha"))
    write(root / "beta.py", WHERE.format(name="beta"))
    source = write(
        root / "env.py", "from alpha import env as alpha\nfrom beta import env as beta\n"
    )
    rows = [Task(env=name, id="where", slug=name) for name in ("alpha", "beta")]
    prompts = []

    class Reader(Agent):
        async def __call__(self, run: Run) -> None:
            prompts.append(run.prompt_text.split(" serves")[0])

    await Taskset("mixed", rows).run(Reader(), runtime=serve(source), max_concurrent=1)
    await Taskset("mixed", rows).run(Reader(), runtime=serve(source, env="beta"), max_concurrent=1)

    assert prompts == ["alpha", "beta", "beta", "beta"]


async def test_a_child_process_serves_from_its_sources_directory(tmp_path: Path) -> None:
    source = write(tmp_path / "project" / "env.py", WHERE.format(name="lab"))
    prompts = []

    class Reader(Agent):
        async def __call__(self, run: Run) -> None:
            prompts.append(run.prompt_text)

    await Task(env="lab", id="where").run(Reader(), runtime=SubprocessRuntime(source))

    assert prompts == [f"lab serves from {tmp_path / 'project'}"]


async def test_a_template_can_import_a_sibling_module_while_it_runs(tmp_path: Path) -> None:
    write(tmp_path / "lazy_helper.py", "ANSWER = 'from the sibling'\n")
    source = write(
        tmp_path / "env.py",
        """
        from hud import Environment

        env = Environment("lab")


        @env.template()
        async def solve():
            import lazy_helper

            answer = yield "go"
            yield 1.0 if answer == lazy_helper.ANSWER else 0.0
        """,
    )

    job = await Task(env="lab", id="solve").run(
        ScriptedAgent("from the sibling"), runtime=LocalRuntime(source)
    )

    assert job.reward == 1.0


async def test_a_constructor_builds_an_env_for_each_rollout_and_they_run_at_once() -> None:
    built = []
    started = asyncio.Event()

    def build(task: Task) -> Environment:
        built.append(task.slug)
        env = Environment("lab")

        @env.initialize
        async def wait_for_the_others() -> None:
            if len(built) == 3:
                started.set()
            await started.wait()

        @env.template()
        async def solve():
            yield "go"
            yield 1.0

        return env

    row = Task(env="lab", id="solve")
    job = await row.run(ScriptedAgent("x"), runtime=LocalRuntime(build, ready_timeout=5), group=3)

    assert ([run.reward for run in job.runs], built) == ([1.0] * 3, [row.slug] * 3)


def test_a_live_environment_serves_rollouts_from_one_event_loop_after_another() -> None:
    env = Environment("lab")

    @env.template()
    async def solve():
        yield "go"
        yield 1.0

    rewards = [asyncio.run(solve().run(ScriptedAgent("x"), group=2)).reward for _ in range(2)]

    assert rewards == [1.0, 1.0]


async def test_a_child_process_serves_a_live_env_from_the_file_its_templates_are_in(
    tmp_path: Path,
) -> None:
    source = write(tmp_path / "env.py", WHERE.format(name="lab"))

    job = await Task(env="lab", id="where").run(
        ScriptedAgent("x"), runtime=SubprocessRuntime(load_environment(source))
    )

    assert job.reward == 1.0


@pytest.mark.filterwarnings("ignore:coroutine 'sleep' was never awaited:RuntimeWarning")
@pytest.mark.parametrize(
    ("source", "row_env", "error"),
    [
        pytest.param(
            "from hud import Environment\n",
            "other",
            "no Environment named 'other'",
            id="a-source-without-the-rows-env",
        ),
        pytest.param(
            "import asyncio\n\nasyncio.run(asyncio.sleep(0))\n",
            "lab",
            "the env source ran async code while being imported to place a rollout — guard "
            'top-level run calls with `if __name__ == "__main__":`',
            id="a-source-that-runs-an-event-loop-at-import",
        ),
    ],
)
async def test_a_source_that_cannot_serve_the_row_fails_the_run(
    source: str, row_env: str, error: str, tmp_path: Path
) -> None:
    path = write(tmp_path / "env.py", source)

    run = await rollout(
        Task(env=row_env, id="solve"), ScriptedAgent("x"), runtime=LocalRuntime(path)
    )

    assert (run.trace.status, run.prompt) == ("error", None)
    assert error in (run.trace.error or "")


async def test_a_constructor_that_returns_something_else_fails_the_run() -> None:
    def build(task: Task) -> Any:
        return "lab"

    run = await rollout(
        Task(env="lab", id="solve"), ScriptedAgent("x"), runtime=LocalRuntime(build)
    )

    assert (
        run.trace.error
        == "[provisioning] TypeError: LocalRuntime: constructor returned 'lab', not an Environment"
    )


async def test_an_in_process_env_is_torn_down_while_a_peer_still_holds_a_connection() -> None:
    env = Environment("lab")

    @env.template()
    async def solve():
        yield "go"
        yield 1.0

    held = []

    class HoldsAConnection(Agent):
        async def __call__(self, run: Run) -> None:
            assert run.runtime is not None
            host, port = run.runtime.removeprefix("tcp://").rsplit(":", 1)
            held.append(await asyncio.open_connection(host, int(port)))

    job = await asyncio.wait_for(solve().run(HoldsAConnection()), timeout=10)

    assert job.reward == 1.0
    ((_, writer),) = held
    writer.close()


async def test_an_env_whose_initialize_raises_is_still_shut_down() -> None:
    events = []
    env = Environment("lab")

    @env.initialize
    async def explode() -> None:
        raise RuntimeError("daemon failed to start")

    @env.shutdown
    async def down() -> None:
        events.append("shut down")

    @env.template()
    async def solve():
        yield "go"
        yield 1.0

    job = await solve().run(ScriptedAgent("x"))

    assert "daemon failed to start" in (job.runs[0].trace.error or "")
    assert events == ["shut down"]


async def test_a_missing_source_fails_the_run(tmp_path: Path) -> None:
    run = await rollout(
        Task(env="lab", id="solve"),
        ScriptedAgent("x"),
        runtime=SubprocessRuntime(tmp_path / "missing.py"),
    )

    assert run.trace.error == (
        f"[provisioning] FileNotFoundError: SubprocessRuntime: source not found: "
        f"{tmp_path / 'missing.py'}"
    )


def gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    status = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, check=False
    )
    return status.stdout.strip().startswith("Z")


async def test_a_child_that_exits_before_serving_reports_its_output_and_takes_its_children(
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
        Task(env="leaky", id="solve"), ScriptedAgent("x"), runtime=SubprocessRuntime(source)
    )

    assert f"spawned env exited with code 3 before serving (source: {source}):\n" in (
        run.trace.error or ""
    )
    assert "daemon starting" in (run.trace.error or "")
    pid = int((tmp_path / "child.pid").read_text())
    try:
        await eventually(lambda: gone(pid), within=5)
    finally:
        if not gone(pid):
            os.kill(pid, signal.SIGKILL)


async def test_a_child_that_closes_its_output_before_serving_fails_the_run(
    tmp_path: Path,
) -> None:
    source = write(
        tmp_path / "env.py",
        """
        import os
        import sys
        import time

        print("closing stdout", file=sys.stderr, flush=True)
        os.close(1)
        time.sleep(60)
        """,
    )

    run = await rollout(
        Task(env="lab", id="solve"), ScriptedAgent("x"), runtime=SubprocessRuntime(source)
    )

    assert "spawned env closed stdout before serving" in (run.trace.error or "")
    assert "closing stdout" in (run.trace.error or "")


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


        @env.template()
        async def solve():
            print("solving", flush=True)
            yield "go"
            yield 1.0
        """,
    )

    job = await Task(env="lab", id="solve").run(
        ScriptedAgent("x"), runtime=SubprocessRuntime(source)
    )

    out, err = capfd.readouterr()
    assert job.reward == 1.0
    assert "x" * 100_000 + "\nenvironment booted\n" in out
    assert "solving\n" in out
    assert "y" * 100_000 + "\nenvironment warning\n" in err
