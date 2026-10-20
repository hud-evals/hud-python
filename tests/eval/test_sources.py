"""``LocalRuntime`` and ``SubprocessRuntime`` serve an environment from the source a user
has: a file, a directory, a package, a live instance, or a constructor."""

from __future__ import annotations

import asyncio
import os
import signal
import textwrap
from typing import TYPE_CHECKING

import pytest

from hud import Environment
from hud.agents.base import Agent
from hud.eval import LocalRuntime, SubprocessRuntime, Task, Taskset, rollout
from tests.conftest import eventually
from tests.eval.conftest import gone
from tests.fixtures.agents import ScriptedAgent

if TYPE_CHECKING:
    from pathlib import Path

    from hud.eval import Run


def write(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source))
    return path


LAB = """
from hud import Environment

env = Environment("lab")


@env.template()
async def solve():
    yield "go"
    yield 1.0
"""


async def test_a_package_entry_serves_the_env_its_relative_import_brings_in(
    tmp_path: Path,
) -> None:
    package = tmp_path / f"envs_{abs(hash(tmp_path)):x}"
    write(package / "__init__.py", "")
    write(package / "sums.py", LAB)
    entry = write(package / "env.py", "from .sums import env\n")

    job = await Task(env="lab", id="solve").run(ScriptedAgent("x"), runtime=LocalRuntime(entry))

    assert job.reward == 1.0


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


def test_a_live_environment_serves_rollouts_from_one_event_loop_after_another() -> None:
    env = Environment("lab")

    @env.template()
    async def solve():
        yield "go"
        yield 1.0

    rewards = [asyncio.run(solve().run(ScriptedAgent("x"), group=2)).reward for _ in range(2)]

    assert rewards == [1.0, 1.0]


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

    assert (run.trace.error or "").startswith("[provisioning] ")
    assert "exited with code 3" in (run.trace.error or "")
    assert "daemon starting" in (run.trace.error or "")
    pid = int((tmp_path / "child.pid").read_text())
    try:
        await eventually(lambda: gone(pid), within=5)
    finally:
        if not gone(pid):
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
