"""Serving an environment: the entry points that own a process, and the lifecycle they run.

Both entry points, ``python -m hud.environment.server`` (what images run) and
``hud serve``, are rows of each table. They resolve a reference in a fixture tree
whose environments leave a ``stopped-<name>`` marker from a shutdown hook, print
``HUD_SERVE_PORT=<port>`` once the control channel accepts, and stop cleanly on a
signal, even with a session still grading. A reference that does not load ends
the process with the error and without a port.
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from typing import TYPE_CHECKING

import pytest

from hud.environment import Environment
from tests.support import served, wire

if TYPE_CHECKING:
    from pathlib import Path


def declare(name: str, variable: str = "env") -> str:
    """Source declaring Environment ``name``, whose shutdown hook leaves a marker in cwd."""
    return (
        "from pathlib import Path\n"
        "from hud import Environment\n"
        f"{variable} = Environment({name!r})\n"
        f"@{variable}.shutdown\n"
        f"async def _stopped_{variable}():\n"
        f"    Path('stopped-{name}').write_text('stopped')\n"
    )


TREE = {
    "env.py": declare("from-env-py"),
    "pkg/__init__.py": (
        "from pathlib import Path\n"
        "from hud.environment import Environment\n"
        "def make_env(name='from-factory'):\n"
        "    env = Environment(name)\n"
        "    @env.shutdown\n"
        "    async def _stopped():\n"
        "        Path(f'stopped-{name}').write_text('stopped')\n"
        "    return env\n"
    ),
    "multi.py": declare("env-one", "first") + declare("env-two", "second"),
    "stalls.py": declare("stalls")
    + "import asyncio\n"
    + '@env.template(id="stall")\nasync def stall():\n    yield "go"\n'
    + "    await asyncio.Event().wait()\n    yield 1.0\n",
}

SERVER = [sys.executable, "-m", "hud.environment.server"]
HUD_SERVE = [sys.executable, "-m", "hud.cli", "serve", "--port", "0"]


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    for name, content in TREE.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return root


async def announced_port(process: asyncio.subprocess.Process) -> int:
    assert process.stdout is not None
    async with asyncio.timeout(60):
        while line := await process.stdout.readline():
            text = line.decode().strip()
            if text.startswith("HUD_SERVE_PORT="):
                return int(text.removeprefix("HUD_SERVE_PORT="))
    assert process.stderr is not None
    raise AssertionError(f"exited without a port:\n{(await process.stderr.read()).decode()}")


@pytest.mark.parametrize(
    ("command", "served_name", "stop", "grading"),
    [
        pytest.param([*SERVER, "env.py"], "from-env-py", signal.SIGTERM, False, id="server"),
        pytest.param([*HUD_SERVE, "env:env"], "from-env-py", signal.SIGINT, False, id="hud-serve"),
        pytest.param(
            [*HUD_SERVE, "pkg:make_env", "--arg", "name=demo"],
            "demo",
            signal.SIGTERM,
            False,
            id="hud-serve-factory",
        ),
        pytest.param(
            [*SERVER, "stalls.py"], "stalls", signal.SIGTERM, True, id="server-while-grading"
        ),
        pytest.param(
            [*HUD_SERVE, "stalls:env"],
            "stalls",
            signal.SIGTERM,
            True,
            id="hud-serve-while-grading",
        ),
    ],
)
async def test_a_serving_process_announces_its_port_and_runs_shutdown_hooks_on_a_signal(
    tree: Path, command: list[str], served_name: str, stop: signal.Signals, grading: bool
) -> None:
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=tree,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, "NO_COLOR": "1"},
    )
    try:
        port = await announced_port(process)
        async with wire(f"tcp://127.0.0.1:{port}") as control:
            hello = await control.call("hello", {})
            if grading:
                await control.call("tasks.start", {"id": "stall", "args": {}})
                await control.send("tasks.grade", {"answer": "x"})
            process.send_signal(stop)
            code = await asyncio.wait_for(process.wait(), timeout=10)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()

    assert hello["result"]["env"]["name"] == served_name
    assert code == 0
    assert [name for name in os.listdir(tree) if name.startswith("stopped-")] == [
        f"stopped-{served_name}"
    ]


@pytest.mark.parametrize(
    ("command", "code", "error"),
    [
        pytest.param(
            [*SERVER, "multi.py"], 1, "multiple Environments in multi.py", id="server-ambiguous"
        ),
        pytest.param(
            [*HUD_SERVE, "multi.py"],
            2,
            "multiple Environments in multi.py",
            id="hud-serve-ambiguous",
        ),
        pytest.param(
            [*HUD_SERVE, "pkg:make_env", "--arg", "name"], 2, "--arg", id="hud-serve-malformed-arg"
        ),
        pytest.param(
            [*SERVER, "missing.py"], 1, "no environment source at missing.py", id="server-missing"
        ),
        pytest.param(
            [*HUD_SERVE, "missing.py"],
            1,
            "no environment source at missing.py",
            id="hud-serve-missing",
        ),
    ],
)
async def test_a_serving_process_exits_with_the_load_error(
    tree: Path, command: list[str], code: int, error: str
) -> None:
    process = await asyncio.create_subprocess_exec(
        *command, cwd=tree, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=60)

    assert process.returncode == code
    assert b"HUD_SERVE_PORT=" not in stdout
    assert error in stderr.decode()


async def test_initialize_hooks_finish_before_hello_and_shutdown_hooks_run_in_reverse() -> None:
    env = Environment("ordered")
    log: list[str] = []

    @env.initialize
    async def first_up() -> None:
        await asyncio.sleep(0.1)
        log.append("first up")

    @env.initialize
    async def second_up() -> None:
        log.append("second up")

    @env.shutdown
    async def first_down() -> None:
        log.append("first down")

    @env.shutdown
    async def second_down() -> None:
        log.append("second down")

    async with served(env):
        seen_at_hello = list(log)

    assert seen_at_hello == ["first up", "second up"]
    assert log == ["first up", "second up", "second down", "first down"]
