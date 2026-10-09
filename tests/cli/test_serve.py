"""``hud serve``: serve an environment's control channel until interrupted."""

from __future__ import annotations

import signal
from typing import TYPE_CHECKING

import pytest

from hud.clients import connect
from hud.environment.server import PORT_ANNOUNCEMENT
from hud.eval import Runtime

if TYPE_CHECKING:
    from pathlib import Path

    from tests.harness import Hud

ENV_PY = """\
from hud import Environment

env = Environment("greeter")


@env.template(id="greet")
async def greet(name: str = "world"):
    answer = yield f"Greet {name}."
    yield 1.0 if answer == f"hello {name}" else 0.0
"""
FACTORY_PY = """\
from hud import Environment


def make_env(name: str) -> Environment:
    env = Environment(name)

    @env.template(id="greet")
    async def greet():
        answer = yield f"Greet {name}."
        yield 1.0 if answer == f"hello {name}" else 0.0

    return env
"""


def write_sources(directory: Path) -> None:
    (directory / "env.py").write_text(ENV_PY)
    (directory / "pkg").mkdir()
    (directory / "pkg" / "__init__.py").write_text("")
    (directory / "pkg" / "factory.py").write_text(FACTORY_PY)


@pytest.mark.parametrize(
    ("argv", "name"),
    [
        (["env.py"], "world"),
        ([], "world"),
        (["env:env"], "world"),
        (["pkg.factory:make_env", "--arg", "name=demo"], "demo"),
    ],
)
async def test_serve_answers_tasks_until_interrupted(hud: Hud, argv: list[str], name: str) -> None:
    write_sources(hud.cwd)
    process = hud.start("serve", *argv, "--port", "0")
    assert process.stdout is not None and process.stderr is not None
    announced = process.stdout.readline()
    assert announced.startswith(PORT_ANNOUNCEMENT), process.stderr.read()
    url = f"tcp://127.0.0.1:{announced.removeprefix(PORT_ANNOUNCEMENT).strip()}"

    try:
        async with connect(Runtime(url)) as client:
            started = await client.start_task("greet", {})
            graded = await client.grade({"answer": f"hello {name}"})
    finally:
        process.send_signal(signal.SIGINT)
        _, stderr = process.communicate(timeout=30)

    assert started["prompt"] == f"Greet {name}."
    assert graded["score"] == 1.0
    assert process.returncode == 0
    banner = " ".join(stderr.split())
    expected = "greeter" if name == "world" else "demo"
    assert f"• {expected} • serving on tcp://127.0.0.1:" in banner
    assert "• 1 task(s), 0 capability(ies)" in banner
    assert "Press Ctrl+C to stop." in banner
    assert stderr.rstrip().endswith("Stopped.")


@pytest.mark.parametrize(
    ("argv", "exit_code", "document"),
    [
        (
            ["pkg.factory:make_env", "--arg", "name"],
            2,
            {"error": "usage", "message": "--arg expects key=value, got 'name'"},
        ),
        (["missing.py"], 1, {"error": "failure", "message": "No module named 'missing'"}),
    ],
)
def test_serve_rejects_a_bad_target_before_serving(
    hud: Hud, argv: list[str], exit_code: int, document: dict[str, str]
) -> None:
    write_sources(hud.cwd)

    result = hud("serve", *argv, "--port", "0", "--json")

    assert result.exit_code == exit_code, result
    assert result.json == document
