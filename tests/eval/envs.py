"""Fixture environments for the rollout engine scenarios.

Each template's reward proves the path it names ran: ``add`` pays 1.0 only for
the right sum, so a graded run shows the prompt reached the agent and the answer
reached the grader. Hooks append to an ``events`` list the test owns.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from hud import Environment
from hud.environment.env import current_session_id
from hud.eval import LocalRuntime, Task

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable
    from pathlib import Path

    from tests.harness import FakeDocker

SUMS_SOURCE = """
from hud import Environment

env = Environment("lab")


@env.template()
async def add(a: int, b: int):
    answer = yield f"add {a} {b}"
    yield 1.0 if answer == str(a + b) else 0.0
"""


async def eventually(condition: Callable[[], bool], *, within: float = 10.0) -> None:
    """Wait for work a rollout left running in the background to finish."""
    deadline = asyncio.get_running_loop().time() + within
    while not condition():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(f"condition still false after {within}s")
        await asyncio.sleep(0.01)


def solve(prompt: str) -> str:
    """The right answer to an ``add`` prompt."""
    _, a, b = prompt.split()
    return str(int(a) + int(b))


def lab(events: list[str] | None = None) -> Environment:
    """The ``lab`` environment: one template per way a task can start or grade."""
    log = events if events is not None else []
    env = Environment("lab")

    @env.initialize
    async def initialize() -> None:
        log.append("initialize")

    @env.shutdown
    async def shutdown() -> None:
        log.append("shutdown")

    @env.template()
    async def add(a: int, b: int):
        log.append(f"start add {a} {b}")
        try:
            answer = yield f"add {a} {b}"
            log.append(f"grade {answer}")
            yield 1.0 if answer == str(a + b) else 0.0
        finally:
            log.append("end add")

    @env.template()
    async def chat(messages: list[dict[str, Any]]):
        yield messages
        yield 1.0

    @env.template()
    async def silent():
        yield None
        yield 1.0

    @env.template()
    async def claim(published: Any):
        yield {"prompt": "go", "bindings": published}
        yield 1.0

    @env.template()
    async def grade_raises():
        yield "go"
        raise RuntimeError("grader exploded")
        yield 0.0

    @env.template()
    async def frame(result: dict[str, Any]):
        yield "go"
        yield result

    @env.template()
    async def hang_grading():
        try:
            yield "go"
            await asyncio.Event().wait()
            yield 1.0
        finally:
            log.append("grading cancelled")

    @env.template()
    async def hang_start():
        try:
            await asyncio.Event().wait()
            yield "never"
            yield 0.0
        finally:
            log.append("start cancelled")

    @env.template()
    async def large(criteria: str):
        log.append("large started")
        yield f"{len(criteria)}"
        yield 1.0

    return env


def minted(events: list[str] | None = None) -> Task:
    """A row minted by a live ``lab`` environment, which places it by default."""
    env = lab(events)

    @env.template()
    async def total(a: int, b: int):
        answer = yield f"add {a} {b}"
        yield 1.0 if answer == str(a + b) else 0.0

    return total(a=2, b=3)


def actor(
    env: Environment | None = None,
    *,
    grade: str = "score",
    sessions: Path | None = None,
    link: bool = False,
) -> Environment:
    """Declare the actor side of a verifier pair: ``solve`` grades 0.25 and carries the answer.

    It goes on ``env``, or on a new ``actor`` environment. ``grade="raise"`` makes
    its grading raise and ``grade="scoreless"`` yields a frame without a score.
    With ``sessions``, the template writes the answer into its control session's
    directory under that root, as an environment in a container would, plus a
    symbolic link when ``link`` is set.
    """
    env = env or Environment("actor")

    @env.template()
    async def solve():
        answer = yield "answer secret"
        if sessions is not None:
            session = sessions / "runtime" / "sessions" / str(current_session_id.get())
            session.mkdir(parents=True)
            (session / "work.txt").write_text(str(answer))
            if link:
                (session / "escape").symlink_to("/etc/passwd")
        if grade == "raise":
            raise RuntimeError("actor grade exploded")
        yield {"score": 0.25, "answer": answer} if grade == "score" else {"answer": answer}

    return env


def judge(
    env: Environment | None = None, *, verdict: str = "check", sessions: Path | None = None
) -> Environment:
    """Declare the judge side: ``verify`` pays 1.0 when the actor's answer was ``secret``.

    It goes on ``env``, or on a new ``judge`` environment. ``verdict`` picks a
    failure instead: ``"raise"``, ``"scoreless"`` or ``"zero"``. With
    ``sessions``, it reads the answer from its restored session directory and
    pays 0.0 when nothing was restored.
    """
    env = env or Environment("judge")

    @env.template()
    async def verify():
        result = yield ""
        if verdict == "raise":
            raise RuntimeError("verifier exploded")
        if verdict == "scoreless":
            yield {"verdict": "pass"}
        elif verdict == "zero":
            yield 0.0
        elif sessions is not None:
            session = sessions / "runtime" / "sessions" / str(current_session_id.get())
            work = session / "work.txt"
            yield 1.0 if work.exists() and work.read_text() == "secret" else 0.0
        else:
            yield 1.0 if result["answer"] == "secret" else 0.0

    return env


DF_FREE = (
    "Filesystem 1024-blocks Used Available Capacity Mounted on\n"
    "overlay 104857600 0 104857600 0% /\n"
)


def container(image: str) -> str:
    """The name the fake docker gives a container started from ``image``."""
    return re.sub(r"[^\w.-]", "-", image)


@asynccontextmanager
async def containers(
    fake_docker: FakeDocker, rootfs: Path, images: dict[str, Environment]
) -> AsyncIterator[dict[str, str]]:
    """Make each image a container the fake docker starts, served in this process.

    ``docker run <image>`` answers the container :func:`container` names,
    ``docker port`` the address its environment serves on here, and ``docker
    cp``/``exec`` work on ``<rootfs>/<container>``. Yields each image's address.
    """
    fake_docker.on(r"^(cp|exec) ", rootfs=rootfs)
    addresses: dict[str, str] = {}
    async with contextlib.AsyncExitStack() as stack:
        for image, env in images.items():
            runtime = await stack.enter_async_context(
                LocalRuntime(env)(Task(env=env.name, id="serve"))
            )
            name = container(image)
            (rootfs / name).mkdir(parents=True, exist_ok=True)
            addresses[image] = runtime.url.removeprefix("tcp://")
            fake_docker.on(rf"^run .* {re.escape(image)}$", stdout=f"{name}\n")
            fake_docker.on(rf"^port {re.escape(name)} 8765$", stdout=f"{addresses[image]}\n")
        yield addresses
