"""Control sessions: each connection drives its own task, and a dropped one parks it.

A session is keyed by the id ``hello`` returns. Concurrent connections start and
grade independently; a connection that drops leaves its task parked for a later
connection to grade, by resuming the session (``hello`` with its id) or, when
exactly one session is parked, by a plain ``tasks.grade`` (the split ``hud task
start`` / ``hud task grade`` flow).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from dirty_equals import IsStr

from hud.capabilities import Connection
from hud.clients import HudProtocolError, connect
from hud.environment import Environment, WorkspaceRoute
from hud.eval import LocalRuntime, Task

from .conftest import Wire, wire

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from contextlib import AbstractAsyncContextManager
    from pathlib import Path

ECHO = Task(env="sessions", id="echo")


def echo_env() -> Environment:
    env = Environment("sessions")

    @env.template()
    async def echo(tag: str):
        yield f"go {tag}"
        yield {"score": 1.0, "tag": tag}

    return env


def start(tag: str) -> tuple[str, dict[str, Any]]:
    return "tasks.start", {"id": "echo", "args": {"tag": tag}}


GRADE = ("tasks.grade", {"answer": "x"})

# A step is (connection, action, frame). ``call`` sends the frame and reads the
# reply; ``drop`` sends it and hangs up without reading; ``close`` hangs up. A
# hang-up returns once the server has ended the connection. ``{"resume": "a"}``
# in hello params stands for connection a's session id.
SCRIPTS: list[Any] = [
    pytest.param(
        [
            ("a", "call", start("a")),
            ("b", "call", start("b")),
            ("a", "call", GRADE),
            ("b", "call", GRADE),
        ],
        ["go a", "go b", {"score": 1.0, "tag": "a"}, {"score": 1.0, "tag": "b"}],
        id="concurrent-sessions-grade-their-own-tasks",
    ),
    pytest.param(
        [
            ("b", "call", start("b")),
            ("a", "call", start("first")),
            ("a", "call", start("second")),
            ("a", "call", GRADE),
            ("b", "call", GRADE),
        ],
        [
            "go b",
            "go first",
            "go second",
            {"score": 1.0, "tag": "second"},
            {"score": 1.0, "tag": "b"},
        ],
        id="restart-replaces-only-the-sessions-own-task",
    ),
    pytest.param(
        [
            ("a", "drop", start("parked")),
            ("b", "call", ("hello", {"resume": "a"})),
            ("b", "call", GRADE),
        ],
        [None, "resumed a", {"score": 1.0, "tag": "parked"}],
        id="drop-before-the-prompt-parks-the-task-for-resume",
    ),
    pytest.param(
        [("a", "call", start("parked")), ("a", "close", None), ("b", "call", GRADE)],
        ["go parked", None, {"score": 1.0, "tag": "parked"}],
        id="a-plain-grade-adopts-the-only-parked-session",
    ),
    pytest.param(
        [
            ("a", "call", start("one")),
            ("a", "close", None),
            ("b", "call", start("two")),
            ("b", "close", None),
            ("c", "call", GRADE),
            ("c", "call", ("hello", {"resume": "b"})),
            ("c", "call", GRADE),
        ],
        [
            "go one",
            None,
            "go two",
            None,
            {"error": IsStr(regex=r"2 parked sessions \(sess-\w+, sess-\w+\); resume one .*")},
            "resumed b",
            {"score": 1.0, "tag": "two"},
        ],
        id="several-parked-sessions-need-a-resume",
    ),
    pytest.param(
        [("a", "call", start("a")), ("b", "call", ("hello", {"resume": "a"}))],
        ["go a", {"error": IsStr(regex=r"session 'sess-\w+' has a live connection")}],
        id="a-live-session-cannot-be-resumed",
    ),
]


async def play(url: str, script: list[tuple[str, str, Any]]) -> list[Any]:
    connections: dict[str, Wire] = {}
    sessions: dict[str, str] = {}
    contexts: list[AbstractAsyncContextManager[Wire]] = []
    seen: list[Any] = []
    try:
        for name, action, frame in script:
            if name not in connections:
                context = wire(url)
                contexts.append(context)
                connections[name] = await context.__aenter__()
                sessions[name] = (await connections[name].call("hello", {}))["result"]["session_id"]
            connection = connections[name]
            if action == "close":
                await connection.hang_up()
                seen.append(None)
                continue
            method, params = frame
            if "resume" in params:
                params = {"session_id": sessions[params["resume"]]}
            if action == "drop":
                await connection.send(method, params)
                await connection.hang_up()
                seen.append(None)
                continue
            reply = await connection.call(method, params)
            seen.append(outcome(reply, sessions))
    finally:
        for context in reversed(contexts):
            await context.__aexit__(None, None, None)
    return seen


def outcome(reply: dict[str, Any], sessions: dict[str, str]) -> Any:
    if "error" in reply:
        return {"error": reply["error"]["message"]}
    result = reply["result"]
    if "prompt" in result:
        return result["prompt"]
    if "session_id" in result:
        owner = next(name for name, sid in sessions.items() if sid == result["session_id"])
        return f"resumed {owner}"
    return result


@pytest.mark.parametrize(("script", "expected"), SCRIPTS)
async def test_each_session_keeps_its_own_task(
    script: list[tuple[str, str, Any]], expected: list[Any]
) -> None:
    async with LocalRuntime(echo_env())(ECHO) as runtime:
        assert await play(runtime.url, script) == expected


def inference(token: str) -> Connection:
    return Connection(
        name="inference",
        capability="ssh",
        url="https://inference.hud.so",
        headers={"Authorization": f"Bearer {token}"},
    )


async def a_finished_session_releases_its_connection(env: Environment, root: Path) -> None:
    workspace = env.workspace(root, network=False)
    async with LocalRuntime(env)(ECHO) as runtime:
        async with connect(runtime, connections=[inference("first")]) as first:
            await first.start_task("echo", {"tag": "a"})
            await first.grade({"answer": "x"})
        async with connect(runtime, connections=[inference("rotated")]) as second:
            await second.start_task("echo", {"tag": "b"})
            assert [peer.name for peer in workspace.peers].count("inference.hud.so") == 1
            assert (await second.grade({"answer": "x"}))["tag"] == "b"


async def a_live_session_keeps_its_connection_name(env: Environment, root: Path) -> None:
    env.workspace(root, network=False)
    async with LocalRuntime(env)(ECHO) as runtime, connect(runtime, connections=[inference("a")]):
        with pytest.raises(HudProtocolError, match="bound to another control session"):
            async with connect(runtime, connections=[inference("b")]):
                pass


async def a_finished_session_takes_its_route_with_it(env: Environment, root: Path) -> None:
    workspace = env.workspace(root, network=False)
    route = WorkspaceRoute("ssh", "inference.hud.so", 443)
    async with LocalRuntime(env)(ECHO) as runtime:
        async with connect(runtime, workspace_routes=[route]) as first:
            await first.start_task("echo", {"tag": "a"})
            assert [peer.name for peer in workspace.peers] == ["inference.hud.so"]
            await first.grade({"answer": "x"})
        async with connect(runtime) as later:
            await later.start_task("echo", {"tag": "later"})
            assert workspace.peers == ()
            await later.grade({"answer": "x"})


@pytest.mark.e2e
@pytest.mark.sandbox
@pytest.mark.parametrize(
    "scenario",
    [
        a_finished_session_releases_its_connection,
        a_live_session_keeps_its_connection_name,
        a_finished_session_takes_its_route_with_it,
    ],
)
async def test_a_sessions_workspace_bindings_end_with_the_session(
    tmp_path: Path, scenario: Callable[[Environment, Path], Awaitable[None]]
) -> None:
    await scenario(echo_env(), tmp_path / "root")
