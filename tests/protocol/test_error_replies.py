"""A bad request gets an error reply, and the server keeps serving.

Every JSON-RPC server an environment runs is a row in ``SERVERS``. A case sends
its bad frames on one connection and checks each reply's error code, then sends
the server's ordinary request on the same connection and checks that it is
answered and that nothing the bad frames asked for ran: a template that starts
leaves a log in the server's ``effects`` directory.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest

from hud.capabilities import Capability
from hud.environment import Environment
from hud.eval import LocalRuntime, Task
from tests.support import served, wire

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable
    from contextlib import AbstractAsyncContextManager
    from pathlib import Path

    from tests.support import Wire


@asynccontextmanager
async def control_channel(effects: Path) -> AsyncIterator[Wire]:
    env = Environment(
        "errors", capabilities=[Capability.cdp(url="ws://127.0.0.1:9/devtools/browser")]
    )

    @env.template()
    async def guarded(name: str):
        (effects / "log").write_text(f"started {name}\n")
        yield "go"
        yield 1.0

    @env.template()
    async def lookup():
        settings: dict[str, str] = {}
        yield settings["missing"]

    async with (
        LocalRuntime(env)(Task(env=env.name, id="errors")) as runtime,
        wire(runtime.url) as connection,
    ):
        yield connection


@asynccontextmanager
async def file_tracking(effects: Path) -> AsyncIterator[Wire]:
    root = effects.parent / "workspace"
    root.mkdir()
    (root / "a.txt").write_text("one\n")
    env = Environment("tracked")
    env.workspace(root, track_files=True)
    async with served(env) as client, wire(client.binding("filetracking").url) as connection:
        yield connection


@dataclass(frozen=True)
class Server:
    open: Callable[[Path], AbstractAsyncContextManager[Wire]]
    #: A request the server answers with a result.
    ordinary: str


SERVERS = {
    "control": Server(control_channel, "tasks.list"),
    "filetracking": Server(file_tracking, "snapshot"),
}
# The robot control port is served by a robot bridge, not by an Environment;
# the robot area holds its wire.

INVALID, NOT_FOUND, BAD_STATE, FAILED = -32602, -32601, -32600, -32000
ROUTE = {"capability": "ssh", "host": "h", "port": 443}
CONNECTION = {
    "name": "inference",
    "capability": "ssh",
    "url": "https://inference.example",
    "headers": {"Authorization": "Bearer t"},
}
Frames = list[tuple[str, dict[str, Any] | None]]
CASES: dict[str, tuple[str, Frames, list[Any]]] = {
    "start-unknown-template": ("control", [("tasks.start", {"id": "missing"})], [INVALID]),
    "start-list-args": ("control", [("tasks.start", {"id": "guarded", "args": ["x"]})], [INVALID]),
    "start-signature-mismatch": (
        "control",
        [("tasks.start", {"id": "guarded", "args": {"nmae": "x"}})],
        [FAILED],
    ),
    "start-template-raises-keyerror": ("control", [("tasks.start", {"id": "lookup"})], [FAILED]),
    "grade-without-start": ("control", [("tasks.grade", {"answer": "x"})], [BAD_STATE]),
    "unknown-control-method": ("control", [("tasks.stop", {})], [NOT_FOUND]),
    "tunnel-open-mid-session": (
        "control",
        [("tasks.cancel", None), ("tunnel.open", {"capability": "browser"})],
        [{"cancelled": True}, NOT_FOUND],
    ),
    "hello-unknown-session": ("control", [("hello", {"session_id": "sess-00000000"})], [BAD_STATE]),
    "hello-route-port-not-an-int": (
        "control",
        [("hello", {"workspace_routes": [{**ROUTE, "port": "443"}]})],
        [INVALID],
    ),
    "hello-connection-without-a-workspace": (
        "control",
        [("hello", {"connections": [CONNECTION]})],
        [FAILED],
    ),
    "unknown-filetracking-method": ("filetracking", [("rewind", None)], [FAILED]),
}


@pytest.mark.parametrize(("server", "frames", "expected"), CASES.values(), ids=CASES.keys())
async def test_a_bad_request_gets_an_error_reply_and_the_server_keeps_serving(
    tmp_path: Path, server: str, frames: Frames, expected: list[Any]
) -> None:
    effects = tmp_path / "effects"
    effects.mkdir()

    async with SERVERS[server].open(effects) as connection:
        replies = [await connection.call(method, params) for method, params in frames]
        after = await connection.call(SERVERS[server].ordinary)

    assert [
        reply["result"] if "result" in reply else reply["error"]["code"] for reply in replies
    ] == expected
    assert "result" in after
    assert list(effects.iterdir()) == []
