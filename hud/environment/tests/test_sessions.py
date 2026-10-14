"""Control-channel sessions: each connection drives its own task.

Suspended tasks are keyed by session id: concurrent connections start and
grade independently, a dropped connection parks its task, and a later
connection grades a parked task by resuming the session (``hello`` with its
id) or — only when exactly one is parked — by a plain ``tasks.grade`` (the
split ``hud task start`` / ``hud task grade`` flow).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import pytest

import hud.clients.client as client_module
from hud.capabilities import Connection
from hud.clients import HudProtocolError, connect
from hud.environment import Environment, Workspace, WorkspaceRoute
from hud.environment.utils import read_frame, send_frame
from hud.eval import LocalRuntime, Task
from hud.settings import settings

if TYPE_CHECKING:
    from pathlib import Path

_SESSION = Task(env="sessions", id="echo")


def _env() -> Environment:
    env = Environment("sessions")

    @env.template()
    async def echo(tag: str):
        yield f"go {tag}"
        yield {"score": 1.0, "tag": tag}

    return env


async def test_concurrent_sessions_grade_their_own_tasks() -> None:
    async with (
        LocalRuntime(_env())(_SESSION) as runtime,
        connect(runtime) as a,
        connect(runtime) as b,
    ):
        await a.start_task("echo", {"tag": "a"})
        await b.start_task("echo", {"tag": "b"})  # must not disturb a's task
        assert (await a.grade({"answer": "x"}))["tag"] == "a"
        assert (await b.grade({"answer": "x"}))["tag"] == "b"


async def test_restart_replaces_only_the_sessions_own_task() -> None:
    async with (
        LocalRuntime(_env())(_SESSION) as runtime,
        connect(runtime) as a,
        connect(runtime) as b,
    ):
        await b.start_task("echo", {"tag": "b"})
        await a.start_task("echo", {"tag": "first"})
        await a.start_task("echo", {"tag": "second"})
        assert (await a.grade({"answer": "x"}))["tag"] == "second"
        assert (await b.grade({"answer": "x"}))["tag"] == "b"


@pytest.mark.parametrize("write_error", [None, ConnectionError, asyncio.CancelledError])
async def test_disconnect_parks_the_task_for_a_later_connection(
    monkeypatch: pytest.MonkeyPatch, write_error: type[BaseException] | None
) -> None:
    real_write = asyncio.StreamWriter.write
    monkeypatch.setattr(client_module, "_CONTROL_REDIAL_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(settings, "control_outage_budget", 0.1)

    def write(writer: asyncio.StreamWriter, data: bytes) -> None:
        if write_error is not None and b'"prompt":' in data:
            writer.close()
            raise write_error()
        real_write(writer, data)

    monkeypatch.setattr(asyncio.StreamWriter, "write", write)
    async with LocalRuntime(_env())(_SESSION) as runtime:
        async with connect(runtime) as first:
            assert first.manifest is not None
            session_id = first.manifest.session_id
            if write_error is None:
                await first.start_task("echo", {"tag": "parked"})
            else:
                with pytest.raises(EOFError):
                    await first.start_task("echo", {"tag": "parked"})
        async with connect(runtime) as later:
            await later.hello(session_id=session_id)
            assert (await later.grade({"answer": "x"}))["tag"] == "parked"


async def test_grade_with_multiple_parked_sessions_errors_loudly() -> None:
    async with LocalRuntime(_env())(_SESSION) as runtime:
        for tag in ("one", "two"):
            async with connect(runtime) as client:
                await client.start_task("echo", {"tag": tag})
        async with connect(runtime) as later:
            with pytest.raises(HudProtocolError, match="parked sessions"):
                await later.grade({"answer": "x"})


async def test_hello_resumes_a_parked_session_by_id() -> None:
    async with LocalRuntime(_env())(_SESSION) as runtime:
        ids: dict[str, str] = {}
        for tag in ("one", "two"):
            async with connect(runtime) as client:
                assert client.manifest is not None
                ids[tag] = client.manifest.session_id
                await client.start_task("echo", {"tag": tag})
        async with connect(runtime) as later:
            await later.hello(session_id=ids["two"])
            assert (await later.grade({"answer": "x"}))["tag"] == "two"


async def test_hello_with_an_unknown_session_id_errors() -> None:
    async with LocalRuntime(_env())(_SESSION) as runtime, connect(runtime) as client:
        with pytest.raises(HudProtocolError, match="unknown session"):
            await client.hello(session_id="sess-nope")


async def test_hello_cannot_resume_a_live_session() -> None:
    async with (
        LocalRuntime(_env())(_SESSION) as runtime,
        connect(runtime) as a,
        connect(runtime) as b,
    ):
        assert a.manifest is not None
        await a.start_task("echo", {"tag": "a"})
        with pytest.raises(HudProtocolError, match="live connection"):
            await b.hello(session_id=a.manifest.session_id)


async def test_hello_takes_over_a_live_session_when_asked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(client_module, "_CONTROL_REDIAL_INTERVAL_SECONDS", 0.01)
    async with LocalRuntime(_env())(_SESSION) as runtime, connect(runtime) as a:
        assert a.manifest is not None
        await a.start_task("echo", {"tag": "a"})
        port = urlsplit(runtime.url).port
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        try:
            await send_frame(
                writer,
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "hello",
                    "params": {"session_id": a.manifest.session_id, "takeover": True},
                },
            )
            resumed = await read_frame(reader)
            assert resumed is not None
            assert resumed["result"]["session_id"] == a.manifest.session_id
            await send_frame(
                writer,
                {"jsonrpc": "2.0", "id": 2, "method": "tasks.grade", "params": {"answer": "x"}},
            )
            graded = await read_frame(reader)
            assert graded is not None
            assert graded["result"]["tag"] == "a"
        finally:
            writer.close()
        assert (await a.grade({"answer": "x"}))["tag"] == "a"


async def test_a_finished_session_releases_its_connection_for_the_next_rollout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Workspace, "supports_process_connections", property(lambda _self: True))
    env = _env()
    workspace = env.workspace(tmp_path / "root")
    first, second = (
        Connection(
            name="inference",
            capability="ssh",
            url="https://inference.hud.so",
            headers={"Authorization": f"Bearer {token}"},
        )
        for token in ("first-rollout", "rotated")
    )

    async with LocalRuntime(env)(_SESSION) as runtime:
        async with connect(runtime, connections=[first]) as a:
            await a.start_task("echo", {"tag": "a"})
            assert (await a.grade({"answer": "x"}))["tag"] == "a"
        async with connect(runtime, connections=[second]) as b:
            await b.start_task("echo", {"tag": "b"})
            assert [peer.name for peer in workspace.peers].count(second.host) == 1
            assert (await b.grade({"answer": "x"}))["tag"] == "b"


async def test_a_live_session_keeps_its_connection_name_from_other_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Workspace, "supports_process_connections", property(lambda _self: True))
    env = _env()
    env.workspace(tmp_path / "root")
    first, second = (
        Connection(
            name="inference",
            capability="ssh",
            url="https://inference.hud.so",
            headers={"Authorization": f"Bearer {token}"},
        )
        for token in ("first", "second")
    )

    async with LocalRuntime(env)(_SESSION) as runtime, connect(runtime, connections=[first]):
        with pytest.raises(HudProtocolError, match="bound to another control session"):
            async with connect(runtime, connections=[second]):
                pass


async def test_a_finished_session_takes_its_workspace_route_with_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Workspace, "bwrap_available", property(lambda _self: True))
    env = _env()
    workspace = env.workspace(tmp_path / "root", network=False)
    route = WorkspaceRoute("ssh", "inference.hud.so", 443)

    async with LocalRuntime(env)(_SESSION) as runtime:
        async with connect(runtime, workspace_routes=[route]) as a:
            await a.start_task("echo", {"tag": "a"})
            assert [peer.name for peer in workspace.peers] == ["inference.hud.so"]
            await a.grade({"answer": "x"})
        async with connect(runtime) as later:
            await later.start_task("echo", {"tag": "later"})
            assert workspace.peers == ()
            await later.grade({"answer": "x"})


def _counting_env() -> tuple[Environment, dict[str, int]]:
    env = Environment("sessions")
    counts = {"starts": 0, "grades": 0}

    @env.template()
    async def echo(tag: str):
        counts["starts"] += 1
        yield f"go {tag}"
        counts["grades"] += 1
        yield {"score": 1.0, "tag": tag}

    return env, counts


@pytest.mark.parametrize("lost_reply", [b'"prompt":', b'"score":'])
async def test_a_task_call_whose_reply_was_lost_is_answered_once_after_the_redial(
    monkeypatch: pytest.MonkeyPatch, lost_reply: bytes
) -> None:
    monkeypatch.setattr(client_module, "_CONTROL_REDIAL_INTERVAL_SECONDS", 0.01)
    real_write = asyncio.StreamWriter.write
    dropped: list[bytes] = []

    def write(writer: asyncio.StreamWriter, data: bytes) -> None:
        if lost_reply in data and not dropped:
            dropped.append(data)
            writer.close()
            raise ConnectionResetError
        real_write(writer, data)

    monkeypatch.setattr(asyncio.StreamWriter, "write", write)
    env, counts = _counting_env()
    async with LocalRuntime(env)(_SESSION) as runtime, connect(runtime) as client:
        assert (await client.start_task("echo", {"tag": "a"}))["prompt"] == "go a"
        assert (await client.grade({"answer": "x"}))["tag"] == "a"

    assert dropped
    assert counts == {"starts": 1, "grades": 1}


async def test_a_new_start_discards_the_kept_grade() -> None:
    env, counts = _counting_env()
    async with LocalRuntime(env)(_SESSION) as runtime, connect(runtime) as client:
        await client.start_task("echo", {"tag": "first"})
        assert (await client.grade({"answer": "x"}))["tag"] == "first"
        await client.start_task("echo", {"tag": "second"})
        assert (await client.grade({"answer": "x"}))["tag"] == "second"

    assert counts == {"starts": 2, "grades": 2}


async def test_a_start_retried_on_a_new_connection_joins_the_start_in_flight() -> None:
    env = Environment("sessions")
    starts = 0
    release = asyncio.Event()

    @env.template()
    async def slow():
        nonlocal starts
        starts += 1
        await release.wait()
        yield "ready"
        yield 1.0

    async def call(
        writer: asyncio.StreamWriter, msg_id: int, method: str, **params: object
    ) -> None:
        await send_frame(
            writer, {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params}
        )

    async with LocalRuntime(env)(Task(env="sessions", id="slow")) as runtime:
        port = urlsplit(runtime.url).port
        first_reader, first_writer = await asyncio.open_connection("127.0.0.1", port)
        second_reader, second_writer = await asyncio.open_connection("127.0.0.1", port)
        try:
            await call(first_writer, 1, "hello")
            hello = await read_frame(first_reader)
            assert hello is not None
            session_id = hello["result"]["session_id"]
            await call(first_writer, 2, "tasks.start", id="slow", call_id="c1")
            await call(second_writer, 1, "hello", session_id=session_id, takeover=True)
            assert (await read_frame(second_reader)) is not None
            await call(second_writer, 2, "tasks.start", id="slow", call_id="c1")
            release.set()
            started = await asyncio.wait_for(read_frame(second_reader), timeout=5.0)
        finally:
            first_writer.close()
            second_writer.close()

    assert started is not None
    assert started["result"]["prompt"] == "ready"
    assert starts == 1


async def test_hello_resumes_a_session_that_holds_no_task() -> None:
    async with LocalRuntime(_env())(_SESSION) as runtime:
        async with connect(runtime) as first:
            assert first.manifest is not None
            session_id = first.manifest.session_id
        async with connect(runtime) as later:
            await later.hello(session_id=session_id)
            assert later.manifest is not None
            assert later.manifest.session_id == session_id


async def test_hello_cannot_resume_a_session_that_said_bye() -> None:
    async with LocalRuntime(_env())(_SESSION) as runtime:
        reader, writer = await asyncio.open_connection("127.0.0.1", urlsplit(runtime.url).port)
        try:
            await send_frame(writer, {"jsonrpc": "2.0", "id": 1, "method": "hello"})
            hello = await read_frame(reader)
            assert hello is not None
            session_id = hello["result"]["session_id"]
            await send_frame(writer, {"jsonrpc": "2.0", "id": 2, "method": "bye"})
            assert await read_frame(reader) is not None
        finally:
            writer.close()
        async with connect(runtime) as later:
            with pytest.raises(HudProtocolError, match="unknown session"):
                await later.hello(session_id=session_id)
