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

import pytest

from hud.capabilities import Connection
from hud.clients import HudProtocolError, connect
from hud.environment import Environment, Workspace, WorkspaceRoute
from hud.eval import LocalRuntime, Task

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
