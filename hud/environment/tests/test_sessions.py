"""Control-channel sessions: each connection drives its own task.

Suspended tasks are keyed by session id: concurrent connections start and
grade independently, a dropped connection parks its task, and a later
connection grades a parked task by resuming the session (``hello`` with its
id) or — only when exactly one is parked — by a plain ``tasks.grade`` (the
split ``hud task start`` / ``hud task grade`` flow).
"""

from __future__ import annotations

import pytest

from hud.clients import HudProtocolError, connect
from hud.environment import Environment
from hud.eval.runtime import _local


def _env() -> Environment:
    env = Environment("sessions")

    @env.template()
    async def echo(tag: str):
        yield f"go {tag}"
        yield {"score": 1.0, "tag": tag}

    return env


async def test_concurrent_sessions_grade_their_own_tasks() -> None:
    async with _local(_env()) as runtime, connect(runtime) as a, connect(runtime) as b:
        await a.start_task("echo", {"tag": "a"})
        await b.start_task("echo", {"tag": "b"})  # must not disturb a's task
        assert (await a.grade({"answer": "x"}))["tag"] == "a"
        assert (await b.grade({"answer": "x"}))["tag"] == "b"


async def test_restart_replaces_only_the_sessions_own_task() -> None:
    async with _local(_env()) as runtime, connect(runtime) as a, connect(runtime) as b:
        await b.start_task("echo", {"tag": "b"})
        await a.start_task("echo", {"tag": "first"})
        await a.start_task("echo", {"tag": "second"})
        assert (await a.grade({"answer": "x"}))["tag"] == "second"
        assert (await b.grade({"answer": "x"}))["tag"] == "b"


async def test_disconnect_parks_the_task_for_a_later_connection() -> None:
    async with _local(_env()) as runtime:
        async with connect(runtime) as first:
            await first.start_task("echo", {"tag": "parked"})
        async with connect(runtime) as later:
            assert (await later.grade({"answer": "x"}))["tag"] == "parked"


async def test_grade_with_multiple_parked_sessions_errors_loudly() -> None:
    async with _local(_env()) as runtime:
        for tag in ("one", "two"):
            async with connect(runtime) as client:
                await client.start_task("echo", {"tag": tag})
        async with connect(runtime) as later:
            with pytest.raises(HudProtocolError, match="parked sessions"):
                await later.grade({"answer": "x"})


async def test_hello_resumes_a_parked_session_by_id() -> None:
    async with _local(_env()) as runtime:
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
    async with _local(_env()) as runtime, connect(runtime) as client:
        with pytest.raises(HudProtocolError, match="unknown session"):
            await client.hello(session_id="sess-nope")


async def test_hello_cannot_resume_a_live_session() -> None:
    async with _local(_env()) as runtime, connect(runtime) as a, connect(runtime) as b:
        assert a.manifest is not None
        await a.start_task("echo", {"tag": "a"})
        with pytest.raises(HudProtocolError, match="live connection"):
            await b.hello(session_id=a.manifest.session_id)
