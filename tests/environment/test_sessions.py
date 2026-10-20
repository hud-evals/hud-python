"""Control sessions: each connection drives its own task, and a dropped one parks its task
for a later connection to grade, by resuming the session (``hello`` with its id) or,
when exactly one session is parked, by a plain ``tasks.grade``."""

from __future__ import annotations

from hud.environment import Environment
from hud.eval import LocalRuntime, Task
from tests.environment.conftest import wire


def tagged() -> Environment:
    env = Environment("sessions")

    @env.template()
    async def echo(tag: str):
        yield f"go {tag}"
        yield {"score": 1.0, "tag": tag}

    return env


async def test_concurrent_sessions_each_grade_their_own_task() -> None:
    async with (
        LocalRuntime(tagged())(Task(env="sessions", id="echo")) as runtime,
        wire(runtime.url) as a,
        wire(runtime.url) as b,
    ):
        await a.call("tasks.start", {"id": "echo", "args": {"tag": "a"}})
        await b.call("tasks.start", {"id": "echo", "args": {"tag": "b"}})
        graded = [
            (await peer.call("tasks.grade", {"answer": "x"}))["result"]["tag"] for peer in (a, b)
        ]

    assert graded == ["a", "b"]


async def test_a_new_start_replaces_only_its_own_sessions_task() -> None:
    async with (
        LocalRuntime(tagged())(Task(env="sessions", id="echo")) as runtime,
        wire(runtime.url) as a,
        wire(runtime.url) as b,
    ):
        await b.call("tasks.start", {"id": "echo", "args": {"tag": "b"}})
        await a.call("tasks.start", {"id": "echo", "args": {"tag": "first"}})
        await a.call("tasks.start", {"id": "echo", "args": {"tag": "second"}})
        graded = [
            (await peer.call("tasks.grade", {"answer": "x"}))["result"]["tag"] for peer in (a, b)
        ]

    assert graded == ["second", "b"]


async def test_a_session_dropped_before_its_prompt_is_resumed_by_its_id_and_graded() -> None:
    async with LocalRuntime(tagged())(Task(env="sessions", id="echo")) as runtime:
        async with wire(runtime.url) as a:
            session = (await a.call("hello", {}))["result"]["session_id"]
            await a.send("tasks.start", {"id": "echo", "args": {"tag": "parked"}})
            await a.hang_up()
        async with wire(runtime.url) as b:
            resumed = (await b.call("hello", {"session_id": session}))["result"]["session_id"]
            graded = await b.call("tasks.grade", {"answer": "x"})

    assert (resumed, graded["result"]["tag"]) == (session, "parked")


async def test_a_plain_grade_adopts_the_only_parked_session() -> None:
    async with LocalRuntime(tagged())(Task(env="sessions", id="echo")) as runtime:
        async with wire(runtime.url) as a:
            await a.call("tasks.start", {"id": "echo", "args": {"tag": "parked"}})
        async with wire(runtime.url) as b:
            graded = await b.call("tasks.grade", {"answer": "x"})

    assert graded["result"]["tag"] == "parked"


async def test_with_several_parked_sessions_a_grade_must_name_one() -> None:
    sessions = []
    async with LocalRuntime(tagged())(Task(env="sessions", id="echo")) as runtime:
        for tag in ("one", "two"):
            async with wire(runtime.url) as peer:
                sessions.append((await peer.call("hello", {}))["result"]["session_id"])
                await peer.call("tasks.start", {"id": "echo", "args": {"tag": tag}})
        async with wire(runtime.url) as c:
            refused = await c.call("tasks.grade", {"answer": "x"})
            await c.call("hello", {"session_id": sessions[1]})
            graded = await c.call("tasks.grade", {"answer": "x"})

    assert refused["error"]["message"] == (
        f"2 parked sessions ({', '.join(sorted(sessions))}); "
        "resume one by sending hello with its session_id"
    )
    assert graded["result"]["tag"] == "two"


async def test_a_session_with_a_live_connection_cannot_be_resumed() -> None:
    async with (
        LocalRuntime(tagged())(Task(env="sessions", id="echo")) as runtime,
        wire(runtime.url) as a,
        wire(runtime.url) as b,
    ):
        session = (await a.call("hello", {}))["result"]["session_id"]
        refused = await b.call("hello", {"session_id": session})

    assert refused["error"]["message"] == f"session {session!r} has a live connection"
