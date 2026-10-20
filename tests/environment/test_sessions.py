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


async def test_a_session_with_a_live_connection_cannot_be_resumed() -> None:
    async with (
        LocalRuntime(tagged())(Task(env="sessions", id="echo")) as runtime,
        wire(runtime.url) as a,
        wire(runtime.url) as b,
    ):
        session = (await a.call("hello", {}))["result"]["session_id"]
        await a.call("tasks.start", {"id": "echo", "args": {"tag": "a"}})
        refused = await b.call("hello", {"session_id": session})
        graded = await a.call("tasks.grade", {"answer": "x"})

    assert refused["error"]["code"] == -32600
    assert graded["result"]["tag"] == "a"
