"""The control channel as a peer speaks it: a task's typed start and answer, how a
session's template is torn down, a request at the frame limit, sessions that drop and
resume, and capability streams tunneled through its port."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlsplit

import pytest
from pydantic import BaseModel

from hud.capabilities import Capability
from hud.clients import connect
from hud.environment import (
    Environment,
)
from hud.eval import LocalRuntime, Task
from tests.environment.conftest import FRAME_LIMIT, encode, wire

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

INVALID, NOT_FOUND, BAD_STATE, FAILED = -32602, -32601, -32600, -32000


class Point(BaseModel):
    x: int
    y: int


class Payload(BaseModel):
    text: str


@pytest.mark.parametrize(
    ("args", "prompt"),
    [
        ({"mode": '"upper"', "payload": '{"text":"hello"}', "retries": "3"}, "HELLO!!!"),
        ({"mode": "lower", "payload": {"text": "HeLLo"}}, "hello"),
        ({"mode": "upper", "payload": {"text": "hi"}, "retries": None}, "HI"),
    ],
    ids=["json-strings-are-parsed-to-their-types", "native-values-validate", "none-is-kept"],
)
async def test_start_arguments_arrive_as_their_annotated_types(
    args: dict[str, Any], prompt: str
) -> None:
    env = Environment("typed")

    @env.template()
    async def shout(mode: Literal["upper", "lower"], payload: Payload, retries: int | None = None):
        text = payload.text.upper() if mode == "upper" else payload.text.lower()
        yield text + "!" * (retries or 0)
        yield 1.0

    async with LocalRuntime(env)(Task(env="typed", id="x")) as runtime, wire(runtime.url) as peer:
        started = await peer.call("tasks.start", {"id": "shout", "args": args})

    assert started["result"] == {"prompt": prompt}


@pytest.mark.parametrize(
    ("answer", "content"),
    [('{"x": 1, "y": 2}', {"x": 1, "y": 2}), ("about (1, 2)", "about (1, 2)")],
    ids=["an-answer-of-the-type-is-parsed", "any-other-answer-arrives-as-written"],
)
async def test_a_declared_return_type_parses_the_answer_and_keeps_it_raw(
    answer: str, content: Any
) -> None:
    env = Environment("structured")

    @env.template(returns=Point)
    async def point():
        answer = yield "Name a point."
        parsed = answer.content
        yield {
            "score": 1.0 if isinstance(parsed, Point) else 0.0,
            "content": parsed.model_dump() if isinstance(parsed, Point) else parsed,
            "raw": answer.raw,
        }

    async with (
        LocalRuntime(env)(Task(env="structured", id="x")) as runtime,
        wire(runtime.url) as peer,
    ):
        await peer.call("tasks.start", {"id": "point"})
        graded = (await peer.call("tasks.grade", {"answer": answer}))["result"]

    assert (graded["content"], graded["raw"]) == (content, answer)


@pytest.mark.parametrize(
    ("method", "params", "reply"),
    [
        ("tasks.cancel", None, {"cancelled": True}),
        ("bye", None, {"goodbye": True}),
        ("tasks.grade", {"answer": "x"}, {"score": 1.0}),
        ("tasks.start", {"id": "hold", "args": {"name": "second"}}, {"prompt": "holding second"}),
    ],
    ids=["cancel", "bye", "grade", "a-new-start"],
)
async def test_ending_a_sessions_task_closes_its_template(
    method: str, params: dict[str, Any] | None, reply: dict[str, Any]
) -> None:
    events = []
    env = Environment("held")

    @env.template()
    async def hold(name: str):
        try:
            yield f"holding {name}"
            yield 1.0
        finally:
            events.append(f"closed {name}")

    async with LocalRuntime(env)(Task(env="held", id="x")) as runtime, wire(runtime.url) as peer:
        await peer.call("tasks.start", {"id": "hold", "args": {"name": "first"}})
        answered = await peer.call(method, params)
        closed = list(events)

    assert (answered["result"], closed) == (reply, ["closed first"])


@pytest.mark.parametrize(
    ("method", "params", "reply"),
    [
        ("tasks.cancel", None, {"cancelled": True}),
        ("bye", None, {"goodbye": True}),
        ("tasks.grade", {"answer": "x"}, {"score": 1.0}),
    ],
    ids=["cancel", "bye", "grade"],
)
async def test_a_template_failing_to_close_is_logged_and_its_session_still_ends(
    method: str,
    params: dict[str, Any] | None,
    reply: dict[str, Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    env = Environment("leaky")

    @env.template()
    async def hold():
        try:
            yield "holding"
            yield 1.0
        finally:
            raise RuntimeError("simulator would not stop")

    async with LocalRuntime(env)(Task(env="leaky", id="x")) as runtime, wire(runtime.url) as peer:
        await peer.call("tasks.start", {"id": "hold"})
        answered = await peer.call(method, params)

    assert answered["result"] == reply
    (record,) = [r for r in caplog.records if "'hold'" in r.getMessage()]
    assert record.levelname == "WARNING"
    assert record.exc_info is not None
    assert "simulator would not stop" in str(record.exc_info[1])


def stalled(events: list[str]) -> Environment:
    env = Environment("stalled")

    @env.template()
    async def stall():
        try:
            yield "go"
            await asyncio.Event().wait()
            yield 1.0
        finally:
            events.append("closed")

    return env


@pytest.mark.parametrize("ending", ["cancel", "hang-up"])
async def test_a_grade_still_running_ends_with_its_session(ending: str) -> None:
    events: list[str] = []
    async with LocalRuntime(stalled(events))(Task(env="stalled", id="stall")) as runtime:
        async with wire(runtime.url) as peer:
            await peer.call("tasks.start", {"id": "stall"})
            await peer.send("tasks.grade", {"answer": "x"})
            if ending == "cancel":
                await peer.call("tasks.cancel")
            else:
                await peer.hang_up()
        closed = list(events)

    assert closed == ["closed"]


def start_of_size(size: int) -> dict[str, Any]:
    """A ``tasks.start`` frame for ``echo`` of exactly ``size`` bytes, without its newline."""
    frame: dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tasks.start",
        "params": {"id": "echo", "args": {"data": ""}},
    }
    frame["params"]["args"]["data"] = "x" * (size - (len(encode(frame)) - 1))
    return frame


async def test_a_request_at_the_frame_limit_round_trips_its_payload() -> None:
    env = Environment("limits")

    @env.template()
    async def echo(data: str):
        yield data
        yield {"score": 1.0, "detail": data}

    frame = start_of_size(FRAME_LIMIT)
    async with LocalRuntime(env)(Task(env="limits", id="x")) as runtime, wire(runtime.url) as peer:
        await peer.write(encode(frame))
        started = await peer.read()
        graded = await peer.call("tasks.grade", {"answer": "done"})

    data = frame["params"]["args"]["data"]
    assert started is not None
    assert (started["result"]["prompt"], graded["result"]["detail"]) == (data, data)


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


@pytest.fixture
async def echo_port() -> AsyncIterator[int]:
    """A daemon on the environment's network that echoes every byte back."""

    async def echo(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while data := await reader.read(65536):
                writer.write(data)
                await writer.drain()
        finally:
            writer.close()

    server = await asyncio.start_server(echo, "127.0.0.1", 0)
    yield server.sockets[0].getsockname()[1]
    server.close()
    await server.wait_closed()


def echo_env(port: int) -> Environment:
    return Environment(
        "echo-env",
        capabilities=[Capability(name="echo", protocol="rfb/3.8", url=f"rfb://127.0.0.1:{port}")],
    )


async def round_trip(host: str | None, port: int | None, payload: bytes) -> bytes:
    reader, writer = await asyncio.open_connection(host, port)
    writer.write(payload)
    await writer.drain()
    data = await reader.readexactly(len(payload))
    writer.close()
    await writer.wait_closed()
    return data


async def test_concurrent_tunnel_streams_do_not_interleave(echo_port: int) -> None:
    payloads = [f"stream-{i}".encode() * 100 for i in range(8)]

    env = echo_env(echo_port)
    async with (
        LocalRuntime(env)(Task(env=env.name, id="scenario")) as runtime,
        connect(runtime) as client,
    ):
        binding = urlsplit(client.binding("echo").url)
        echoed = await asyncio.gather(
            *(round_trip(binding.hostname, binding.port, payload) for payload in payloads)
        )

    assert echoed == payloads


@pytest.mark.parametrize("preface_size", [128, FRAME_LIMIT])
async def test_raw_bytes_sent_with_the_preface_reach_the_daemon(
    echo_port: int, preface_size: int
) -> None:
    preface: dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tunnel.open",
        "params": {"capability": "echo"},
        "padding": "",
    }
    preface["padding"] = "x" * (preface_size - (len(encode(preface)) - 1))
    env = echo_env(echo_port)

    async with (
        LocalRuntime(env)(Task(env=env.name, id="tunnel")) as runtime,
        wire(runtime.url) as stream,
    ):
        await stream.write(encode(preface) + b"raw bytes")
        opened = await stream.read()
        echoed = await asyncio.wait_for(stream.reader.readexactly(9), timeout=30)

    assert opened == {"jsonrpc": "2.0", "id": 1, "result": {"capability": "echo"}}
    assert echoed == b"raw bytes"
