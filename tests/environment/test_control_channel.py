"""The control channel as a peer speaks it: the handshake, a task's start and grade, how a
session's template is torn down, and a request at the frame limit."""

from __future__ import annotations

from typing import Any, Literal

import pytest
from pydantic import BaseModel

from hud.capabilities import Capability
from hud.environment import (
    Environment,
)
from hud.eval import LocalRuntime, Task
from hud.graders import EvaluationResult, SubScore
from tests.environment.conftest import FRAME_LIMIT, SESSION_ID, encode, wire

INVALID, NOT_FOUND, BAD_STATE, FAILED = -32602, -32601, -32600, -32000


async def test_hello_names_the_session_the_env_and_what_its_hooks_published() -> None:
    env = Environment("protocol", version="1.2.3")

    @env.initialize
    async def publish_browser() -> None:
        env.add_capability(Capability.cdp(url="ws://127.0.0.1:9/devtools/browser"))

    async with (
        LocalRuntime(env)(Task(env="protocol", id="x")) as runtime,
        wire(runtime.url) as peer,
    ):
        hello = (await peer.call("hello", {}))["result"]

    assert (hello["session_id"], hello["env"]) == (
        SESSION_ID,
        {"name": "protocol", "version": "1.2.3"},
    )
    assert [(b["name"], b["url"]) for b in hello["bindings"]] == [
        ("browser", "ws://127.0.0.1:9/devtools/browser")
    ]


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


async def test_an_evaluation_result_reaches_the_peer_with_its_details_and_breakdown() -> None:
    env = Environment("graded")

    @env.template()
    async def judged():
        yield "go"
        yield EvaluationResult(
            reward=0.75,
            content="nice",
            info={"max_tile": 256},
            subscores=[
                SubScore(
                    name="judge",
                    value=0.75,
                    children=[SubScore(name="criterion", value=1.0, info={"reason": "because"})],
                )
            ],
        )

    async with LocalRuntime(env)(Task(env="graded", id="x")) as runtime, wire(runtime.url) as peer:
        await peer.call("tasks.start", {"id": "judged"})
        graded = (await peer.call("tasks.grade", {"answer": "x"}))["result"]

    assert (graded["score"], graded["content"], graded["info"]) == (0.75, "nice", {"max_tile": 256})
    (judge,) = graded["subscores"]
    assert (judge["name"], judge["children"][0]["info"]) == ("judge", {"reason": "because"})


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
