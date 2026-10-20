"""The control channel as a peer speaks it: the handshake, the task catalog, a task's start
and grade, how a session's template is torn down, bad requests, and the frame limit."""

from __future__ import annotations

from typing import Any, Literal

import pytest
from pydantic import BaseModel

from hud.capabilities import Capability
from hud.environment import (
    DataFileArg,
    DataFileRef,
    DataFilesArg,
    Environment,
    GradingArg,
    PromptArg,
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


class Criterion(BaseModel):
    requirement: str


async def test_the_catalog_publishes_each_templates_arguments_and_io_types() -> None:
    env = Environment("catalog")

    @env.template(description="Adds two numbers.")
    async def add(a: int, suite: str = "math"):
        yield "go"
        yield 1.0

    @env.template()
    async def flexible(n: int, **rest: str):
        yield "go"
        yield 1.0

    @env.template(input=Point, returns=Point)
    async def structured():
        yield "go"
        yield 1.0

    @env.template()
    async def files(
        prompt: PromptArg,
        attachment: DataFileArg[DataFileRef],
        fixtures: DataFilesArg[DataFileRef],
        criteria: GradingArg[Criterion],
    ):
        yield "go"
        yield 1.0

    async with LocalRuntime(env)(Task(env="catalog", id="x")) as runtime, wire(runtime.url) as peer:
        listed = {t["id"]: t for t in (await peer.call("tasks.list"))["result"]["tasks"]}

    add_args = listed["add"]["args"]
    assert (listed["add"]["description"], add_args["required"]) == ("Adds two numbers.", ["a"])
    assert add_args["properties"]["suite"]["default"] == "math"
    assert [name for name, t in listed.items() if t["args"]["additionalProperties"]] == ["flexible"]
    assert [name for name, t in listed.items() if {"input", "returns"} <= t.keys()] == [
        "structured"
    ]
    assert {k: v["x-hud-hint"] for k, v in listed["files"]["args"]["properties"].items()} == {
        "prompt": "prompt",
        "attachment": "data-file",
        "fixtures": "data-files",
        "criteria": "grading",
    }


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


async def test_data_file_and_grading_arguments_arrive_as_models() -> None:
    env = Environment("files")

    @env.template()
    async def review(attachment: DataFileArg[DataFileRef], criteria: GradingArg[Criterion]):
        yield f"{type(attachment).__name__}:{attachment.path} {type(criteria[0]).__name__}"
        yield 1.0

    args = {
        "attachment": {"file_id": "file-1", "path": "brief.pdf"},
        "criteria": [{"requirement": "Answer it"}],
    }
    async with LocalRuntime(env)(Task(env="files", id="x")) as runtime, wire(runtime.url) as peer:
        started = await peer.call("tasks.start", {"id": "review", "args": args})

    assert started["result"] == {"prompt": "DataFileRef:brief.pdf Criterion"}


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


async def test_a_template_that_ends_without_grading_grades_zero() -> None:
    env = Environment("ungraded")

    @env.template()
    async def quits():
        yield "go"

    async with (
        LocalRuntime(env)(Task(env="ungraded", id="x")) as runtime,
        wire(runtime.url) as peer,
    ):
        await peer.call("tasks.start", {"id": "quits"})
        graded = await peer.call("tasks.grade", {"answer": "x"})

    assert graded["result"] == {"score": 0.0}


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


async def test_a_dropped_connections_task_closes_its_template_when_the_server_stops() -> None:
    events = []
    env = Environment("held")

    @env.template()
    async def hold():
        try:
            yield "holding"
            yield 1.0
        finally:
            events.append("closed")

    async with LocalRuntime(env)(Task(env="held", id="x")) as runtime, wire(runtime.url) as peer:
        await peer.call("tasks.start", {"id": "hold"})

    assert events == ["closed"]


ROUTE = {"capability": "ssh", "host": "h", "port": "443"}
CONNECTION = {
    "name": "inference",
    "capability": "ssh",
    "url": "https://inference.example",
    "headers": {"Authorization": "Bearer t"},
}


@pytest.mark.parametrize(
    ("method", "params", "code"),
    [
        ("tasks.start", {"id": "missing"}, INVALID),
        ("tasks.start", {"id": "record", "args": ["x"]}, INVALID),
        ("tasks.start", {"id": 5}, INVALID),
        ("tasks.start", {"id": "record", "args": {"nmae": "x"}}, FAILED),
        ("tasks.start", {"id": "lookup"}, FAILED),
        ("tasks.grade", {"answer": "x"}, BAD_STATE),
        ("tasks.stop", {}, NOT_FOUND),
        ("hello", {"session_id": "sess-00000000"}, BAD_STATE),
        ("hello", {"session_id": 7}, INVALID),
        ("hello", {"workspace_routes": [ROUTE]}, INVALID),
        ("hello", {"workspace_routes": "all"}, INVALID),
        ("hello", {"connections": [CONNECTION]}, FAILED),
        ("hello", {"connections": "all"}, INVALID),
    ],
    ids=[
        "an-unknown-template",
        "arguments-that-are-not-an-object",
        "a-template-id-that-is-not-a-string",
        "arguments-the-template-does-not-take",
        "a-template-that-raises-keyerror-is-not-an-unknown-task",
        "a-grade-before-a-start",
        "an-unknown-method",
        "resuming-an-unknown-session",
        "a-session-id-that-is-not-a-string",
        "a-route-whose-port-is-not-a-number",
        "routes-that-are-not-a-list",
        "a-connection-without-a-workspace-to-bind-it",
        "connections-that-are-not-a-list",
    ],
)
async def test_a_bad_request_gets_an_error_reply_runs_nothing_and_the_session_goes_on(
    method: str, params: dict[str, Any], code: int
) -> None:
    started = []
    env = Environment("errors", capabilities=[Capability.cdp(url="ws://127.0.0.1:9/devtools")])

    @env.template()
    async def record(name: str):
        started.append(name)
        yield "go"
        yield 1.0

    @env.template()
    async def lookup():
        settings: dict[str, str] = {}
        yield settings["missing"]

    async with LocalRuntime(env)(Task(env="errors", id="x")) as runtime, wire(runtime.url) as peer:
        refused = await peer.call(method, params)
        listed = await peer.call("tasks.list")

    assert refused["error"]["code"] == code
    assert ([t["id"] for t in listed["result"]["tasks"]], started) == (["record", "lookup"], [])


async def test_a_tunnel_opened_once_a_session_has_begun_is_an_unknown_method() -> None:
    env = Environment("errors", capabilities=[Capability.cdp(url="ws://127.0.0.1:9/devtools")])

    async with LocalRuntime(env)(Task(env="errors", id="x")) as runtime, wire(runtime.url) as peer:
        await peer.call("hello", {})
        refused = await peer.call("tunnel.open", {"capability": "browser"})
        listed = await peer.call("tasks.list")

    assert (refused["error"]["code"], listed["result"]) == (NOT_FOUND, {"tasks": []})


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


async def test_a_request_over_the_frame_limit_is_refused_without_an_id_and_hung_up() -> None:
    async with (
        LocalRuntime(Environment("limits"))(Task(env="limits", id="x")) as runtime,
        wire(runtime.url) as peer,
    ):
        await peer.write(encode(start_of_size(FRAME_LIMIT + 1)))
        refused = await peer.read()
        after = await peer.read()

    assert refused is not None
    assert (refused["id"], refused["error"]["code"], after) == (None, FAILED, None)


async def test_a_reply_too_large_for_its_own_id_is_an_error_and_the_session_goes_on() -> None:
    frame: dict[str, Any] = {"jsonrpc": "2.0", "id": "", "method": "nope"}
    frame["id"] = "x" * (FRAME_LIMIT - (len(encode(frame)) - 1))

    async with (
        LocalRuntime(Environment("limits"))(Task(env="limits", id="x")) as runtime,
        wire(runtime.url) as peer,
    ):
        await peer.write(encode(frame))
        refused = await peer.read()
        listed = await peer.call("tasks.list")

    assert refused is not None
    assert (refused["id"], refused["error"]["code"]) == (None, FAILED)
    assert listed["result"] == {"tasks": []}


@pytest.mark.parametrize("too_large", ["prompt", "error"])
async def test_a_prompt_or_error_over_the_frame_limit_is_refused_and_its_template_released(
    too_large: str,
) -> None:
    released = []
    env = Environment("bloated")

    @env.template()
    async def huge_prompt():
        try:
            yield "x" * FRAME_LIMIT
            yield 1.0
        finally:
            released.append("prompt")

    @env.template()
    async def huge_error():
        try:
            raise ValueError("x" * FRAME_LIMIT)
            yield "never"
        finally:
            released.append("error")

    async with LocalRuntime(env)(Task(env="bloated", id="x")) as runtime:
        async with wire(runtime.url) as peer:
            refused = await peer.call("tasks.start", {"id": f"huge_{too_large}"})
        async with wire(runtime.url) as later:
            regraded = await later.call("tasks.grade", {"answer": "done"})

    assert (refused["error"]["code"], released, regraded["error"]["code"]) == (
        FAILED,
        [too_large],
        BAD_STATE,
    )
