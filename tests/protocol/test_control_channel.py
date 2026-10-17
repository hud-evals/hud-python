"""The control channel, frame by frame: what a peer sends and what the server answers.

One environment carries a template per grading and argument shape. Each scenario
serves it on loopback, speaks raw JSON-RPC to it, and checks the fields of each
reply frame it is about: the handshake, the task catalog, a task's start and
grade, how a session ends, and the frame limit. A ``guarded`` template appends to
a log when it starts and when its generator closes, so a scenario sees whether a
template ran and whether it was torn down. Bad requests are in
``test_error_replies.py``, with every other JSON-RPC server's.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import pytest
from dirty_equals import IsPartialDict
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
from tests.support import FRAME_LIMIT, SESSION_ID, encode, wire

if TYPE_CHECKING:
    from pathlib import Path

Mode = Literal["upper", "lower"]


class Payload(BaseModel):
    text: str


class Point(BaseModel):
    x: int
    y: int


class Attachment(DataFileRef):
    expand: bool = False


class Criterion(BaseModel):
    requirement: str
    weight: float = 1.0


def protocol_env(log: Path) -> Environment:
    env = Environment("protocol", version="1.2.3")

    @env.initialize
    async def publish_browser() -> None:
        env.add_capability(Capability.cdp(url="ws://127.0.0.1:9/devtools/browser"))

    @env.template(description="Grades 1.0 when the answer is 42.")
    async def plain():
        answer = yield "What is six times seven?"
        yield 1.0 if answer == "42" else 0.0

    @env.template()
    async def model():
        yield "go"
        yield EvaluationResult(
            reward=0.75,
            content="nice",
            info={"max_tile": 256},
            subscores=[
                SubScore(
                    name="judge",
                    value=0.75,
                    info={"model": "judge-model"},
                    children=[SubScore(name="criterion", value=1.0, info={"reason": "because"})],
                )
            ],
        )

    @env.template()
    async def bad_dict():
        yield "go"
        yield {"reward": 1.0}

    @env.template()
    async def no_grade():
        yield "go"

    @env.template()
    async def typed(mode: Mode, payload: Payload, retries: int | None = None):
        text = payload.text.upper() if mode == "upper" else payload.text.lower()
        yield text + "!" * (retries or 0)
        yield 1.0

    @env.template(input=Point, returns=Point)
    async def structured():
        answer = yield "Name a point."
        content = answer.content
        yield {
            "score": 1.0 if isinstance(content, Point) else 0.0,
            "content": content.model_dump() if isinstance(content, Point) else content,
            "raw": answer.raw,
        }

    @env.template()
    async def files(
        prompt: PromptArg,
        attachment: DataFileArg[Attachment],
        fixtures: DataFilesArg[Attachment],
        criteria: GradingArg[Criterion],
    ):
        yield (
            f"{prompt} {type(attachment).__name__}:{attachment.path} "
            f"{[type(f).__name__ for f in fixtures]} "
            f"{type(criteria[0]).__name__}:{criteria[0].requirement}x{criteria[0].weight}"
        )
        yield 1.0

    @env.template()
    async def defaults(difficulty: int, suite: str = "coding"):
        yield f"{suite} at {difficulty}"
        yield 1.0

    @env.template()
    async def flexible(n: int, **rest: str):
        yield f"{n} {sorted(rest.items())}"
        yield 1.0

    @env.template()
    async def loose(anything):
        yield repr(anything)
        yield 1.0

    @env.template()
    async def guarded(name: str):
        with log.open("a") as out:
            out.write(f"start {name}\n")
        try:
            yield f"holding {name}"
            yield 1.0
        finally:
            with log.open("a") as out:
                out.write(f"closed {name}\n")

    return env


def lines(log: Path) -> list[str]:
    return log.read_text().splitlines() if log.exists() else []


async def converse(env: Environment, frames: list[tuple[str, dict[str, Any] | None]]) -> list[Any]:
    """Send ``frames`` on one connection; the replies, ``None`` once the server hangs up."""
    replies: list[Any] = []
    async with (
        LocalRuntime(env)(Task(env=env.name, id="protocol")) as runtime,
        wire(runtime.url) as connection,
    ):
        for method, params in frames:
            await connection.send(method, params)
            replies.append(await connection.read())
    return replies


def answer(reply: dict[str, Any]) -> Any:
    """A reply frame's result, or its error code."""
    assert reply["jsonrpc"] == "2.0"
    return reply["result"] if "result" in reply else reply["error"]["code"]


async def test_tasks_list_publishes_each_templates_argument_and_io_contract(tmp_path: Path) -> None:
    env = protocol_env(tmp_path / "log")
    (reply,) = await converse(env, [("tasks.list", None)])
    listed = {task["id"]: task for task in reply["result"]["tasks"]}
    files = listed["files"]["args"]["properties"]
    defaults = listed["defaults"]["args"]

    assert {name: field["x-hud-hint"] for name, field in files.items()} == {
        "prompt": "prompt",
        "attachment": "data-file",
        "fixtures": "data-files",
        "criteria": "grading",
    }
    assert (defaults["required"], defaults["properties"]["suite"]["default"]) == (
        ["difficulty"],
        "coding",
    )
    assert {name for name, task in listed.items() if task["args"]["additionalProperties"]} == {
        "flexible"
    }
    assert {name: sorted(task.keys() & {"input", "returns"}) for name, task in listed.items()} == {
        name: ["input", "returns"] if name == "structured" else [] for name in listed
    }


async def test_hello_names_the_session_and_publishes_hook_capabilities(tmp_path: Path) -> None:
    (reply,) = await converse(protocol_env(tmp_path / "log"), [("hello", {})])
    hello = answer(reply)

    assert hello["session_id"] == SESSION_ID
    assert hello["env"] == {"name": "protocol", "version": "1.2.3"}
    assert [(binding["name"], binding["url"]) for binding in hello["bindings"]] == [
        ("browser", "ws://127.0.0.1:9/devtools/browser")
    ]


BAD_STATE, FAILED = -32600, -32000
TEMPLATES: dict[str, tuple[str, dict[str, Any], str, str, Any]] = {
    "number-right": ("plain", {}, "42", "What is six times seven?", {"score": 1.0}),
    "evaluation-result-keeps-content-info-and-nested-subscores": (
        "model",
        {},
        "x",
        "go",
        IsPartialDict(
            score=0.75,
            content="nice",
            info={"max_tile": 256},
            subscores=[
                IsPartialDict(
                    name="judge",
                    value=0.75,
                    info={"model": "judge-model"},
                    children=[IsPartialDict(name="criterion", info={"reason": "because"})],
                )
            ],
        ),
    ),
    "dict-without-score-fails-loudly": (
        "bad_dict",
        {},
        "x",
        "go",
        FAILED,
    ),
    "no-grade-yield-grades-zero": ("no_grade", {}, "x", "go", {"score": 0.0}),
    "json-string-args-coerce-to-annotations": (
        "typed",
        {"mode": '"upper"', "payload": '{"text":"hello"}', "retries": "3"},
        "x",
        "HELLO!!!",
        {"score": 1.0},
    ),
    "native-args-validate-against-annotations": (
        "typed",
        {"mode": "lower", "payload": {"text": "HeLLo"}},
        "x",
        "hello",
        {"score": 1.0},
    ),
    "returns-parses": (
        "structured",
        {},
        '{"x": 1, "y": 2}',
        "Name a point.",
        {"score": 1.0, "content": {"x": 1, "y": 2}, "raw": '{"x": 1, "y": 2}'},
    ),
    "returns-keeps-raw": (
        "structured",
        {},
        "about (1, 2)",
        "Name a point.",
        {"score": 0.0, "content": "about (1, 2)", "raw": "about (1, 2)"},
    ),
    "data-file-and-grading-args-arrive-as-models": (
        "files",
        {
            "prompt": "Review",
            "attachment": {"file_id": "file-1", "path": "brief.pdf"},
            "fixtures": [{"file_id": "file-2"}],
            "criteria": [{"requirement": "Answer it", "weight": 2.0}],
        },
        "x",
        "Review Attachment:brief.pdf ['Attachment'] Criterion:Answer itx2.0",
        {"score": 1.0},
    ),
    "unannotated": (
        "loose",
        {"anything": [1, {"a": None}]},
        "x",
        "[1, {'a': None}]",
        {"score": 1.0},
    ),
}


@pytest.mark.parametrize(
    ("template", "args", "reply", "prompt", "grade"), TEMPLATES.values(), ids=TEMPLATES.keys()
)
async def test_a_template_starts_and_grades_over_the_wire(
    tmp_path: Path, template: str, args: dict[str, Any], reply: str, prompt: str, grade: Any
) -> None:
    replies = await converse(
        protocol_env(tmp_path / "log"),
        [("tasks.start", {"id": template, "args": args}), ("tasks.grade", {"answer": reply})],
    )

    assert [answer(frame) for frame in replies] == [{"prompt": prompt}, grade]


FIRST = ["start first", "closed first"]
ENDINGS: dict[
    str, tuple[list[tuple[str, dict[str, Any] | None]], list[Any], list[str], list[str]]
] = {
    "cancel": ([("tasks.cancel", None)], [{"cancelled": True}], FIRST, FIRST),
    "bye": ([("bye", None)], [{"goodbye": True}], FIRST, FIRST),
    "grade": ([("tasks.grade", {"answer": "x"})], [{"score": 1.0}], FIRST, FIRST),
    "restart": (
        [("tasks.start", {"id": "guarded", "args": {"name": "second"}})],
        [{"prompt": "holding second"}],
        [*FIRST, "start second"],
        [*FIRST, "start second", "closed second"],
    ),
    "disconnect-parks": ([], [], ["start first"], FIRST),
}


@pytest.mark.parametrize(
    ("ending", "replies", "while_served", "after"), ENDINGS.values(), ids=ENDINGS.keys()
)
async def test_ending_a_session_closes_its_template(
    tmp_path: Path,
    ending: list[tuple[str, dict[str, Any] | None]],
    replies: list[Any],
    while_served: list[str],
    after: list[str],
) -> None:
    log = tmp_path / "log"
    env = protocol_env(log)

    async with LocalRuntime(env)(Task(env=env.name, id="protocol")) as runtime:
        async with wire(runtime.url) as connection:
            await connection.call("tasks.start", {"id": "guarded", "args": {"name": "first"}})
            sent = [await connection.call(method, params) for method, params in ending]
        served_log = lines(log)

    assert [answer(frame) for frame in sent] == replies
    assert (served_log, lines(log)) == (while_served, after)


def big_args_start(size: int) -> dict[str, Any]:
    """A ``tasks.start`` frame of exactly ``size`` bytes, excluding the newline."""
    frame: dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tasks.start",
        "params": {"id": "echo", "args": {"data": ""}},
    }
    frame["params"]["args"]["data"] = "x" * (size - (len(encode(frame)) - 1))
    return frame


def oversized_env(kind: str, finished: list[str]) -> Environment:
    env = Environment("oversized")
    huge = "x" * FRAME_LIMIT

    @env.template()
    async def echo(data: str):
        yield data
        yield {"score": 1.0, "detail": data}

    @env.template()
    async def bloated():
        try:
            if kind == "error":
                raise ValueError(huge)
            yield huge
        finally:
            finished.append(kind)

    return env


@pytest.mark.parametrize(
    ("size", "accepted"),
    [(FRAME_LIMIT, True), (FRAME_LIMIT + 1, False)],
)
async def test_a_request_up_to_the_frame_limit_round_trips(size: int, accepted: bool) -> None:
    env = oversized_env("prompt", [])
    frame = big_args_start(size)
    data = frame["params"]["args"]["data"]

    async with (
        LocalRuntime(env)(Task(env=env.name, id="echo")) as runtime,
        wire(runtime.url) as connection,
    ):
        await connection.write(encode(frame))
        started = await connection.read()
        graded = await connection.call("tasks.grade", {"answer": "done"}) if accepted else None

    if accepted:
        assert started is not None and started["result"]["prompt"] == data
        assert graded is not None and graded["result"]["detail"] == data
    else:
        assert started is not None
        assert (started["id"], started["error"]["code"]) == (None, FAILED)


@pytest.mark.parametrize("kind", ["prompt", "error"])
async def test_an_oversized_reply_is_an_error_frame_and_releases_the_template(kind: str) -> None:
    finished: list[str] = []
    env = oversized_env(kind, finished)

    async with LocalRuntime(env)(Task(env=env.name, id="bloated")) as runtime:
        async with wire(runtime.url) as connection:
            started = await connection.call("tasks.start", {"id": "bloated"})
            listed = await connection.call("tasks.list")
        async with wire(runtime.url) as later:
            regrade = await later.call("tasks.grade", {"answer": "done"})

    assert started["error"]["code"] == FAILED
    assert finished == [kind]
    assert [task["id"] for task in listed["result"]["tasks"]] == ["echo", "bloated"]
    assert regrade["error"]["code"] == BAD_STATE


@pytest.mark.parametrize(
    ("field", "serving"),
    [
        pytest.param("padding", False, id="oversized-first-frame-hangs-up"),
        pytest.param("id", True, id="reply-too-large-for-its-id-keeps-serving"),
    ],
)
async def test_an_oversized_frame_gets_a_bounded_error_without_an_id(
    field: str, serving: bool
) -> None:
    frame: dict[str, Any] = {"jsonrpc": "2.0", "id": "", "method": "nope"}
    frame[field] = "x" * (FRAME_LIMIT - (len(encode(frame)) - 1) if field == "id" else FRAME_LIMIT)
    env = Environment("wire-limit")

    async with (
        LocalRuntime(env)(Task(env=env.name, id="unused")) as runtime,
        wire(runtime.url) as connection,
    ):
        await connection.write(encode(frame))
        reply = await connection.read()
        if serving:
            await connection.send("tasks.list")
        after = await connection.read()

    assert reply is not None
    assert (reply["id"], reply["error"]["code"]) == (None, FAILED)
    assert (after is not None and after["result"] == {"tasks": []}) is serving
