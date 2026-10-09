"""The control channel, frame by frame: what a peer sends and what the server answers.

One environment carries a template per grading and argument shape. Each scenario
serves it on loopback, speaks raw JSON-RPC to it, and compares every reply frame
with a golden copy (session ids normalized). A ``guarded`` template appends to a
log when it starts and when its generator closes, so a scenario sees whether a
template ran and whether it was torn down.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, cast

import pytest
from dirty_equals import IsStr
from inline_snapshot import snapshot
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

from .conftest import FRAME_LIMIT, SESSION_ID, encode, wire

if TYPE_CHECKING:
    from collections.abc import Callable
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
    async def rich():
        yield "go"
        yield {"score": 0.5, "info": {"detail": "partial credit"}}

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
    async def stringy():
        yield "go"
        yield "great job"

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

    @env.template()
    async def lookup():
        settings: dict[str, str] = {}
        yield settings["missing"]

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


async def test_tasks_list_publishes_each_templates_argument_and_io_contract(tmp_path: Path) -> None:
    env = protocol_env(tmp_path / "log")
    (reply,) = await converse(env, [("tasks.list", None)])
    listed = {task["id"]: task for task in reply["result"]["tasks"]}
    empty = {"properties": {}, "type": "object", "additionalProperties": False}

    assert sorted(listed) == sorted(env.tasks)
    assert listed["plain"]["description"] == "Grades 1.0 when the answer is 42."
    assert {
        name: task
        for name, task in listed.items()
        if task["args"] != empty or "input" in task or "returns" in task
    } == snapshot(
        {
            "typed": {
                "id": "typed",
                "description": "",
                "args": {
                    "$defs": {
                        "Payload": {
                            "properties": {"text": {"title": "Text", "type": "string"}},
                            "required": ["text"],
                            "title": "Payload",
                            "type": "object",
                        }
                    },
                    "properties": {
                        "mode": {"enum": ["upper", "lower"], "title": "Mode", "type": "string"},
                        "payload": {"$ref": "#/$defs/Payload"},
                        "retries": {
                            "anyOf": [{"type": "integer"}, {"type": "null"}],
                            "default": None,
                            "title": "Retries",
                        },
                    },
                    "required": ["mode", "payload"],
                    "type": "object",
                    "additionalProperties": False,
                },
            },
            "structured": {
                "id": "structured",
                "description": "",
                "args": {"properties": {}, "type": "object", "additionalProperties": False},
                "input": {
                    "properties": {
                        "x": {"title": "X", "type": "integer"},
                        "y": {"title": "Y", "type": "integer"},
                    },
                    "required": ["x", "y"],
                    "title": "Point",
                    "type": "object",
                },
                "returns": {
                    "properties": {
                        "x": {"title": "X", "type": "integer"},
                        "y": {"title": "Y", "type": "integer"},
                    },
                    "required": ["x", "y"],
                    "title": "Point",
                    "type": "object",
                },
            },
            "files": {
                "id": "files",
                "description": "",
                "args": {
                    "$defs": {
                        "Attachment": {
                            "additionalProperties": False,
                            "properties": {
                                "file_id": {
                                    "description": "HUD data-file id",
                                    "title": "File Id",
                                    "type": "string",
                                },
                                "path": {
                                    "anyOf": [{"type": "string"}, {"type": "null"}],
                                    "default": None,
                                    "description": "Environment-owned destination path",
                                    "title": "Path",
                                },
                                "expand": {"default": False, "title": "Expand", "type": "boolean"},
                            },
                            "required": ["file_id"],
                            "title": "Attachment",
                            "type": "object",
                        },
                        "Criterion": {
                            "properties": {
                                "requirement": {"title": "Requirement", "type": "string"},
                                "weight": {"default": 1.0, "title": "Weight", "type": "number"},
                            },
                            "required": ["requirement"],
                            "title": "Criterion",
                            "type": "object",
                        },
                    },
                    "properties": {
                        "prompt": {"title": "Prompt", "type": "string", "x-hud-hint": "prompt"},
                        "attachment": {"$ref": "#/$defs/Attachment", "x-hud-hint": "data-file"},
                        "fixtures": {
                            "items": {"$ref": "#/$defs/Attachment"},
                            "title": "Fixtures",
                            "type": "array",
                            "x-hud-hint": "data-files",
                        },
                        "criteria": {
                            "items": {"$ref": "#/$defs/Criterion"},
                            "title": "Criteria",
                            "type": "array",
                            "x-hud-hint": "grading",
                        },
                    },
                    "required": ["prompt", "attachment", "fixtures", "criteria"],
                    "type": "object",
                    "additionalProperties": False,
                },
            },
            "defaults": {
                "id": "defaults",
                "description": "",
                "args": {
                    "properties": {
                        "difficulty": {"title": "Difficulty", "type": "integer"},
                        "suite": {"default": "coding", "title": "Suite", "type": "string"},
                    },
                    "required": ["difficulty"],
                    "type": "object",
                    "additionalProperties": False,
                },
            },
            "flexible": {
                "id": "flexible",
                "description": "",
                "args": {
                    "properties": {"n": {"title": "N", "type": "integer"}},
                    "required": ["n"],
                    "type": "object",
                    "additionalProperties": True,
                },
            },
            "loose": {
                "id": "loose",
                "description": "",
                "args": {
                    "properties": {"anything": {"title": "Anything"}},
                    "required": ["anything"],
                    "type": "object",
                    "additionalProperties": False,
                },
            },
            "guarded": {
                "id": "guarded",
                "description": "",
                "args": {
                    "properties": {"name": {"title": "Name", "type": "string"}},
                    "required": ["name"],
                    "type": "object",
                    "additionalProperties": False,
                },
            },
        }
    )


async def test_hello_names_the_session_and_publishes_hook_capabilities(tmp_path: Path) -> None:
    replies = await converse(protocol_env(tmp_path / "log"), [("hello", {}), ("bye", None)])

    assert replies == snapshot(
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "session_id": SESSION_ID,
                    "env": {"name": "protocol", "version": "1.2.3"},
                    "bindings": [
                        {
                            "name": "browser",
                            "protocol": "cdp/1.3",
                            "url": "ws://127.0.0.1:9/devtools/browser",
                            "params": {},
                        }
                    ],
                },
            },
            {"jsonrpc": "2.0", "id": 2, "result": {"goodbye": True}},
        ]
    )


@pytest.mark.parametrize(
    ("template", "args", "answer", "expected"),
    [
        pytest.param(
            "plain",
            {},
            "42",
            snapshot(
                [
                    {"jsonrpc": "2.0", "id": 1, "result": {"prompt": "What is six times seven?"}},
                    {"jsonrpc": "2.0", "id": 2, "result": {"score": 1.0}},
                ]
            ),
            id="number-right",
        ),
        pytest.param(
            "plain",
            {},
            "41",
            snapshot(
                [
                    {"jsonrpc": "2.0", "id": 1, "result": {"prompt": "What is six times seven?"}},
                    {"jsonrpc": "2.0", "id": 2, "result": {"score": 0.0}},
                ]
            ),
            id="number-wrong",
        ),
        pytest.param(
            "rich",
            {},
            "x",
            snapshot(
                [
                    {"jsonrpc": "2.0", "id": 1, "result": {"prompt": "go"}},
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "result": {"score": 0.5, "info": {"detail": "partial credit"}},
                    },
                ]
            ),
            id="score-dict-keeps-extras",
        ),
        pytest.param(
            "model",
            {},
            "x",
            snapshot(
                [
                    {"jsonrpc": "2.0", "id": 1, "result": {"prompt": "go"}},
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "result": {
                            "done": True,
                            "content": "nice",
                            "info": {"max_tile": 256},
                            "isError": False,
                            "subscores": [
                                {
                                    "name": "judge",
                                    "weight": 1.0,
                                    "value": 0.75,
                                    "children": [
                                        {
                                            "name": "criterion",
                                            "weight": 1.0,
                                            "value": 1.0,
                                            "children": None,
                                            "info": {"reason": "because"},
                                        }
                                    ],
                                    "info": {"model": "judge-model"},
                                }
                            ],
                            "score": 0.75,
                        },
                    },
                ]
            ),
            id="evaluation-result-whole-frame",
        ),
        pytest.param(
            "bad_dict",
            {},
            "x",
            snapshot(
                [
                    {"jsonrpc": "2.0", "id": 1, "result": {"prompt": "go"}},
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "error": {
                            "code": -32000,
                            "message": (
                                "task 'bad_dict' graded with a dict missing a numeric 'score' "
                                "(keys: ['reward'])"
                            ),
                        },
                    },
                ]
            ),
            id="dict-without-score-fails-loudly",
        ),
        pytest.param(
            "stringy",
            {},
            "x",
            snapshot(
                [
                    {"jsonrpc": "2.0", "id": 1, "result": {"prompt": "go"}},
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "error": {
                            "code": -32000,
                            "message": (
                                "task graded with str: yield a number, an object with a numeric "
                                ".reward, or a dict containing a numeric 'score'"
                            ),
                        },
                    },
                ]
            ),
            id="string-grade-fails-loudly",
        ),
        pytest.param(
            "no_grade",
            {},
            "x",
            snapshot(
                [
                    {"jsonrpc": "2.0", "id": 1, "result": {"prompt": "go"}},
                    {"jsonrpc": "2.0", "id": 2, "result": {"score": 0.0}},
                ]
            ),
            id="no-grade-yield-grades-zero",
        ),
        pytest.param(
            "typed",
            {"mode": '"upper"', "payload": '{"text":"hello"}', "retries": "3"},
            "x",
            snapshot(
                [
                    {"jsonrpc": "2.0", "id": 1, "result": {"prompt": "HELLO!!!"}},
                    {"jsonrpc": "2.0", "id": 2, "result": {"score": 1.0}},
                ]
            ),
            id="json-string-args-coerce-to-annotations",
        ),
        pytest.param(
            "typed",
            {"mode": "lower", "payload": {"text": "HeLLo"}},
            "x",
            snapshot(
                [
                    {"jsonrpc": "2.0", "id": 1, "result": {"prompt": "hello"}},
                    {"jsonrpc": "2.0", "id": 2, "result": {"score": 1.0}},
                ]
            ),
            id="native-args-validate-against-annotations",
        ),
        pytest.param(
            "structured",
            {},
            '{"x": 1, "y": 2}',
            snapshot(
                [
                    {"jsonrpc": "2.0", "id": 1, "result": {"prompt": "Name a point."}},
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "result": {
                            "score": 1.0,
                            "content": {"x": 1, "y": 2},
                            "raw": '{"x": 1, "y": 2}',
                        },
                    },
                ]
            ),
            id="returns-parses",
        ),
        pytest.param(
            "structured",
            {},
            "about (1, 2)",
            snapshot(
                [
                    {"jsonrpc": "2.0", "id": 1, "result": {"prompt": "Name a point."}},
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "result": {"score": 0.0, "content": "about (1, 2)", "raw": "about (1, 2)"},
                    },
                ]
            ),
            id="returns-keeps-raw",
        ),
        pytest.param(
            "files",
            {
                "prompt": "Review",
                "attachment": {"file_id": "file-1", "path": "brief.pdf"},
                "fixtures": [{"file_id": "file-2"}],
                "criteria": [{"requirement": "Answer it", "weight": 2.0}],
            },
            "x",
            snapshot(
                [
                    {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "result": {
                            "prompt": (
                                "Review Attachment:brief.pdf ['Attachment'] Criterion:Answer itx2.0"
                            )
                        },
                    },
                    {"jsonrpc": "2.0", "id": 2, "result": {"score": 1.0}},
                ]
            ),
            id="data-file-and-grading-args-arrive-as-models",
        ),
        pytest.param(
            "defaults",
            {"difficulty": 3},
            "x",
            snapshot(
                [
                    {"jsonrpc": "2.0", "id": 1, "result": {"prompt": "coding at 3"}},
                    {"jsonrpc": "2.0", "id": 2, "result": {"score": 1.0}},
                ]
            ),
            id="defaults-fill-in",
        ),
        pytest.param(
            "flexible",
            {"n": 1, "b": "2", "a": "1"},
            "x",
            snapshot(
                [
                    {"jsonrpc": "2.0", "id": 1, "result": {"prompt": "1 [('a', '1'), ('b', '2')]"}},
                    {"jsonrpc": "2.0", "id": 2, "result": {"score": 1.0}},
                ]
            ),
            id="var-kwargs",
        ),
        pytest.param(
            "loose",
            {"anything": [1, {"a": None}]},
            "x",
            snapshot(
                [
                    {"jsonrpc": "2.0", "id": 1, "result": {"prompt": "[1, {'a': None}]"}},
                    {"jsonrpc": "2.0", "id": 2, "result": {"score": 1.0}},
                ]
            ),
            id="unannotated",
        ),
    ],
)
async def test_a_template_starts_and_grades_over_the_wire(
    tmp_path: Path, template: str, args: dict[str, Any], answer: str, expected: Any
) -> None:
    replies = await converse(
        protocol_env(tmp_path / "log"),
        [("tasks.start", {"id": template, "args": args}), ("tasks.grade", {"answer": answer})],
    )

    assert replies == expected


@pytest.mark.parametrize(
    ("frames", "expected"),
    [
        pytest.param(
            [("tasks.start", {"id": "missing"})],
            snapshot(
                (
                    [
                        {
                            "jsonrpc": "2.0",
                            "id": 1,
                            "error": {"code": -32602, "message": "unknown task: 'missing'"},
                        }
                    ],
                    [],
                )
            ),
            id="start-unknown-template",
        ),
        pytest.param(
            [("tasks.start", {"id": 7})],
            snapshot(
                (
                    [
                        {
                            "jsonrpc": "2.0",
                            "id": 1,
                            "error": {
                                "code": -32602,
                                "message": "tasks.start: 'id' must be a string",
                            },
                        }
                    ],
                    [],
                )
            ),
            id="start-non-string-id",
        ),
        pytest.param(
            [("tasks.start", {"id": "guarded", "args": ["x"]})],
            snapshot(
                (
                    [
                        {
                            "jsonrpc": "2.0",
                            "id": 1,
                            "error": {
                                "code": -32602,
                                "message": "tasks.start: 'args' must be an object",
                            },
                        }
                    ],
                    [],
                )
            ),
            id="start-list-args",
        ),
        pytest.param(
            [("tasks.start", {"id": "guarded", "args": []})],
            snapshot(
                (
                    [
                        {
                            "jsonrpc": "2.0",
                            "id": 1,
                            "error": {
                                "code": -32602,
                                "message": "tasks.start: 'args' must be an object",
                            },
                        }
                    ],
                    [],
                )
            ),
            id="start-empty-list-args",
        ),
        pytest.param(
            [("tasks.start", {"id": "guarded", "args": {"nmae": "x"}})],
            snapshot(
                (
                    [
                        {
                            "jsonrpc": "2.0",
                            "id": 1,
                            "error": {
                                "code": -32000,
                                "message": (
                                    "task 'guarded': bad args ['nmae']: missing a required "
                                    "argument: 'name'"
                                ),
                            },
                        }
                    ],
                    [],
                )
            ),
            id="start-signature-mismatch",
        ),
        pytest.param(
            [("tasks.start", {"id": "lookup"})],
            snapshot(
                (
                    [
                        {
                            "jsonrpc": "2.0",
                            "id": 1,
                            "error": {"code": -32000, "message": "'missing'"},
                        }
                    ],
                    [],
                )
            ),
            id="start-template-raises-keyerror",
        ),
        pytest.param(
            [("tasks.grade", {"answer": "x"})],
            snapshot(
                (
                    [
                        {
                            "jsonrpc": "2.0",
                            "id": 1,
                            "error": {"code": -32600, "message": "no task in progress"},
                        }
                    ],
                    [],
                )
            ),
            id="grade-without-start",
        ),
        pytest.param(
            [("tasks.stop", {})],
            snapshot(
                (
                    [
                        {
                            "jsonrpc": "2.0",
                            "id": 1,
                            "error": {"code": -32601, "message": "method not found: tasks.stop"},
                        }
                    ],
                    [],
                )
            ),
            id="unknown-method",
        ),
        pytest.param(
            [("tasks.cancel", None), ("tunnel.open", {"capability": "browser"})],
            snapshot(
                (
                    [
                        {
                            "jsonrpc": "2.0",
                            "id": 1,
                            "result": {"cancelled": True},
                        },
                        {
                            "jsonrpc": "2.0",
                            "id": 2,
                            "error": {"code": -32601, "message": "method not found: tunnel.open"},
                        },
                    ],
                    [],
                )
            ),
            id="tunnel-open-mid-session",
        ),
        pytest.param(
            [("hello", {"session_id": 5})],
            snapshot(
                (
                    [
                        {
                            "jsonrpc": "2.0",
                            "id": 1,
                            "error": {
                                "code": -32602,
                                "message": "hello: 'session_id' must be a string",
                            },
                        }
                    ],
                    [],
                )
            ),
            id="hello-non-string-session",
        ),
        pytest.param(
            [("hello", {"session_id": "sess-00000000"})],
            snapshot(
                (
                    [
                        {
                            "jsonrpc": "2.0",
                            "id": 1,
                            "error": {
                                "code": -32600,
                                "message": "unknown session: 'sess-00000000'",
                            },
                        }
                    ],
                    [],
                )
            ),
            id="hello-unknown-session",
        ),
        pytest.param(
            [("hello", {"workspace_routes": "ssh"})],
            snapshot(
                (
                    [
                        {
                            "jsonrpc": "2.0",
                            "id": 1,
                            "error": {
                                "code": -32602,
                                "message": "hello: 'workspace_routes' must be a list",
                            },
                        }
                    ],
                    [],
                )
            ),
            id="hello-routes-not-a-list",
        ),
        pytest.param(
            [("hello", {"workspace_routes": [{"capability": "ssh", "host": "h", "port": "443"}]})],
            snapshot(
                (
                    [
                        {
                            "jsonrpc": "2.0",
                            "id": 1,
                            "error": {
                                "code": -32602,
                                "message": "hello: workspace route port must be an integer",
                            },
                        }
                    ],
                    [],
                )
            ),
            id="hello-route-port-not-an-int",
        ),
        pytest.param(
            [("hello", {"connections": {"name": "x"}})],
            snapshot(
                (
                    [
                        {
                            "jsonrpc": "2.0",
                            "id": 1,
                            "error": {
                                "code": -32602,
                                "message": "hello: 'connections' must be a list",
                            },
                        }
                    ],
                    [],
                )
            ),
            id="hello-connections-object",
        ),
        pytest.param(
            [("hello", {"connections": [{"name": "inference"}]})],
            snapshot(
                (
                    [
                        {
                            "jsonrpc": "2.0",
                            "id": 1,
                            "error": {
                                "code": -32602,
                                "message": (
                                    "hello: connection name, capability, and url must be strings"
                                ),
                            },
                        }
                    ],
                    [],
                )
            ),
            id="hello-connection-missing-fields",
        ),
        pytest.param(
            [
                (
                    "hello",
                    {
                        "connections": [
                            {
                                "name": "inference",
                                "capability": "ssh",
                                "url": "https://inference.example",
                                "headers": {"Authorization": "Bearer t"},
                            }
                        ]
                    },
                )
            ],
            snapshot(
                (
                    [
                        {
                            "jsonrpc": "2.0",
                            "id": 1,
                            "error": {
                                "code": -32000,
                                "message": "workspace capability 'ssh' does not exist",
                            },
                        }
                    ],
                    [],
                )
            ),
            id="hello-connection-without-a-workspace",
        ),
        pytest.param(
            [("hello", {"workspace_routes": [{"capability": "ssh", "host": "h", "port": 443}]})],
            snapshot(
                (
                    [
                        {
                            "jsonrpc": "2.0",
                            "id": 1,
                            "error": {
                                "code": -32000,
                                "message": "workspace capability 'ssh' does not exist",
                            },
                        }
                    ],
                    [],
                )
            ),
            id="hello-route-without-a-workspace",
        ),
    ],
)
async def test_a_bad_request_gets_an_error_reply_and_the_session_keeps_serving(
    tmp_path: Path, frames: list[tuple[str, dict[str, Any] | None]], expected: Any
) -> None:
    log = tmp_path / "log"

    replies = await converse(protocol_env(log), [*frames, ("tasks.list", None)])

    *errors, listed = replies
    assert (errors, lines(log)) == expected
    assert "tasks" in listed["result"]


@pytest.mark.parametrize(
    ("ending", "expected"),
    [
        pytest.param(
            [("tasks.cancel", None)],
            snapshot(
                (
                    [{"jsonrpc": "2.0", "id": 2, "result": {"cancelled": True}}],
                    ["start first", "closed first"],
                    ["start first", "closed first"],
                )
            ),
            id="cancel",
        ),
        pytest.param(
            [("bye", None)],
            snapshot(
                (
                    [{"jsonrpc": "2.0", "id": 2, "result": {"goodbye": True}}],
                    ["start first", "closed first"],
                    ["start first", "closed first"],
                )
            ),
            id="bye",
        ),
        pytest.param(
            [("tasks.grade", {"answer": "x"})],
            snapshot(
                (
                    [{"jsonrpc": "2.0", "id": 2, "result": {"score": 1.0}}],
                    ["start first", "closed first"],
                    ["start first", "closed first"],
                )
            ),
            id="grade",
        ),
        pytest.param(
            [("tasks.start", {"id": "guarded", "args": {"name": "second"}})],
            snapshot(
                (
                    [{"jsonrpc": "2.0", "id": 2, "result": {"prompt": "holding second"}}],
                    ["start first", "closed first", "start second"],
                    ["start first", "closed first", "start second", "closed second"],
                )
            ),
            id="restart",
        ),
        pytest.param(
            [],
            snapshot(([], ["start first"], ["start first", "closed first"])),
            id="disconnect-parks",
        ),
    ],
)
async def test_ending_a_session_closes_its_template(
    tmp_path: Path, ending: list[tuple[str, dict[str, Any] | None]], expected: Any
) -> None:
    log = tmp_path / "log"
    env = protocol_env(log)

    async with LocalRuntime(env)(Task(env=env.name, id="protocol")) as runtime:
        async with wire(runtime.url) as connection:
            await connection.call("tasks.start", {"id": "guarded", "args": {"name": "first"}})
            replies = [await connection.call(method, params) for method, params in ending]
        while_served = lines(log)

    assert (replies, while_served, lines(log)) == expected


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
            yield huge if kind == "prompt" else "ready"
            yield {"score": 1.0, "detail": huge}
        finally:
            finished.append(kind)

    return env


@pytest.mark.parametrize(
    ("size", "accepted"),
    [(72483, True), (1024 * 1024, True), (FRAME_LIMIT, True), (FRAME_LIMIT + 1, False)],
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
        assert started == {
            "jsonrpc": "2.0",
            "id": None,
            "error": {
                "code": -32000,
                "message": IsStr(regex=rf"Control-channel frame .* {FRAME_LIMIT} bytes.*"),
            },
        }


@pytest.mark.parametrize("kind", ["prompt", "grade", "error"])
async def test_an_oversized_reply_is_an_error_frame_and_releases_the_template(kind: str) -> None:
    finished: list[str] = []
    env = oversized_env(kind, finished)

    async with LocalRuntime(env)(Task(env=env.name, id="bloated")) as runtime:
        async with wire(runtime.url) as connection:
            started = await connection.call("tasks.start", {"id": "bloated"})
            graded = None
            if "result" in started:
                graded = await connection.call("tasks.grade", {"answer": "done"})
            listed = await connection.call("tasks.list")
        async with wire(runtime.url) as later:
            regrade = await later.call("tasks.grade", {"answer": "done"})

    error = graded if graded is not None else started
    assert error["error"] == {
        "code": -32000,
        "message": IsStr(
            regex=rf"Control-channel response is \d+ bytes; limit is {FRAME_LIMIT} bytes .*"
        ),
    }
    assert finished == [kind]
    assert [task["id"] for task in listed["result"]["tasks"]] == ["echo", "bloated"]
    assert regrade["error"]["message"] == "no task in progress"


@pytest.mark.parametrize(
    ("preface", "field", "serving"),
    [
        pytest.param(False, "padding", False, id="oversized-first-frame-hangs-up"),
        pytest.param(True, "padding", False, id="oversized-frame-mid-session-hangs-up"),
        pytest.param(False, "id", True, id="reply-too-large-for-its-id-keeps-serving"),
    ],
)
async def test_an_oversized_frame_gets_a_bounded_error_without_an_id(
    preface: bool, field: str, serving: bool
) -> None:
    frame: dict[str, Any] = {"jsonrpc": "2.0", "id": "", "method": "nope"}
    frame[field] = "x" * (FRAME_LIMIT - (len(encode(frame)) - 1) if field == "id" else FRAME_LIMIT)
    env = Environment("wire-limit")

    async with (
        LocalRuntime(env)(Task(env=env.name, id="unused")) as runtime,
        wire(runtime.url) as connection,
    ):
        if preface:
            await connection.call("hello", {})
        await connection.write(encode(frame))
        reply = await connection.read()
        if serving:
            await connection.send("tasks.list")
        after = await connection.read()

    assert reply == {
        "jsonrpc": "2.0",
        "id": None,
        "error": {
            "code": -32000,
            "message": IsStr(regex=rf"Control-channel .* {FRAME_LIMIT} bytes.*"),
        },
    }
    assert (after is not None and after["result"] == {"tasks": []}) is serving


def attach_template_to_a_plain_function(env: Environment) -> None:
    def not_a_generator() -> str:
        return "x"

    env.template()(cast("Any", not_a_generator))


def register_a_template_twice(env: Environment) -> None:
    @env.template(id="twice")
    async def first():
        yield "a"

    @env.template(id="twice")
    async def second():
        yield "b"


@pytest.mark.parametrize(
    ("declare", "error", "message"),
    [
        pytest.param(
            attach_template_to_a_plain_function, TypeError, "must be an async generator function"
        ),
        pytest.param(register_a_template_twice, ValueError, "'twice' already registered"),
        pytest.param(
            lambda env: [env.workspace("a"), env.workspace("b")],
            ValueError,
            "workspace capability 'shell' is already attached",
        ),
        pytest.param(
            lambda env: env.capability("missing"), KeyError, "unknown capability: 'missing'"
        ),
        pytest.param(
            lambda env: env.add_capability(cast("Any", {"name": "x"})),
            TypeError,
            "expected Capability",
        ),
        pytest.param(
            lambda env: env.add_capability(
                Capability(name="db", protocol="tcp", url="", params={})
            ),
            ValueError,
            "capability 'db' has no url",
        ),
    ],
)
def test_an_authoring_mistake_raises_at_definition(
    declare: Callable[[Environment], Any], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        declare(Environment("authoring"))


async def test_publishing_a_capability_after_initialize_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    env = Environment("late")
    await env.start()

    with caplog.at_level(logging.WARNING, logger="hud.environment"):
        env.add_capability(Capability.cdp(url="ws://127.0.0.1:9/late"))
    await env.stop()

    assert "add_capability('browser') called after @env.initialize hooks" in caplog.text
