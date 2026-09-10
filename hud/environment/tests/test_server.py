"""The wire grade contract: ``tasks.grade`` frames carry a numeric ``score``.

The server normalizes every grade yield to the canonical frame and fails
loudly on authoring bugs (a grade that is neither a number, a ``.reward``
object, nor a ``{"score": ...}`` dict) instead of silently grading 0.0.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Literal
from urllib.parse import urlsplit

import pytest
from pydantic import BaseModel

from hud.clients import HudProtocolError, connect
from hud.environment import Answer, Environment
from hud.environment.utils import FrameTooLargeError, encode_frame, read_frame, send_frame
from hud.eval import LocalRuntime, Run, Task
from hud.graders import EvaluationResult, SubScore

from .conftest import served


class _Payload(BaseModel):
    text: str


_Mode = Literal["upper", "lower"]


async def test_dict_grade_without_numeric_score_errors_loudly() -> None:
    env = Environment("badgrade")

    @env.template()
    async def reward_keyed():
        yield "go"
        yield {"reward": 1.0}  # wrong key: the wire grade frame is {"score": ...}

    async with served(env) as client:
        with pytest.raises(HudProtocolError, match="score"):
            async with Run(client, "reward_keyed", {}) as run:
                run.trace.content = "x"


async def test_non_numeric_grade_errors_loudly() -> None:
    env = Environment("badgrade")

    @env.template()
    async def stringy():
        yield "go"
        yield "great job"

    async with served(env) as client:
        with pytest.raises(HudProtocolError, match="yield a number"):
            async with Run(client, "stringy", {}) as run:
                run.trace.content = "x"


async def test_score_dict_passes_through_with_extra_keys() -> None:
    env = Environment("richgrade")

    @env.template()
    async def rich():
        yield "go"
        yield {"score": 0.5, "info": {"detail": "partial credit"}}

    async with served(env) as client:
        async with Run(client, "rich", {}) as run:
            run.trace.content = "x"
        assert run.reward == 0.5
        assert run.grade.info == {"detail": "partial credit"}


async def test_evaluation_result_info_reaches_evaluate_step() -> None:
    env = Environment("modelgrade")

    @env.template()
    async def graded():
        yield "go"
        yield EvaluationResult(
            reward=0.75,
            content="nice",
            info={"max_tile": 256},
            subscores=[
                SubScore(
                    name="judge",
                    value=0.75,
                    children=[
                        SubScore(
                            name="criterion",
                            value=1.0,
                            info={"reason": "because"},
                        )
                    ],
                    info={"model": "judge-model"},
                )
            ],
        )

    async with served(env) as client:
        async with Run(client, "graded", {}) as run:
            run.trace.content = "x"
        assert run.reward == 0.75
        assert run.grade.info == {"max_tile": 256}
        assert "info" not in run.evaluation["subscores"][0]
        assert "info" not in run.evaluation["subscores"][0]["children"][0]
        evaluate_step = run.trace.steps[-1]
        assert evaluate_step.task_call is not None
        assert evaluate_step.task_call.phase == "evaluate"
        assert evaluate_step.task_call.result == {
            "score": 0.75,
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
        }


def test_answer_holds_parsed_content_and_raw_string() -> None:
    answer = Answer(content={"final": "42"}, raw='{"final": "42"}')
    assert answer.content == {"final": "42"}
    assert answer.raw == '{"final": "42"}'


async def test_start_coerces_postponed_rich_annotations() -> None:
    env = Environment("coerce")

    @env.template()
    async def typed(mode: _Mode, payload: _Payload, retries: int | None = None):
        if mode == "upper":
            prompt = payload.text.upper()
        elif mode == "lower":
            prompt = payload.text.lower()
        else:
            raise ValueError(f"unexpected mode: {mode!r}")
        if retries is not None:
            prompt += "!" * retries
        yield prompt
        yield 1.0

    assert callable(typed)
    async with served(env) as client:
        async with Run(
            client,
            "typed",
            {
                "mode": '"upper"',
                "payload": '{"text":"hello"}',
                "retries": "3",
            },
        ) as run:
            run.trace.content = "x"
        assert run.prompt == "HELLO!!!"


@pytest.mark.parametrize("extra_bytes", [0, 1])
async def test_task_request_frame_size_boundary(extra_bytes: int) -> None:
    env = Environment("frame-limit")
    received: list[str] = []

    @env.template()
    async def task(data: str):
        received.append(data)
        yield "ready"
        yield 1.0

    frame = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tasks.start",
        "params": {"id": "task", "args": {"data": ""}},
    }
    overhead = len(json.dumps(frame, separators=(",", ":")).encode())
    data = "x" * (16 * 1024 * 1024 - overhead + extra_bytes)
    async with served(env) as client:
        if extra_bytes:
            with pytest.raises(FrameTooLargeError, match="16777217 bytes; limit is 16777216 bytes"):
                await client.start_task("task", {"data": data})
            assert received == []
            assert (await client.list_tasks())[0]["id"] == "task"
            await client.start_task("task", {"data": "small"})
            assert received == ["small"]
            await client.cancel()
        else:
            await client.start_task("task", {"data": data})
            assert received == [data]
            await client.cancel()


@pytest.mark.parametrize("result_kind", ["prompt", "grade", "error"])
async def test_oversized_task_response_returns_protocol_error(result_kind: str) -> None:
    env = Environment("large-response")

    @env.template()
    async def task():
        if result_kind == "error":
            raise ValueError("x" * (16 * 1024 * 1024))
        yield "x" * (16 * 1024 * 1024) if result_kind == "prompt" else "ready"
        yield {"score": 1.0, "detail": "x" * (16 * 1024 * 1024)}

    async with served(env) as client:
        with pytest.raises(HudProtocolError, match=r"response is .*limit is 16777216 bytes"):
            await client.start_task("task")
            await client.grade({"answer": "done"})
        await client.cancel()
        assert (await client.list_tasks())[0]["id"] == "task"


@pytest.mark.parametrize("size", [72400, 1024 * 1024])
async def test_large_task_arguments_prompt_and_grade_roundtrip(size: int) -> None:
    env = Environment("large-task")
    data = "x" * size

    @env.template()
    async def task(criteria: str):
        yield criteria
        yield {"score": 1.0, "detail": criteria}

    async with served(env) as client:
        async with Run(client, "task", {"criteria": data}) as run:
            assert run.prompt == data
            run.trace.content = "done"
        assert run.reward == 1.0
        assert run.evaluation["detail"] == data


async def test_rejected_start_response_tears_down_task_before_disconnect() -> None:
    env = Environment("failed-start")
    released = asyncio.Event()

    @env.template()
    async def task():
        try:
            yield "x" * (16 * 1024 * 1024)
            yield 1.0
        finally:
            released.set()

    async with LocalRuntime(env)(Task(env=env.name, id="task")) as runtime:
        async with connect(runtime) as client:
            with pytest.raises(HudProtocolError, match="limit is 16777216 bytes"):
                await client.start_task("task")
            assert released.is_set()
        async with connect(runtime) as resumed:
            with pytest.raises(HudProtocolError, match="no task in progress"):
                await resumed.grade({"answer": "done"})


@pytest.mark.parametrize("hello_first", [False, True])
async def test_wire_request_over_limit_receives_error_before_disconnect(hello_first: bool) -> None:
    env = Environment("wire-limit")
    async with LocalRuntime(env)(Task(env=env.name, id="unused")) as runtime:
        address = urlsplit(runtime.url)
        reader, writer = await asyncio.open_connection(address.hostname, address.port)
        try:
            if hello_first:
                await send_frame(writer, {"jsonrpc": "2.0", "id": 1, "method": "hello"})
                assert await read_frame(reader) is not None
            writer.write(
                encode_frame(
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "tasks.start",
                        "params": {"id": "unused", "args": {"data": "x" * (16 * 1024 * 1024)}},
                    }
                )
            )
            response = await asyncio.wait_for(read_frame(reader), timeout=5.0)
            assert response is not None
            assert response["id"] is None
            assert "16777216 bytes" in response["error"]["message"]
        finally:
            writer.close()
            with contextlib.suppress(ConnectionError):
                await writer.wait_closed()


async def test_large_request_id_cannot_make_fallback_error_exceed_limit() -> None:
    env = Environment("large-id")
    frame = {"jsonrpc": "2.0", "id": "", "method": "nope"}
    overhead = len(encode_frame(frame)) - 1
    frame["id"] = "x" * (16 * 1024 * 1024 - overhead)
    async with LocalRuntime(env)(Task(env=env.name, id="unused")) as runtime:
        address = urlsplit(runtime.url)
        reader, writer = await asyncio.open_connection(address.hostname, address.port)
        try:
            await send_frame(writer, frame)
            response = await asyncio.wait_for(read_frame(reader), timeout=5.0)
            assert response is not None
            assert response["id"] is None
            assert "limit is 16777216 bytes" in response["error"]["message"]
        finally:
            writer.close()
            await writer.wait_closed()
