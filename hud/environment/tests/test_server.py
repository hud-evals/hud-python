"""The wire grade contract: ``tasks.grade`` frames carry a numeric ``score``.

The server normalizes every grade yield to the canonical frame and fails
loudly on authoring bugs (a grade that is neither a number, a ``.reward``
object, nor a ``{"score": ...}`` dict) instead of silently grading 0.0.
"""

from __future__ import annotations

import asyncio
import contextlib
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


@pytest.mark.parametrize("frame_size", [72483, 1024 * 1024, 16777216, 16777217])
async def test_task_frame_limit_and_retry(frame_size: int) -> None:
    env = Environment("frame-limit")

    @env.template()
    async def task(data: str):
        yield data
        yield {"score": 1.0, "detail": data}

    frame = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tasks.start",
        "params": {"id": "task", "args": {"data": ""}},
    }
    data = "x" * (frame_size - (len(encode_frame(frame)) - 1))
    async with served(env) as client:
        if frame_size > 16777216:
            with pytest.raises(FrameTooLargeError, match="16777217 bytes; limit is 16777216 bytes"):
                await client.start_task("task", {"data": data})
            data = "small retry"
        assert (await client.start_task("task", {"data": data}))["prompt"] == data
        assert (await client.grade({"answer": "done"}))["detail"] == data


@pytest.mark.parametrize("result_kind", ["prompt", "grade", "error"])
async def test_oversized_response_releases_task_and_reports_error(result_kind: str) -> None:
    env = Environment("large-response")
    released = asyncio.Event()

    @env.template()
    async def task():
        try:
            if result_kind == "error":
                raise ValueError("x" * (16 * 1024 * 1024))
            yield "x" * (16 * 1024 * 1024) if result_kind == "prompt" else "ready"
            yield {"score": 1.0, "detail": "x" * (16 * 1024 * 1024)}
        finally:
            released.set()

    async with LocalRuntime(env)(Task(env=env.name, id="task")) as runtime:
        async with connect(runtime) as client:
            with pytest.raises(HudProtocolError, match=r"response is .*limit is 16777216 bytes"):
                await client.start_task("task")
                await client.grade({"answer": "done"})
            assert released.is_set()
            assert (await client.list_tasks())[0]["id"] == "task"
        async with connect(runtime) as resumed:
            with pytest.raises(HudProtocolError, match="no task in progress"):
                await resumed.grade({"answer": "done"})


@pytest.mark.parametrize("hello_first,large_id", [(False, False), (True, False), (False, True)])
async def test_wire_size_errors_are_bounded(hello_first: bool, large_id: bool) -> None:
    frame = {"jsonrpc": "2.0", "id": "", "method": "nope"}
    field = "id" if large_id else "padding"
    frame[field] = "x" * (16777216 - (len(encode_frame(frame)) - 1) if large_id else 16777216)
    env = Environment("wire-limit")
    async with LocalRuntime(env)(Task(env=env.name, id="unused")) as runtime:
        address = urlsplit(runtime.url)
        reader, writer = await asyncio.open_connection(address.hostname, address.port)
        try:
            if hello_first:
                await send_frame(writer, {"jsonrpc": "2.0", "id": 1, "method": "hello"})
                assert await read_frame(reader) is not None
            writer.write(encode_frame(frame))
            response = await asyncio.wait_for(read_frame(reader), timeout=5.0)
            assert response is not None and response["id"] is None
            assert "16777216 bytes" in response["error"]["message"]
        finally:
            writer.close()
            with contextlib.suppress(ConnectionError):
                await writer.wait_closed()
