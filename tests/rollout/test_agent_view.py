"""What the agent sees of its run: the prompt as turns and as text, the episode bindings the
task published, and the connections it was given, only while it runs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from hud.agents.base import Agent
from hud.capabilities import Connection
from hud.eval import LocalRuntime, Run, Task, rollout
from hud.telemetry.context import get_current_trace_id
from tests.support import lab

if TYPE_CHECKING:
    from pathlib import Path


def add(a: int = 2, b: int = 3) -> Task:
    return Task(env="lab", id="add", args={"a": a, "b": b})


def chat_row(messages: list[Any]) -> Task:
    return Task(env="lab", id="chat", args={"messages": messages})


PROMPTS = {
    "text is one user turn": (add(), [("user", "add 2 3")], "add 2 3"),
    "no prompt is one empty user turn": (Task(env="lab", id="silent"), [("user", "")], ""),
    "chat turns keep their roles and system becomes user": (
        chat_row(
            [
                {"role": "system", "content": "be brief"},
                {"role": "assistant", "content": {"type": "text", "text": "hi"}},
                {"role": "user", "content": "add 2 3"},
            ]
        ),
        [("user", "be brief"), ("assistant", "hi"), ("user", "add 2 3")],
        "be brief\n\nhi\n\nadd 2 3",
    ),
    "non-text turns are dropped from the text": (
        chat_row(
            [
                {"role": "user", "content": "first"},
                {"role": "user", "content": {"type": "image", "data": "aW1n", "mimeType": "x"}},
                "second",
            ]
        ),
        [("user", "first"), ("user", None), ("user", "second")],
        "first\n\nsecond",
    ),
    "a turn of several blocks is one message per block": (
        chat_row(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this?"},
                        {"type": "image", "data": "aW1n", "mimeType": "image/png"},
                    ],
                }
            ]
        ),
        [("user", "what is this?"), ("user", None)],
        "what is this?",
    ),
}


@pytest.mark.parametrize(("row", "messages", "text"), PROMPTS.values(), ids=PROMPTS.keys())
async def test_the_agent_sees_the_prompt_as_turns_and_as_text(
    row: Task, messages: list[tuple[str, str | None]], text: str
) -> None:
    seen: list[tuple[list[tuple[str, str | None]], str]] = []

    class Reader(Agent):
        async def __call__(self, run: Run) -> None:
            turns: list[tuple[str, str | None]] = [
                (message.role, getattr(message.content, "text", None))
                for message in run.prompt_messages
            ]
            seen.append((turns, run.prompt_text))

    await rollout(row, Reader(), runtime=LocalRuntime(lab()))

    assert seen == [(messages, text)]


async def test_episode_bindings_from_the_start_frame_reach_the_agent() -> None:
    seen: list[tuple[dict[str, Any], str | None]] = []

    class Inspector(Agent):
        async def __call__(self, run: Run) -> None:
            seen.append((run.bindings, get_current_trace_id()))

    run = await rollout(
        Task(env="lab", id="claim", args={"published": {"robot": {"token": "slot-2"}}}),
        Inspector(),
        runtime=LocalRuntime(lab()),
    )

    assert seen == [({"robot": {"token": "slot-2"}}, run.trace_id)]


@pytest.mark.e2e
@pytest.mark.sandbox
async def test_connections_reach_the_agent_only_while_it_runs(tmp_path: Path) -> None:
    env = lab()
    env.workspace(tmp_path / "ws")
    search = Connection(
        name="search",
        capability="shell",
        url="https://search.example/v1",
        headers={"Authorization": "Bearer secret"},
    )
    seen: list[list[str]] = []

    class Inspector(Agent):
        async def __call__(self, run: Run) -> None:
            seen.append(sorted(run.connections))
            run.trace.content = "5"

    run = await rollout(add(), Inspector(), runtime=LocalRuntime(env), connections=[search])

    assert seen == [["search"]]
    assert (run.reward, run.connections) == (1.0, {})
