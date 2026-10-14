"""The steps a rollout exports carry each provider's turn in the shared step schema.

The platform reads ``hud.step.v1`` span payloads; these rows pin what an agent
turn, a tool round-trip and a nested rollout look like there, per provider:
content, reasoning, citations, refusals, usage, model and training samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest
from dirty_equals import IsStr
from inline_snapshot import snapshot

from hud.agents.base import Agent
from hud.agents.types import (
    SubagentStep,
)
from hud.types import Trace
from tests.agents.support import chat, claude, gemini, openai, run_task, workspace_env
from tests.harness import Turn, call, say, steps

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from hud.eval import Run
    from tests.harness import HudEnv, Models

TIMESTAMP = IsStr(regex=r"\d{4}-\d{2}-\d{2}T.*")


def agent_steps(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The agent, tool and subagent step payloads: timestamps checked, then they, the step
    ids, the provider's raw response and every empty field dropped."""
    found = []
    for payload in payloads:
        if payload["source"] not in {"agent", "tool", "subagent"}:
            continue
        for key in ("started_at", "ended_at"):
            if key in payload:
                assert payload.pop(key) == TIMESTAMP
        del payload["step_id"]
        found.append(
            {key: value for key, value in payload.items() if value not in ([], {}) and key != "raw"}
        )
    return found


@dataclass(frozen=True)
class Row:
    agent: Callable[[], Agent]
    script: list[Turn]
    steps: Any


CLAUDE_CITATIONS = (
    {
        "type": "char_location",
        "cited_text": "the sky is blue",
        "document_index": 0,
        "document_title": "notes",
        "start_char_index": 4,
        "end_char_index": 19,
    },
    {
        "type": "web_search_result_location",
        "cited_text": "blue skies",
        "url": "https://example.com/sky",
        "title": "Sky",
        "encrypted_index": "x",
    },
)

ROWS = {
    "claude": Row(
        agent=claude(),
        script=[
            Turn(
                reasoning="look it up",
                native=({"type": "text", "text": "Blue.", "citations": list(CLAUDE_CITATIONS)},),
            )
        ],
        steps=snapshot(
            [
                {
                    "source": "agent",
                    "content": "Blue.",
                    "reasoning": "look it up",
                    "done": True,
                    "finish_reason": "end_turn",
                    "citations": [
                        {
                            "type": "document_citation",
                            "text": "the sky is blue",
                            "source": "0",
                            "title": "notes",
                            "start_index": 4,
                            "end_index": 19,
                        },
                        {
                            "type": "web_search_result_location",
                            "text": "blue skies",
                            "source": "https://example.com/sky",
                            "title": "Sky",
                        },
                    ],
                    "model": "claude-sonnet-4-6",
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }
            ]
        ),
    ),
    "claude-refusal": Row(
        agent=claude(),
        script=[
            Turn(
                finish="refusal",
                extra={
                    "stop_details": {
                        "type": "refusal",
                        "category": "cyber",
                        "explanation": "This request was blocked under the Usage Policy.",
                    }
                },
            )
        ],
        steps=snapshot(
            [
                {
                    "source": "agent",
                    "content": "",
                    "done": True,
                    "finish_reason": "refusal",
                    "refusal": "This request was blocked under the Usage Policy.",
                    "model": "claude-sonnet-4-6",
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }
            ]
        ),
    ),
    "claude-refusal-category": Row(
        agent=claude(),
        script=[
            Turn(
                finish="refusal",
                extra={
                    "stop_details": {"type": "refusal", "category": "cyber", "explanation": None}
                },
            )
        ],
        steps=snapshot(
            [
                {
                    "source": "agent",
                    "content": "",
                    "done": True,
                    "finish_reason": "refusal",
                    "refusal": "This request was refused (cyber).",
                    "model": "claude-sonnet-4-6",
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }
            ]
        ),
    ),
    "openai": Row(
        agent=openai(),
        script=[
            Turn(
                reasoning="look it up",
                native=(
                    {
                        "type": "message",
                        "id": "msg_cited",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Blue.",
                                "annotations": [
                                    {
                                        "type": "url_citation",
                                        "url": "https://example.com/sky",
                                        "title": "Sky",
                                        "start_index": 0,
                                        "end_index": 5,
                                    },
                                    {
                                        "type": "file_citation",
                                        "file_id": "file-1",
                                        "filename": "notes.txt",
                                        "index": 0,
                                    },
                                ],
                            }
                        ],
                    },
                ),
            )
        ],
        steps=snapshot(
            [
                {
                    "source": "agent",
                    "content": "Blue.",
                    "reasoning": "look it up",
                    "done": True,
                    "citations": [
                        {
                            "type": "url_citation",
                            "text": "Sky",
                            "source": "https://example.com/sky",
                            "title": "Sky",
                            "start_index": 0,
                            "end_index": 5,
                        },
                        {
                            "type": "file_citation",
                            "text": "notes.txt",
                            "source": "file-1",
                            "title": "notes.txt",
                        },
                    ],
                    "model": "gpt-5.6",
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cached_tokens": 0},
                }
            ]
        ),
    ),
    "gemini": Row(
        agent=gemini(),
        script=[
            Turn(
                text="Blue.",
                reasoning="look it up",
                extra={
                    "groundingMetadata": {
                        "groundingChunks": [
                            {"web": {"uri": "https://example.com/sky", "title": "Sky"}},
                            {"web": {"uri": "https://example.com/unused", "title": "Unused"}},
                        ],
                        "groundingSupports": [
                            {
                                "segment": {"text": "Blue.", "startIndex": 0, "endIndex": 5},
                                "groundingChunkIndices": [0],
                            }
                        ],
                    }
                },
            )
        ],
        steps=snapshot(
            [
                {
                    "source": "agent",
                    "content": "Blue.",
                    "reasoning": "look it up",
                    "done": True,
                    "finish_reason": "STOP",
                    "citations": [
                        {
                            "type": "grounding",
                            "text": "Blue.",
                            "source": "https://example.com/sky",
                            "title": "Sky",
                            "start_index": 0,
                            "end_index": 5,
                        },
                        {
                            "type": "grounding",
                            "text": "",
                            "source": "https://example.com/unused",
                            "title": "Unused",
                        },
                    ],
                    "model": "gemini-3.1-pro-preview",
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }
            ]
        ),
    ),
    "claude-tool": Row(
        agent=claude(),
        script=[call("bash", command="echo hi"), say("done")],
        steps=snapshot(
            [
                {
                    "source": "agent",
                    "content": "",
                    "tool_calls": [
                        {
                            "meta": {"citations_enabled": False},
                            "name": "bash",
                            "arguments": {"command": "echo hi"},
                            "id": "toolu_1",
                        }
                    ],
                    "done": False,
                    "finish_reason": "tool_use",
                    "model": "claude-sonnet-4-6",
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
                {
                    "source": "tool",
                    "call": {
                        "meta": {"citations_enabled": False},
                        "name": "bash",
                        "arguments": {"command": "echo hi"},
                        "id": "toolu_1",
                    },
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": """\
$ echo hi
hi

(exit 0)\
""",
                            }
                        ],
                        "isError": False,
                    },
                },
                {
                    "source": "agent",
                    "content": "done",
                    "done": True,
                    "finish_reason": "end_turn",
                    "model": "claude-sonnet-4-6",
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            ]
        ),
    ),
    "chat-training": Row(
        agent=chat(completion_kwargs={"extra_body": {"return_token_ids": True}}),
        script=[
            Turn(
                text="Blue.",
                extra={
                    "prompt_token_ids": [1, 2],
                    "token_ids": [3, 4],
                    "logprobs": {
                        "content": [
                            {"token": "Bl", "logprob": -0.5, "bytes": None, "top_logprobs": []},
                            {"token": "ue.", "logprob": -0.25, "bytes": None, "top_logprobs": []},
                        ]
                    },
                },
            )
        ],
        steps=snapshot(
            [
                {
                    "source": "agent",
                    "content": "Blue.",
                    "done": True,
                    "finish_reason": "stop",
                    "model": "qwen3.6-plus",
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                    "sample": {
                        "prompt_token_ids": [1, 2],
                        "output_token_ids": [3, 4],
                        "output_logprobs": [-0.5, -0.25],
                    },
                }
            ]
        ),
    ),
    "chat": Row(
        agent=chat(),
        script=[Turn(text="Blue.", reasoning="look it up")],
        steps=snapshot(
            [
                {
                    "source": "agent",
                    "content": "Blue.",
                    "reasoning": "look it up",
                    "done": True,
                    "finish_reason": "stop",
                    "model": "qwen3.6-plus",
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }
            ]
        ),
    ),
}


@pytest.mark.parametrize("row", ROWS.values(), ids=ROWS.keys())
async def test_a_provider_turn_exports_as_an_agent_step(
    row: Row, models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    models.script(row.script)

    run = await run_task(workspace_env(tmp_path / "ws"), row.agent())

    assert agent_steps(steps(run.trace_id)) == row.steps


class Delegating(Agent):
    """Records a nested rollout the way a delegating agent reports one."""

    async def __call__(self, run: Run) -> None:
        run.record(SubagentStep(subagent=Trace(content="inner answer", status="completed")))
        run.trace.content = "outer answer"


async def test_a_nested_rollout_exports_as_a_subagent_step(tmp_path: Path) -> None:
    run = await run_task(workspace_env(tmp_path / "ws"), Delegating())

    assert agent_steps(steps(run.trace_id)) == snapshot(
        [
            {
                "source": "subagent",
                "subagent": {
                    "steps": [],
                    "status": "completed",
                    "content": "inner answer",
                    "extra": {},
                },
            }
        ]
    )
