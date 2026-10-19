"""What a provider's reply becomes on the step the platform reads (``hud.step.v1``).

A real model will not cite, refuse or return training samples on demand, so a
scripted provider sends each in that provider's own shape.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from hud.agents import ClaudeAgent, GeminiAgent, OpenAIAgent, OpenAIChatAgent
from hud.agents.types import ClaudeConfig, GeminiConfig, OpenAIChatConfig, OpenAIConfig
from tests.fixtures.environments import run_task, workspace_env
from tests.support import Turn, steps

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from hud.agents.base import Agent
    from tests.support import HudEnv, Models

WEB = "https://example.com/sky"


async def agent_step(env_root: Path, agent: Agent) -> dict[str, Any]:
    run = await run_task(workspace_env(env_root), agent)
    (step,) = [step for step in steps(run.trace_id) if step["source"] == "agent"]
    return step


@pytest.mark.parametrize(
    ("agent", "turn", "cited"),
    [
        pytest.param(
            lambda: ClaudeAgent(ClaudeConfig(model="claude-sonnet-4-6")),
            Turn(
                reasoning="look it up",
                native=(
                    {
                        "type": "text",
                        "text": "Blue.",
                        "citations": [
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
                                "url": WEB,
                                "title": "Sky",
                                "encrypted_index": "x",
                            },
                        ],
                    },
                ),
            ),
            [
                ("document_citation", "0", "the sky is blue"),
                ("web_search_result_location", WEB, "blue skies"),
            ],
            id="claude",
        ),
        pytest.param(
            lambda: OpenAIAgent(OpenAIConfig(model="gpt-5.6")),
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
                                        "url": WEB,
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
            ),
            [("url_citation", WEB, "Sky"), ("file_citation", "file-1", "notes.txt")],
            id="openai",
        ),
        pytest.param(
            lambda: GeminiAgent(GeminiConfig(model="gemini-3.1-pro-preview")),
            Turn(
                text="Blue.",
                reasoning="look it up",
                extra={
                    "groundingMetadata": {
                        "groundingChunks": [
                            {"web": {"uri": WEB, "title": "Sky"}},
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
            ),
            [("grounding", WEB, "Blue."), ("grounding", "https://example.com/unused", "")],
            id="gemini",
        ),
    ],
)
async def test_a_providers_citations_and_reasoning_land_on_the_agent_step(
    agent: Callable[[], Agent],
    turn: Turn,
    cited: list[tuple[str, str, str]],
    models: Models,
    hud_env: HudEnv,
    tmp_path: Path,
) -> None:
    hud_env.set(HUD_API_KEY="k")
    models.script([turn])

    step = await agent_step(tmp_path / "ws", agent())

    assert (step["content"], step["reasoning"]) == ("Blue.", "look it up")
    assert [(c["type"], c["source"], c["text"]) for c in step["citations"]] == cited


@pytest.mark.parametrize(
    ("explanation", "refusal"),
    [
        (
            "This request was blocked under the Usage Policy.",
            "This request was blocked under the Usage Policy.",
        ),
        (None, "This request was refused (cyber)."),
    ],
    ids=["with-explanation", "category-only"],
)
async def test_a_claude_refusal_lands_on_the_agent_step(
    explanation: str | None, refusal: str, models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    details = {"type": "refusal", "category": "cyber", "explanation": explanation}
    models.script([Turn(finish="refusal", extra={"stop_details": details})])

    step = await agent_step(tmp_path / "ws", ClaudeAgent(ClaudeConfig(model="claude-sonnet-4-6")))

    assert (step["finish_reason"], step["refusal"]) == ("refusal", refusal)


async def test_a_chat_completion_with_token_ids_exports_its_training_sample(
    models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    logprobs = [
        {"token": "Bl", "logprob": -0.5, "bytes": None, "top_logprobs": []},
        {"token": "ue.", "logprob": -0.25, "bytes": None, "top_logprobs": []},
    ]
    models.script(
        [
            Turn(
                text="Blue.",
                extra={
                    "prompt_token_ids": [1, 2],
                    "token_ids": [3, 4],
                    "logprobs": {"content": logprobs},
                },
            )
        ]
    )
    config = OpenAIChatConfig(
        model="qwen3.6-plus", completion_kwargs={"extra_body": {"return_token_ids": True}}
    )

    step = await agent_step(tmp_path / "ws", OpenAIChatAgent(config))

    assert step["sample"] == {
        "prompt_token_ids": [1, 2],
        "output_token_ids": [3, 4],
        "output_logprobs": [-0.5, -0.25],
    }
