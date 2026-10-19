"""What only ``OpenAIChatAgent`` puts on the wire: its completion options on every Chat
Completions request (training samples, checkpoints), and how it replays a cut-off call."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from tests.agents.families import CHAT, Wire, carried
from tests.support import Turn, call, say

if TYPE_CHECKING:
    from pathlib import Path

    from tests.support import HudEnv, ModelRequest, Models


def options(requests: list[ModelRequest]) -> list[dict[str, Any]]:
    """Each request without its conversation and tool list."""
    return [
        {key: value for key, value in request.body.items() if key not in {"messages", "tools"}}
        for request in requests
    ]


BASE = {"model": "qwen3.6-plus", "stream": False}
TOKEN_IDS = {"extra_body": {"return_token_ids": True}}
SAMPLED = BASE | {"logprobs": True, "return_token_ids": True}
CALL = call("bash", command="true").tool_calls
CUT_OFF = Turn(
    native=(
        {
            "id": "call_cut",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"command": "ec'},
        },
    )
)

WIRE = {
    "token-ids-continue-the-kv-cache": Wire(
        {"completion_kwargs": TOKEN_IDS},
        options,
        [SAMPLED, SAMPLED | {"prompt_token_ids": [1, 2, 3], "continuation_from": 2}],
        turns=[
            Turn(tool_calls=CALL, extra={"prompt_token_ids": [1, 2], "token_ids": [3]}),
            Turn(text="done", extra={"prompt_token_ids": [1, 2, 3, 4], "token_ids": [5]}),
        ],
    ),
    "multimodal-prompt-chunks-reset-the-continuation": Wire(
        {"completion_kwargs": TOKEN_IDS},
        options,
        [SAMPLED, SAMPLED],
        turns=[
            Turn(
                tool_calls=CALL,
                extra={"prompt_chunks": [{"type": "text", "text": "hi"}], "token_ids": [3]},
            ),
            say("done"),
        ],
    ),
    "checkpoint-routes-in-the-body": Wire(
        {"checkpoint": "ckpt-7"}, options, [BASE | {"checkpoint": "ckpt-7"}]
    ),
    "reserved-completion-kwargs-are-dropped": Wire(
        {
            "completion_kwargs": {
                "model": "other",
                "stream": True,
                "tools": [],
                "messages": [],
                "max_tokens": 50,
            }
        },
        options,
        [BASE | {"max_tokens": 50}],
    ),
    "a-cut-off-call-is-replayed-with-empty-arguments": Wire(
        {},
        lambda requests: requests[-1].body["messages"][1]["tool_calls"][0]["function"]["arguments"],
        "{}",
        turns=[CUT_OFF, say("done")],
    ),
}


@pytest.mark.parametrize("wire", WIRE.values(), ids=WIRE.keys())
async def test_chat_carries_its_configuration_onto_every_request(
    wire: Wire, models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")

    assert await carried(CHAT, wire, models, tmp_path / "ws") == wire.expected
