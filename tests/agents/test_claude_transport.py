# ruff: noqa: E501 -- snapshots quote product messages verbatim
"""How ClaudeAgent recovers from what the Messages stream throws at it.

Rows script the Anthropic stream behind the fake gateway to fail transiently,
fail for good, cut out, or emit tool input that is not JSON, and pin how many
requests the agent made, what it sent on a retry, and how the run ended. A
Bedrock inference profile goes through Bedrock's own invoke endpoint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from inline_snapshot import snapshot

from hud.agents import ClaudeAgent
from hud.agents.types import ClaudeConfig
from tests.agents.support import run_task, wire, workspace_env
from tests.harness import Turn, interrupted, say, steps, stream_error

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from tests.harness import FakeServices, HudEnv, ModelRequest, Models

NO_WAIT = {"retry-after-ms": "0"}


def sequence(*turns: Turn) -> Callable[[ModelRequest], Turn]:
    remaining = iter(turns)
    return lambda _request: next(remaining)


def bad_json(fragment: str) -> Turn:
    """A tool call whose streamed input is not JSON."""
    return Turn(
        native=({"type": "tool_use", "id": "toolu_bad", "name": "bash", "input": fragment},)
    )


STREAMS = {
    "transient-error-then-answer": (
        [stream_error("upstream_error", headers=NO_WAIT), say("recovered")],
        snapshot(
            {
                "status": "completed",
                "content": "recovered",
                "error": None,
                "requests": 2,
                "last_messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Do the task.",
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    }
                ],
                "steps": ["task", "user", "agent", "task"],
            }
        ),
    ),
    "two-transient-errors-then-answer": (
        [
            stream_error("timeout_error", headers=NO_WAIT),
            stream_error("overloaded_error", headers=NO_WAIT),
            say("recovered"),
        ],
        snapshot(
            {
                "status": "completed",
                "content": "recovered",
                "error": None,
                "requests": 3,
                "last_messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Do the task.",
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    }
                ],
                "steps": ["task", "user", "agent", "task"],
            }
        ),
    ),
    "transient-errors-exhaust-the-retries": (
        [stream_error("gateway_timeout", headers=NO_WAIT)] * 3,
        snapshot(
            {
                "status": "error",
                "content": None,
                "error": "{'type': 'error', 'error': {'type': 'gateway_timeout', 'message': 'scripted gateway_timeout'}}",
                "requests": 3,
                "last_messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Do the task.",
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    }
                ],
                "steps": ["task", "user", "system", "task"],
            }
        ),
    ),
    "a-request-error-is-not-retried": (
        [stream_error("invalid_request_error", headers=NO_WAIT)],
        snapshot(
            {
                "status": "error",
                "content": None,
                "error": "{'type': 'error', 'error': {'type': 'invalid_request_error', 'message': 'scripted invalid_request_error'}}",
                "requests": 1,
                "last_messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Do the task.",
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    }
                ],
                "steps": ["task", "user", "system", "task"],
            }
        ),
    ),
    "an-interrupted-stream-is-retried": (
        [interrupted(), say("recovered")],
        snapshot(
            {
                "status": "completed",
                "content": "recovered",
                "error": None,
                "requests": 2,
                "last_messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Do the task.",
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    }
                ],
                "steps": ["task", "user", "agent", "task"],
            }
        ),
    ),
    "invalid-tool-json-is-retried-silently": (
        [bad_json('{"command": ls}'), say("recovered")],
        snapshot(
            {
                "status": "completed",
                "content": "recovered",
                "error": None,
                "requests": 2,
                "last_messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Do the task.",
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    }
                ],
                "steps": ["task", "user", "agent", "task"],
            }
        ),
    ),
    "invalid-tool-json-twice-is-pointed-out": (
        [bad_json('{"command": ls}'), bad_json('{"command": ls}'), say("recovered")],
        snapshot(
            {
                "status": "completed",
                "content": "recovered",
                "error": None,
                "requests": 3,
                "last_messages": [
                    {"role": "user", "content": [{"type": "text", "text": "Do the task."}]},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """\
Your previous tool-call arguments were invalid JSON. Retry the same tool call with valid JSON arguments.
Malformed payload (wrapped): {"INVALID_JSON": "{\\"command\\": ls}"}\
""",
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    },
                ],
                "steps": ["task", "user", "agent", "task"],
            }
        ),
    ),
    "invalid-tool-json-three-times-fails-the-run": (
        [bad_json('{"command": ls}')] * 3,
        snapshot(
            {
                "status": "error",
                "content": None,
                "error": 'Unable to parse tool parameter JSON from model. Please retry your request or adjust your prompt. Error: expected value at line 1 column 13. JSON: {"command": ls}',
                "requests": 3,
                "last_messages": [
                    {"role": "user", "content": [{"type": "text", "text": "Do the task."}]},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """\
Your previous tool-call arguments were invalid JSON. Retry the same tool call with valid JSON arguments.
Malformed payload (wrapped): {"INVALID_JSON": "{\\"command\\": ls}"}\
""",
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    },
                ],
                "steps": ["task", "user", "system", "task"],
            }
        ),
    ),
}


@pytest.mark.parametrize(("turns", "expected"), STREAMS.values(), ids=STREAMS.keys())
async def test_claude_retries_what_the_stream_lets_it_retry(
    turns: list[Turn], expected: Any, models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    models.respond(sequence(*turns))

    run = await run_task(
        workspace_env(tmp_path / "ws"), ClaudeAgent(ClaudeConfig(model="claude-sonnet-4-6"))
    )

    requests = models.requests()
    observed = {
        "status": run.trace.status,
        "content": run.trace.content,
        "error": run.trace.error,
        "requests": len(requests),
        "last_messages": wire(requests[-1].body["messages"]),
        "steps": [step["source"] for step in steps(run.trace_id)],
    }
    assert observed == expected


async def test_a_bedrock_inference_profile_is_invoked_on_bedrock_without_streaming(
    services: FakeServices, hud_env: HudEnv, tmp_path: Path
) -> None:
    pytest.importorskip("botocore", reason="needs the bedrock extra")
    profile = "arn:aws:bedrock:us-east-1:123456789012:inference-profile/claude"
    hud_env.set(
        AWS_ACCESS_KEY_ID="AKIA",
        AWS_SECRET_ACCESS_KEY="secret",
        AWS_REGION="us-east-1",
        ANTHROPIC_BEDROCK_BASE_URL=services.url("gateway"),
    )
    services.route(
        "gateway",
        "POST",
        "/model/{model:path}",
        json={
            "id": "msg_bedrock",
            "type": "message",
            "role": "assistant",
            "model": "claude",
            "content": [{"type": "text", "text": "from bedrock"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    )

    run = await run_task(workspace_env(tmp_path / "ws"), ClaudeAgent(ClaudeConfig(model=profile)))

    (request,) = services.requests("gateway", "POST")
    assert run.trace.content == "from bedrock"
    assert (request.path, request.headers["authorization"].split(" ")[0]) == (
        "/model/arn:aws:bedrock:us-east-1:123456789012:inference-profile%2Fclaude/invoke",
        "AWS4-HMAC-SHA256",
    )
    body = request.json
    assert (body["anthropic_version"], "model" in body, "stream" in body) == (
        "bedrock-2023-05-31",
        False,
        False,
    )
    assert body["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Do the task.", "cache_control": {"type": "ephemeral"}}
            ],
        }
    ]
