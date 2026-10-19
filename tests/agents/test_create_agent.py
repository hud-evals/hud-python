"""``create_agent``: a model name resolves, through the gateway's catalog, to the agent that
speaks its protocol, and that agent's requests go through the HUD gateway."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from hud.agents import OpenAIAgent, create_agent
from tests.fixtures.environments import run_task, workspace_env
from tests.support import say
from tests.support.platform import API_KEY

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from hud.agents.base import Agent
    from tests.support import HudEnv, Models, Platform

CATALOG = [
    {
        "id": "ft:custom-123",
        "model_name": "gpt-5.5",
        "sdk_agent_type": "openai_compatible",
        "provider": {"name": "openai"},
    },
    {"id": "MiniMax-M3", "model_name": "MiniMax-M3", "sdk_agent_type": "openai_compatible"},
    {"id": "claude-sonnet-4-6", "name": "Claude Sonnet 4.6", "sdk_agent_type": "claude"},
]


@pytest.mark.parametrize(
    ("build", "protocol", "model", "credential"),
    [
        pytest.param(
            lambda: create_agent("openai"), "responses", "gpt-5.6", f"Bearer {API_KEY}", id="openai"
        ),
        pytest.param(
            lambda: create_agent("claude"), "anthropic", "claude-sonnet-4-6", API_KEY, id="claude"
        ),
        pytest.param(
            lambda: create_agent("ft:custom-123"),
            "chat",
            "gpt-5.5",
            f"Bearer {API_KEY}",
            id="a-catalog-id",
        ),
        pytest.param(
            lambda: create_agent("minimax-m3"),
            "chat",
            "MiniMax-M3",
            f"Bearer {API_KEY}",
            id="a-catalog-name-in-any-case",
        ),
        pytest.param(
            lambda: OpenAIAgent.load(create_agent("openai").dump()),
            "responses",
            "gpt-5.6",
            f"Bearer {API_KEY}",
            id="dumped-and-loaded",
        ),
    ],
)
async def test_a_created_agent_speaks_its_models_protocol_through_the_gateway(
    build: Callable[[], Agent],
    protocol: str,
    model: str,
    credential: str,
    platform: Platform,
    models: Models,
    hud_env: HudEnv,
    tmp_path: Path,
) -> None:
    """The provider's own keys are set too, and still not used."""
    hud_env.set(OPENAI_API_KEY="openai-key", ANTHROPIC_API_KEY="anthropic-key")
    platform.page_size = 2
    for entry in CATALOG:
        platform.add_model(**entry)
    models.script([say("done")])

    await run_task(workspace_env(tmp_path / "ws"), build())

    (request,) = models.requests()
    sent = request.headers.get("authorization") or request.headers.get("x-api-key")
    assert (request.protocol, request.model, sent) == (protocol, model, credential)
