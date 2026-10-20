"""``create_agent``: a model name resolves, through the gateway's catalog, to the agent that
speaks its protocol, and that agent's requests go through the HUD gateway."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from hud.agents import create_agent
from tests.fixtures.environments import workspace_task
from tests.fixtures.models import Turn
from tests.fixtures.services import Reply

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import HudEnv
    from tests.fixtures.models import Models
    from tests.fixtures.services import FakeServices

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
    ("name", "protocol", "model"),
    [
        ("openai", "responses", "gpt-5.6"),
        ("claude", "anthropic", "claude-sonnet-4-6"),
        ("Claude Sonnet 4.6", "anthropic", "claude-sonnet-4-6"),
        ("ft:custom-123", "chat", "gpt-5.5"),
        ("minimax-m3", "chat", "MiniMax-M3"),
    ],
    ids=["an-agent-type", "another-agent-type", "a-display-name", "a-catalog-id", "any-case"],
)
async def test_a_created_agent_speaks_its_models_protocol_through_the_gateway(
    name: str,
    protocol: str,
    model: str,
    services: FakeServices,
    models: Models,
    hud_env: HudEnv,
    tmp_path: Path,
) -> None:
    hud_env.set(HUD_API_KEY="hud-key", OPENAI_API_KEY="own-key", ANTHROPIC_API_KEY="own-key")
    services.route(
        "api",
        "GET",
        "/v2/models",
        Reply(json={"items": CATALOG[:2], "total": 3}),
        Reply(json={"items": CATALOG[2:], "total": 3}),
    )
    models.script([Turn(text="done")])

    task = workspace_task(tmp_path / "ws")
    await task().run(create_agent(name))

    (request,) = models.requests()
    assert (request.protocol, request.model, request.credential) == (protocol, model, "hud-key")
