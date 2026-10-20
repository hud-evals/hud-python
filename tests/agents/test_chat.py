"""What only ``OpenAIChatAgent`` does over Chat Completions: replay the model's reasoning."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from hud.agents import OpenAIChatAgent
from hud.agents.types import OpenAIChatConfig
from tests.fixtures.environments import workspace_task
from tests.fixtures.models import Turn, call

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import HudEnv
    from tests.fixtures.models import Models

MODEL = "qwen3.6-plus"


@pytest.fixture(autouse=True)
def signed_in(hud_env: HudEnv) -> None:
    hud_env.set(HUD_API_KEY="hud-key")


async def test_the_models_reasoning_is_replayed_in_the_conversation(
    models: Models, tmp_path: Path
) -> None:
    models.script(
        [replace(call("bash", command="true"), reasoning="check first"), Turn(text="done")]
    )

    task = workspace_task(tmp_path / "ws")
    await task().run(OpenAIChatAgent(OpenAIChatConfig(model=MODEL)))

    replayed = models.requests()[-1].body["messages"][1]
    assert (replayed["role"], replayed["reasoning_content"]) == ("assistant", "check first")
