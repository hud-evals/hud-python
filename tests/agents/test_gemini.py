"""What only ``GeminiAgent`` does: how it drops old screenshots, and how its replies
fail."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from hud.agents import GeminiAgent
from hud.agents.types import GeminiConfig
from hud.capabilities import Capability
from tests.fixtures.environments import workspace_task
from tests.fixtures.models import Turn, call
from tests.fixtures.screen import fake_screen
from tests.fixtures.services import Reply

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import HudEnv
    from tests.fixtures.models import Models

MODEL = "gemini-3.1-pro-preview"


@pytest.fixture(autouse=True)
def signed_in(hud_env: HudEnv) -> None:
    hud_env.set(HUD_API_KEY="k")


async def test_only_the_three_newest_computer_turns_keep_their_screenshots(
    models: Models, tmp_path: Path
) -> None:
    models.script([call("click_at", x=156, y=208)] * 4 + [Turn(text="done")])

    async with fake_screen() as screen:
        task = workspace_task(tmp_path / "ws", capabilities=(Capability.rfb(url=screen.url),))
        await task().run(GeminiAgent(GeminiConfig(model=MODEL)))

    responses = [
        part["functionResponse"]
        for content in models.requests()[-1].body["contents"]
        for part in content["parts"]
        if "functionResponse" in part
    ]
    assert [bool(response.get("parts")) for response in responses] == [False, True, True, True]


@pytest.mark.parametrize(
    ("turn", "error"),
    [
        (Turn(fail=Reply(json={"candidates": []})), "Gemini returned no candidates"),
        (Turn(finish="MALFORMED_FUNCTION_CALL"), "Provider returned a malformed function call"),
    ],
    ids=["no-candidates", "a-malformed-function-call"],
)
async def test_a_reply_it_cannot_use_ends_the_run_with_its_error(
    turn: Turn, error: str, models: Models, tmp_path: Path
) -> None:
    models.queue([turn])

    task = workspace_task(tmp_path / "ws")
    (run,) = (await task().run(GeminiAgent(GeminiConfig(model=MODEL)))).runs

    assert (run.trace.status, len(models.requests())) == ("error", 1)
    assert error in (run.trace.error or "")
