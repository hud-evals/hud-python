"""What only ``ClaudeAgent`` does: which stream errors it retries."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from hud.agents import ClaudeAgent
from hud.agents.types import ClaudeConfig
from tests.fixtures.environments import workspace_task
from tests.fixtures.models import Turn, stream_error

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import HudEnv
    from tests.fixtures.models import Models

MODEL = "claude-sonnet-4-6"


@pytest.fixture(autouse=True)
def signed_in(hud_env: HudEnv) -> None:
    hud_env.set(HUD_API_KEY="k")


@pytest.mark.parametrize(
    ("errors", "kind", "status", "requests"),
    [
        (1, "overloaded_error", "completed", 2),
        (3, "gateway_timeout", "error", 3),
        (1, "invalid_request_error", "error", 1),
    ],
    ids=["a-transient-error-is-retried", "three-transient-errors-fail", "a-request-error-fails"],
)
async def test_an_error_inside_the_stream_is_retried_only_when_transient(
    errors: int, kind: str, status: str, requests: int, models: Models, tmp_path: Path
) -> None:
    models.queue(
        [stream_error(kind, headers={"retry-after-ms": "0"})] * errors + [Turn(text="done")]
    )

    task = workspace_task(tmp_path / "ws")
    (run,) = (await task().run(ClaudeAgent(ClaudeConfig(model=MODEL)))).runs

    assert (run.trace.status, len(models.requests())) == (status, requests)
