"""What an agent does when its provider fails: what it retries, and how the run ends.

A real provider will not fail on demand, so a scripted one does here, and a stub
CLI reports the failure its real CLI would. How each CLI's exit status decides
the run lives in that CLI's wire module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from tests.agents.families import TOOL_FAMILIES, NotApplicable, over_families, over_family_cases
from tests.fixtures.environments import run_task, workspace_env
from tests.support import Reply, Turn, say, stream_error

if TYPE_CHECKING:
    from pathlib import Path

    from tests.agents.families import Driver
    from tests.support import HudEnv, Models


@pytest.mark.parametrize(
    ("family", "row"),
    over_families(
        {
            **dict.fromkeys(("claude", "openai", "gemini", "openai_compatible")),
            "claude_cli": None,
            "codex_cli": None,
            "browser_use": NotApplicable(
                "browser-use handles its model's errors inside its own loop"
            ),
        }
    ),
)
async def test_a_provider_fault_ends_the_run_with_its_error(
    family: str, row: None, driver: Driver
) -> None:
    driver.fault("model unavailable")

    run = await driver.run(driver.agent())

    assert run.trace.status == "error"
    assert "model unavailable" in (run.trace.error or "")


@dataclass(frozen=True)
class Replies:
    """The provider's replies in arrival order, and how the run ends: its status, how many
    requests it made, and a phrase of its error."""

    turns: list[Turn]
    status: str
    requests: int
    error: str | None = None


TRANSIENT = {"retry-after-ms": "0"}
BUSY = Turn(fail=Reply(status=529, headers=TRANSIENT, json={"error": {"message": "busy"}}))
BAD_JSON = Turn(
    native=({"type": "tool_use", "id": "toolu_bad", "name": "bash", "input": '{"command": ls}'},)
)
SDK_RETRIES = {"an-overloaded-reply-is-retried": Replies([BUSY, say("ok")], "completed", 2)}

REPLIES: dict[str, dict[str, Replies] | NotApplicable] = {
    "claude": {
        **SDK_RETRIES,
        "a-transient-stream-error-is-retried": Replies(
            [stream_error("overloaded_error", headers=TRANSIENT), say("ok")], "completed", 2
        ),
        "three-transient-stream-errors-end-the-run": Replies(
            [stream_error("gateway_timeout", headers=TRANSIENT)] * 3, "error", 3
        ),
        "a-stream-request-error-is-not-retried": Replies(
            [stream_error("invalid_request_error", headers=TRANSIENT)], "error", 1
        ),
        "invalid-tool-json-is-retried": Replies([BAD_JSON, say("ok")], "completed", 2),
    },
    "openai": SDK_RETRIES,
    "gemini": {
        "no-candidates": Replies(
            [Turn(fail=Reply(json={"candidates": []}))],
            "error",
            1,
            "Gemini returned no candidates for model gemini-3.1-pro-preview",
        ),
        "a-malformed-function-call": Replies(
            [Turn(finish="MALFORMED_FUNCTION_CALL")],
            "error",
            1,
            "Provider returned a malformed function call",
        ),
    },
    "openai_compatible": SDK_RETRIES,
    "claude_cli": NotApplicable("the CLI retries inside its own process"),
    "codex_cli": NotApplicable("the CLI retries inside its own process"),
    "browser_use": NotApplicable("browser-use retries inside its own loop"),
}


@pytest.mark.parametrize(("family", "case"), over_family_cases(REPLIES))
async def test_what_a_provider_reply_lets_the_agent_retry(
    family: str, case: Replies, models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    models.queue(case.turns)
    tools = TOOL_FAMILIES[family]

    run = await run_task(workspace_env(tmp_path / "ws"), tools.agent_cls(tools.config()))

    assert (run.trace.status, len(models.requests())) == (case.status, case.requests)
    if case.error is not None:
        assert case.error in (run.trace.error or "")
