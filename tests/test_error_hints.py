"""SDK errors carry the hints a user needs, and the console renders them with the error."""

from __future__ import annotations

from typing import Any

import pytest

from hud.utils import HUDConsole
from hud.utils.exceptions import (
    HudRequestError,
)
from hud.utils.hints import (
    CREDITS_EXHAUSTED,
    PRO_PLAN_REQUIRED,
    RATE_LIMIT_HIT,
    Hint,
)

STATUS_HINTS = {
    "a 402 asks for credits": (
        {"status_code": 402, "response_json": {"detail": "no credits"}},
        [CREDITS_EXHAUSTED],
    ),
    "a 403 whose detail names Pro asks for the plan": (
        {"status_code": 403, "response_json": {"detail": "Requires Pro plan"}},
        [PRO_PLAN_REQUIRED],
    ),
}


@pytest.mark.parametrize(("kwargs", "hints"), STATUS_HINTS.values(), ids=STATUS_HINTS.keys())
def test_a_request_error_hints_at_the_fix_for_its_status(
    kwargs: dict[str, Any], hints: list[Hint]
) -> None:
    error = HudRequestError(kwargs.pop("message", "failed"), **kwargs)

    assert error.hints == hints


def test_the_console_renders_an_error_with_its_details_hints_and_support_link(
    capsys: pytest.CaptureFixture[str],
) -> None:
    HUDConsole().render_exception(
        HudRequestError("Too many", status_code=429, response_text="slow down")
    )

    err = " ".join(capsys.readouterr().err.split())
    for shown in [
        "HudRequestError: Too many",
        "Status 429",
        "Response slow down",
        f"{RATE_LIMIT_HIT.title}: {RATE_LIMIT_HIT.message}",
        "github.com/hud-evals/hud-python/issues",
    ]:
        assert shown in err
