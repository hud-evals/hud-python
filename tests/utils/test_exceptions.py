"""SDK errors carry the hints a user needs, and the console renders them with the error."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from hud.utils import HUDConsole
from hud.utils.exceptions import (
    HudAuthenticationError,
    HudConfigError,
    HudException,
    HudRequestError,
)
from hud.utils.hints import (
    CREDITS_EXHAUSTED,
    HUD_API_KEY_MISSING,
    INVALID_CONFIG,
    PRO_PLAN_REQUIRED,
    RATE_LIMIT_HIT,
    Hint,
)

STATUS_HINTS = {
    "401 asks for an API key": ({"status_code": 401}, [HUD_API_KEY_MISSING]),
    "402 says credits ran out": ({"status_code": 402}, [CREDITS_EXHAUSTED]),
    "429 says the rate limit was hit": ({"status_code": 429}, [RATE_LIMIT_HIT]),
    "a bare 403 asks for an API key": ({"status_code": 403}, [HUD_API_KEY_MISSING]),
    "a 403 whose message names Pro asks for the plan": (
        {"status_code": 403, "message": "Feature requires Pro plan"},
        [PRO_PLAN_REQUIRED],
    ),
    "a 403 whose detail names Pro asks for the plan": (
        {"status_code": 403, "response_json": {"detail": "Requires Pro plan"}},
        [PRO_PLAN_REQUIRED],
    ),
    "a 500 has no hint": ({"status_code": 500}, []),
    "explicit hints replace the status default": (
        {"status_code": 401, "hints": [Hint(title="Custom", message="custom")]},
        [Hint(title="Custom", message="custom")],
    ),
}


@pytest.mark.parametrize(("kwargs", "hints"), STATUS_HINTS.values(), ids=STATUS_HINTS.keys())
def test_a_request_error_hints_at_the_fix_for_its_status(
    kwargs: dict[str, Any], hints: list[Hint]
) -> None:
    error = HudRequestError(kwargs.pop("message", "failed"), **kwargs)

    assert error.hints == hints


@pytest.mark.parametrize(
    ("error", "hints"),
    [
        (HudException("broke"), []),
        (HudAuthenticationError("no key"), [HUD_API_KEY_MISSING]),
        (HudConfigError("bad config"), [INVALID_CONFIG]),
    ],
)
def test_each_error_kind_carries_its_default_hints(error: HudException, hints: list[Hint]) -> None:
    assert error.hints == hints


def test_a_request_error_from_httpx_keeps_status_context_and_body() -> None:
    request = httpx.Request("GET", "https://api.test")
    response = httpx.Response(404, json={"detail": "Not found"}, request=request)

    error = HudRequestError.from_httpx_error(
        httpx.HTTPStatusError("Not found", request=request, response=response), context="Testing"
    )

    assert (error.status_code, error.response_json) == (404, {"detail": "Not found"})
    assert str(error).startswith("Testing: ")
    assert "Not found" in str(error)


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
