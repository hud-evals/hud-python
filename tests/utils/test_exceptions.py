"""Tests for the HUD SDK exception hierarchy."""

from __future__ import annotations

import httpx

from hud.utils.exceptions import (
    HudAuthenticationError,
    HudException,
    HudRequestError,
)
from hud.utils.hints import (
    HUD_API_KEY_MISSING,
    PRO_PLAN_REQUIRED,
    RATE_LIMIT_HIT,
)


class TestHudException:
    def test_message_and_str(self):
        error = HudException("Something broke")
        assert error.message == "Something broke"
        assert str(error) == "Something broke"
        assert error.hints == []

    def test_response_json_in_str(self):
        error = HudException("Bad payload", response_json={"detail": "nope"})
        assert str(error) == "Bad payload | Response: {'detail': 'nope'}"

    def test_subclass_default_hints(self):
        error = HudAuthenticationError("API key missing")
        assert error.hints == [HUD_API_KEY_MISSING]
        assert error.hints[0].title == "HUD API key required"
        # Hint copy evolved; keep the assertion robust to minor copy changes
        tips = error.hints[0].tips
        assert tips and "Set HUD_API_KEY" in tips[0]


class TestHudRequestError:
    """Test HudRequestError specific behavior."""

    def test_401_adds_auth_hint(self):
        error = HudRequestError("Unauthorized", status_code=401)
        assert HUD_API_KEY_MISSING in error.hints

    def test_403_adds_auth_hint(self):
        error = HudRequestError("Forbidden", status_code=403)
        assert HUD_API_KEY_MISSING in error.hints

    def test_403_pro_plan_message_sets_pro_hint(self):
        """403 with Pro wording should map to PRO_PLAN_REQUIRED, not auth."""
        error = HudRequestError("Feature requires Pro plan", status_code=403)
        assert PRO_PLAN_REQUIRED in error.hints
        assert HUD_API_KEY_MISSING not in error.hints

    def test_403_pro_plan_detail_sets_pro_hint(self):
        """403 with detail indicating Pro should map to PRO_PLAN_REQUIRED."""
        error = HudRequestError(
            "Forbidden",
            status_code=403,
            response_json={"detail": "Requires Pro plan"},
        )
        assert PRO_PLAN_REQUIRED in error.hints
        assert HUD_API_KEY_MISSING not in error.hints

    def test_429_adds_rate_limit_hint(self):
        error = HudRequestError("Too Many Requests", status_code=429)
        assert RATE_LIMIT_HIT in error.hints

    def test_other_status_no_default_hints(self):
        error = HudRequestError("Server Error", status_code=500)
        assert error.hints == []

    def test_explicit_hints_override_defaults(self):
        from hud.utils.hints import Hint

        custom_hint = Hint(title="Custom Error", message="This is a custom hint")
        error = HudRequestError("Unauthorized", status_code=401, hints=[custom_hint])
        assert error.hints == [custom_hint]
        assert HUD_API_KEY_MISSING not in error.hints

    def test_from_httpx_error(self):
        request = httpx.Request("GET", "https://api.test.com")
        response = httpx.Response(404, json={"detail": "Not found"}, request=request)
        httpx_error = httpx.HTTPStatusError("Not found", request=request, response=response)

        error = HudRequestError.from_httpx_error(httpx_error, context="Testing")

        assert error.status_code == 404
        assert "Testing" in str(error)
        assert "Not found" in str(error)
        assert error.response_json == {"detail": "Not found"}

    def test_string_representation(self):
        error = HudRequestError(
            "Request failed", status_code=404, response_json={"error": "Not found"}
        )

        error_str = str(error)
        assert "Request failed" in error_str
        assert "Status: 404" in error_str
        assert "Response JSON: {'error': 'Not found'}" in error_str
