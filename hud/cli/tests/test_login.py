"""CLI behavior when the platform serves no browser login.

``hud login`` speaks the device flow (RFC 8628) to ``/auth/device/code``. A
deployment that does not serve that endpoint answers 404, and the command used
to print the status code and the body, which reads as an outage rather than as
the one thing the user can do about it: authenticate with an API key instead.

The transport is mocked rather than the module's own helpers, so the command
runs its real request path and the assertions are on what a user sees.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
from typer.testing import CliRunner

from hud.cli import app
from hud.cli import login as login_module

if TYPE_CHECKING:
    import pytest

runner = CliRunner()


# Bound before any test replaces the name, so the factory below builds a real
# client rather than calling itself.
_HTTPX_CLIENT = httpx.Client


def _client_serving(response: httpx.Response):
    """A drop-in ``httpx.Client`` whose every request gets *response*."""

    def factory(*args: object, **kwargs: object) -> httpx.Client:
        return _HTTPX_CLIENT(transport=httpx.MockTransport(lambda _request: response))

    return factory


def test_login_points_at_an_api_key_where_the_device_flow_is_not_served(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        login_module.httpx,
        "Client",
        _client_serving(httpx.Response(404, json={"error": "not_found"})),
    )

    result = runner.invoke(app, ["login", "--quiet"])

    assert result.exit_code == 1
    assert "404" in result.output
    assert "hud set HUD_API_KEY" in result.output


def test_login_reports_any_other_failure_as_it_finds_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        login_module.httpx,
        "Client",
        _client_serving(httpx.Response(503, text="upstream is down")),
    )

    result = runner.invoke(app, ["login", "--quiet"])

    assert result.exit_code == 1
    assert "503" in result.output
    assert "upstream is down" in result.output
