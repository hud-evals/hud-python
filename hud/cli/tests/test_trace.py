from __future__ import annotations

from typing import TYPE_CHECKING

from typer.testing import CliRunner

from hud.cli import trace
from hud.settings import settings

if TYPE_CHECKING:
    import pytest


def test_trace_link_uses_web_uuid(monkeypatch: pytest.MonkeyPatch) -> None:
    trace_id = "03dd2a73d3df4d10a54ae3d87c2d530d"
    monkeypatch.setattr(settings, "api_key", "test-key")
    monkeypatch.setattr(
        trace,
        "_load_remote",
        lambda _: [{"kind": "agent_message", "text": "done"}],
    )

    result = CliRunner().invoke(trace.trace_app, [trace_id])

    assert result.exit_code == 0
    assert "https://hud.ai/trace/03dd2a73-d3df-4d10-a54a-e3d87c2d530d" in result.stdout
