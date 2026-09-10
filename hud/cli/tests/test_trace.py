from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from hud.cli import app, trace
from hud.settings import settings
from hud.utils.platform import PlatformClient


def test_trace_link_uses_web_uuid(monkeypatch: pytest.MonkeyPatch) -> None:
    trace_id = "03dd2a73d3df4d10a54ae3d87c2d530d"
    monkeypatch.setattr(settings, "api_key", "test-key")
    monkeypatch.setattr(
        trace,
        "_load_remote",
        lambda _: [{"kind": "agent_message", "text": "done"}],
    )

    result = CliRunner().invoke(app, ["trace", trace_id])

    assert result.exit_code == 0
    assert "https://hud.ai/trace/03dd2a73-d3df-4d10-a54a-e3d87c2d530d" in result.stdout


@pytest.mark.parametrize("options_first", [False, True])
def test_trace_json_accepts_options_on_either_side_of_id(
    monkeypatch: pytest.MonkeyPatch, options_first: bool
) -> None:
    trace_id = "03dd2a73d3df4d10a54ae3d87c2d530d"
    events = [{"kind": "agent_message", "text": "done"}]
    monkeypatch.setattr(settings, "api_key", "test-key")
    monkeypatch.setattr(settings, "telemetry_local_dir", None)

    def get_events(self: PlatformClient, path: str) -> dict[str, object]:
        assert path == f"/trace/{trace_id}/events"
        return {"events": events}

    monkeypatch.setattr(PlatformClient, "get", get_events)

    args = ["--json", trace_id] if options_first else [trace_id, "--json"]
    result = CliRunner().invoke(app, ["trace", *args])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == events
