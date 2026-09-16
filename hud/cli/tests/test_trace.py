from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from hud.cli.__main__ import app
from hud.settings import settings
from hud.utils.platform import PlatformClient

if TYPE_CHECKING:
    from pathlib import Path


def test_trace_link_uses_web_uuid(monkeypatch: pytest.MonkeyPatch) -> None:
    trace_id = "03dd2a73d3df4d10a54ae3d87c2d530d"
    monkeypatch.setattr(settings, "api_key", "test-key")
    monkeypatch.setattr(settings, "telemetry_local_dir", None)
    monkeypatch.setattr(
        PlatformClient,
        "get",
        lambda self, path: {"events": [{"kind": "agent_message", "text": "done"}]},
    )

    result = CliRunner().invoke(app, ["trace", "get", trace_id])

    assert result.exit_code == 0
    assert "https://hud.ai/trace/03dd2a73-d3df-4d10-a54a-e3d87c2d530d" in result.stdout


@pytest.mark.parametrize("options_first", [False, True])
@pytest.mark.parametrize("verb", [[], ["get"]])
def test_trace_json_accepts_options_on_either_side_of_id(
    monkeypatch: pytest.MonkeyPatch, options_first: bool, verb: list[str]
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
    result = CliRunner().invoke(app, ["trace", *verb, *args])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == events


def test_local_trace_skips_a_record_cut_short_by_an_interrupted_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace_id = "03dd2a73d3df4d10a54ae3d87c2d530d"
    monkeypatch.setattr(settings, "telemetry_local_dir", str(tmp_path))
    step = {
        "start_time": "2026-01-01T00:00:00Z",
        "attributes": {
            "hud.schema": "hud.step.v1",
            "hud.payload": {"source": "agent", "content": "done"},
        },
    }
    (tmp_path / f"{trace_id}.jsonl").write_text(
        json.dumps(step) + "\n" + json.dumps(step)[:40], encoding="utf-8"
    )

    result = CliRunner().invoke(app, ["trace", "get", trace_id, "--json"])

    assert result.exit_code == 0, result.output
    assert [event["kind"] for event in json.loads(result.stdout)] == ["agent_message"]
    assert "Skipped 1 incomplete span record" in result.output
