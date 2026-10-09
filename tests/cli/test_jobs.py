from __future__ import annotations

import json
from typing import Any

import pytest
from typer.testing import CliRunner

from hud.cli import jobs
from hud.cli.__main__ import app
from hud.settings import settings
from hud.utils.platform import PlatformClient


class _Client:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    def get(self, path: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self.calls.append((path, params))
        return {"items": []}


def test_job_detail_accepts_compact_id_and_prints_canonical_link(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compact_id = "03dd2a73d3df4d10a54ae3d87c2d530d"
    canonical_id = "03dd2a73-d3df-4d10-a54a-e3d87c2d530d"
    client = _Client()
    monkeypatch.setattr(settings, "api_key", "test-key")
    monkeypatch.setattr(settings, "hud_web_url", "https://hud.test")
    monkeypatch.setattr(PlatformClient, "from_settings", classmethod(lambda cls: client))

    result = CliRunner().invoke(jobs.jobs_app, ["get", compact_id])

    assert result.exit_code == 0
    assert client.calls == [(f"/jobs/{canonical_id}/traces", {"limit": 20})]
    assert f"https://hud.test/jobs/{canonical_id}" in result.stdout


@pytest.mark.parametrize("verb", [[], ["get"]])
@pytest.mark.parametrize(
    ("before", "after"),
    [
        ([], ["--json", "--limit", "7", "--quiet"]),
        (["--json", "--limit", "7", "--quiet"], []),
        (["--json", "--limit", "5", "--quiet"], ["--limit", "7"]),
    ],
)
def test_job_detail_shorthand_preserves_options(monkeypatch, verb, before, after):
    job_id = "03dd2a73-d3df-4d10-a54a-e3d87c2d530d"
    client = _Client()
    monkeypatch.setattr(PlatformClient, "from_settings", classmethod(lambda cls: client))
    result = CliRunner().invoke(app, ["jobs", *verb, *before, job_id, *after])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == []
    assert client.calls == [(f"/jobs/{job_id}/traces", {"limit": 7})]


def test_bare_jobs_still_lists(monkeypatch):
    client = _Client()
    monkeypatch.setattr(PlatformClient, "from_settings", classmethod(lambda cls: client))
    result = CliRunner().invoke(app, ["jobs", "--json", "-n", "7"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == []
    assert client.calls == [("/jobs", {"limit": 7})]


@pytest.mark.parametrize("command", ["jobs", "trace"])
def test_unknown_resource_verb_is_a_usage_error(command):
    result = CliRunner().invoke(app, [command, "unknown", "--json"])
    assert result.exit_code == 2
    assert json.loads(result.stdout)["error"] == "usage"
