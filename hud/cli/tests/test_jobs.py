from __future__ import annotations

from typing import TYPE_CHECKING, Any

from typer.testing import CliRunner

from hud.cli import jobs
from hud.settings import settings
from hud.utils.platform import PlatformClient

if TYPE_CHECKING:
    import pytest


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

    result = CliRunner().invoke(jobs.jobs_app, [compact_id])

    assert result.exit_code == 0
    assert client.calls == [(f"/jobs/{canonical_id}/traces", {"limit": 20})]
    assert f"https://hud.test/jobs/{canonical_id}" in result.stdout
