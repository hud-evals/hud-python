from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from hud.cli.utils.metadata import (
    analyze_from_metadata,
    check_local_cache,
    fetch_lock_from_registry,
)

if TYPE_CHECKING:
    from pathlib import Path


@patch("hud.cli.utils.metadata.settings")
@patch("requests.get")
def test_fetch_lock_from_registry_success(mock_get, mock_settings):
    mock_settings.hud_telemetry_url = "https://api.example.com"
    mock_settings.api_key = None
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"lock": "image: img\n"}
    mock_get.return_value = resp
    lock = fetch_lock_from_registry("org/name:tag")
    assert lock is not None and lock["image"] == "img"


def test_check_local_cache_not_found(tmp_path: Path, monkeypatch):
    # Point registry to empty dir
    from hud.cli.utils import registry as reg

    monkeypatch.setattr(reg, "get_registry_dir", lambda: tmp_path)
    assert check_local_cache("org/name:tag") is None


@pytest.mark.asyncio
@patch("hud.cli.utils.metadata.console")
@patch("hud.cli.utils.metadata.list_registry_entries")
@patch("hud.cli.utils.metadata.load_from_registry")
@patch("hud.cli.utils.metadata.extract_digest_from_image")
async def test_analyze_from_metadata_local(mock_extract, mock_load, mock_list, mock_console):
    mock_extract.return_value = "abcd"
    mock_load.return_value = {"image": "img", "environment": {"toolCount": 0}}
    mock_list.return_value = []
    await analyze_from_metadata("img@sha256:abcd", "json", verbose=False)
    # Should print JSON
    assert mock_console.print_json.called
