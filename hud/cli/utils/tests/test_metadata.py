from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from hud.cli.utils.metadata import (
    analyze_from_metadata,
    fetch_lock_from_registry,
)


@patch("hud.cli.utils.metadata.settings")
@patch("requests.get")
def test_fetch_lock_from_registry_success(mock_get, mock_settings):
    mock_settings.hud_api_url = "https://api.example.com"
    mock_settings.api_key = None
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"lock": "image: img\n"}
    mock_get.return_value = resp
    lock = fetch_lock_from_registry("org/name:tag")
    assert lock is not None and lock["image"] == "img"


@pytest.mark.asyncio
@patch("hud.cli.utils.metadata.console")
@patch("hud.cli.utils.metadata.fetch_lock_from_registry")
async def test_analyze_from_metadata_registry(mock_fetch, mock_console):
    mock_fetch.return_value = {"image": "img", "environment": {"toolCount": 0}}
    await analyze_from_metadata("org/name:tag", "json", verbose=False)
    assert mock_console.print_json.called
