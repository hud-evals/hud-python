from __future__ import annotations

from unittest.mock import MagicMock, patch

from hud.cli.utils.metadata import fetch_lock_from_registry


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
