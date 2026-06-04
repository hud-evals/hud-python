from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from hud.cli.utils.metadata import fetch_lock_from_registry


@patch("hud.cli.utils.metadata.settings")
@patch("requests.get")
def test_fetch_lock_from_registry_success(mock_get: Any, mock_settings: Any) -> None:
    mock_settings.hud_api_url = "https://api.example.com"
    mock_settings.api_key = None
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"lock": "image: img\n"}
    mock_get.return_value = resp
    lock = fetch_lock_from_registry("org/name:tag")
    assert lock is not None and lock["image"] == "img"


@patch("hud.cli.utils.metadata.settings")
@patch("requests.get")
def test_fetch_lock_from_registry_lock_data_branch(mock_get: Any, mock_settings: Any) -> None:
    mock_settings.hud_api_url = "https://api.example.com"
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"lock_data": {"image": "direct"}}
    mock_get.return_value = resp
    # No tag -> ":latest" is appended internally; org/name form.
    lock = fetch_lock_from_registry("org/name")
    assert lock == {"image": "direct"}


@patch("hud.cli.utils.metadata.settings")
@patch("requests.get")
def test_fetch_lock_from_registry_not_found(mock_get: Any, mock_settings: Any) -> None:
    mock_settings.hud_api_url = "https://api.example.com"
    mock_get.return_value = MagicMock(status_code=404)
    assert fetch_lock_from_registry("org/name:tag") is None


@patch("hud.cli.utils.metadata.settings")
@patch("requests.get")
def test_fetch_lock_from_registry_swallows_errors(mock_get: Any, mock_settings: Any) -> None:
    mock_settings.hud_api_url = "https://api.example.com"
    mock_get.side_effect = RuntimeError("network down")
    assert fetch_lock_from_registry("org/name:tag") is None
