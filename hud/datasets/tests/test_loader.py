"""Tests for hud.datasets.loader module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from hud.datasets.loader import load_dataset


class TestLoadDataset:
    """Tests for load_dataset() function."""

    @patch("hud.datasets.loader.httpx.Client")
    @patch("hud.datasets.loader.settings")
    def test_load_dataset_success(
        self, mock_settings: MagicMock, mock_client_class: MagicMock
    ) -> None:
        """load_dataset() successfully loads tasks from API."""
        mock_settings.hud_api_url = "https://api.hud.ai"
        mock_settings.api_key = "test_key"

        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"env": {"name": "test"}, "scenario": "checkout", "args": {"user": "alice"}},
            {"env": {"name": "test"}, "scenario": "login", "args": {"user": "bob"}},
        ]
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None
        mock_client_class.return_value = mock_client

        tasks = load_dataset("test-org/test-dataset")

        assert len(tasks) == 2
        assert tasks[0].scenario == "checkout"
        assert tasks[0].args == {"user": "alice"}
        assert tasks[1].scenario == "login"
        mock_client.get.assert_called_once_with(
            "https://api.hud.ai/evals/test-org/test-dataset",
            headers={"Authorization": "Bearer test_key"},
            params={"all": "true"},
        )

    @patch("hud.datasets.loader.httpx.Client")
    @patch("hud.datasets.loader.settings")
    def test_load_dataset_single_task(
        self, mock_settings: MagicMock, mock_client_class: MagicMock
    ) -> None:
        """load_dataset() handles single task (non-list) response."""
        mock_settings.hud_api_url = "https://api.hud.ai"
        mock_settings.api_key = "test_key"

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "env": {"name": "test"},
            "scenario": "checkout",
            "args": {"user": "alice"},
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None
        mock_client_class.return_value = mock_client

        tasks = load_dataset("test-org/test-dataset")

        assert len(tasks) == 1
        assert tasks[0].scenario == "checkout"

    @patch("hud.datasets.loader.httpx.Client")
    @patch("hud.datasets.loader.settings")
    def test_load_dataset_no_api_key(
        self, mock_settings: MagicMock, mock_client_class: MagicMock
    ) -> None:
        """load_dataset() works without API key."""
        mock_settings.hud_api_url = "https://api.hud.ai"
        mock_settings.api_key = None

        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None
        mock_client_class.return_value = mock_client

        tasks = load_dataset("test-org/test-dataset")

        mock_client.get.assert_called_once_with(
            "https://api.hud.ai/evals/test-org/test-dataset",
            headers={},
            params={"all": "true"},
        )

    @patch("hud.datasets.loader.httpx.Client")
    @patch("hud.datasets.loader.settings")
    def test_load_dataset_http_error(
        self, mock_settings: MagicMock, mock_client_class: MagicMock
    ) -> None:
        """load_dataset() raises ValueError on HTTP error."""
        import httpx

        mock_settings.hud_api_url = "https://api.hud.ai"
        mock_settings.api_key = "test_key"

        mock_client = MagicMock()
        mock_client.get.side_effect = httpx.HTTPError("Network error")
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None
        mock_client_class.return_value = mock_client

        with pytest.raises(ValueError, match="Failed to load dataset"):
            load_dataset("test-org/test-dataset")

    @patch("hud.datasets.loader.httpx.Client")
    @patch("hud.datasets.loader.settings")
    def test_load_dataset_json_error(
        self, mock_settings: MagicMock, mock_client_class: MagicMock
    ) -> None:
        """load_dataset() raises ValueError on JSON processing error."""
        mock_settings.hud_api_url = "https://api.hud.ai"
        mock_settings.api_key = "test_key"

        mock_response = MagicMock()
        mock_response.json.side_effect = Exception("Invalid JSON")
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None
        mock_client_class.return_value = mock_client

        with pytest.raises(ValueError, match="Error processing dataset"):
            load_dataset("test-org/test-dataset")

    @patch("hud.datasets.loader.httpx.Client")
    @patch("hud.datasets.loader.settings")
    def test_load_dataset_empty(
        self, mock_settings: MagicMock, mock_client_class: MagicMock
    ) -> None:
        """load_dataset() handles empty dataset."""
        mock_settings.hud_api_url = "https://api.hud.ai"
        mock_settings.api_key = "test_key"

        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None
        mock_client_class.return_value = mock_client

        tasks = load_dataset("test-org/test-dataset")

        assert len(tasks) == 0

    @patch("hud.datasets.loader.httpx.Client")
    @patch("hud.datasets.loader.settings")
    def test_load_dataset_missing_fields(
        self, mock_settings: MagicMock, mock_client_class: MagicMock
    ) -> None:
        """load_dataset() handles tasks with missing optional fields."""
        mock_settings.hud_api_url = "https://api.hud.ai"
        mock_settings.api_key = "test_key"

        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"scenario": "test"},  # Missing env and args
        ]
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None
        mock_client_class.return_value = mock_client

        tasks = load_dataset("test-org/test-dataset")

        assert len(tasks) == 1
        assert tasks[0].scenario == "test"
        assert tasks[0].env is None
        assert tasks[0].args == {}

