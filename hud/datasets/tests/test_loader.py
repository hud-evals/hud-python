"""Tests for hud.datasets.loader module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from hud.datasets.loader import load_tasks


class TestLoadTasks:
    """Tests for load_tasks() function."""

    @patch("httpx.Client")
    @patch("hud.settings.settings")
    def test_load_tasks_success(
        self, mock_settings: MagicMock, mock_client_class: MagicMock
    ) -> None:
        """load_tasks() successfully loads tasks from API."""
        mock_settings.hud_api_url = "https://api.hud.ai"
        mock_settings.api_key = "test_key"

        mock_response = MagicMock()
        # New EvalsetTasksResponse format: tasks keyed by task ID
        mock_response.json.return_value = {
            "evalset_id": "evalset-123",
            "evalset_name": "test-dataset",
            "tasks": {
                "task-1": {
                    "env": {"name": "test"},
                    "scenario": "checkout",
                    "args": {"user": "alice"},
                },
                "task-2": {
                    "env": {"name": "test"},
                    "scenario": "login",
                    "args": {"user": "bob"},
                },
            },
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None
        mock_client_class.return_value = mock_client

        tasks = load_tasks("test-org/test-dataset")

        assert len(tasks) == 2
        # Tasks are keyed by ID in dict, order may vary
        scenarios = {t.scenario for t in tasks}
        assert scenarios == {"checkout", "login"}
        # Check task IDs are set from dict keys
        task_ids = {t.id for t in tasks}
        assert task_ids == {"task-1", "task-2"}
        mock_client.get.assert_called_once_with(
            "https://api.hud.ai/tasks/evalset/test-org/test-dataset",
            headers={"Authorization": "Bearer test_key"},
            params={"all": "true"},
        )

    @patch("httpx.Client")
    @patch("hud.settings.settings")
    def test_load_tasks_single_task(
        self, mock_settings: MagicMock, mock_client_class: MagicMock
    ) -> None:
        """load_tasks() handles single task in EvalsetTasksResponse."""
        mock_settings.hud_api_url = "https://api.hud.ai"
        mock_settings.api_key = "test_key"

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "evalset_id": "evalset-123",
            "evalset_name": "test-dataset",
            "tasks": {
                "task-1": {
                    "env": {"name": "test"},
                    "scenario": "checkout",
                    "args": {"user": "alice"},
                },
            },
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None
        mock_client_class.return_value = mock_client

        tasks = load_tasks("test-org/test-dataset")

        assert len(tasks) == 1
        assert tasks[0].scenario == "checkout"
        assert tasks[0].id == "task-1"

    @patch("httpx.Client")
    @patch("hud.settings.settings")
    def test_load_tasks_no_api_key(
        self, mock_settings: MagicMock, mock_client_class: MagicMock
    ) -> None:
        """load_tasks() works without API key."""
        mock_settings.hud_api_url = "https://api.hud.ai"
        mock_settings.api_key = None

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "evalset_id": "evalset-123",
            "evalset_name": "test-dataset",
            "tasks": {},
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None
        mock_client_class.return_value = mock_client

        tasks = load_tasks("test-org/test-dataset")

        assert len(tasks) == 0
        mock_client.get.assert_called_once_with(
            "https://api.hud.ai/tasks/evalset/test-org/test-dataset",
            headers={},
            params={"all": "true"},
        )

    @patch("httpx.Client")
    @patch("hud.settings.settings")
    def test_load_tasks_http_error(
        self, mock_settings: MagicMock, mock_client_class: MagicMock
    ) -> None:
        """load_tasks() raises ValueError on HTTP error."""
        import httpx

        mock_settings.hud_api_url = "https://api.hud.ai"
        mock_settings.api_key = "test_key"

        mock_client = MagicMock()
        mock_client.get.side_effect = httpx.HTTPError("Network error")
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None
        mock_client_class.return_value = mock_client

        with pytest.raises(ValueError, match="Failed to load tasks"):
            load_tasks("test-org/test-dataset")

    @patch("httpx.Client")
    @patch("hud.settings.settings")
    def test_load_tasks_json_error(
        self, mock_settings: MagicMock, mock_client_class: MagicMock
    ) -> None:
        """load_tasks() raises ValueError on JSON processing error."""
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

        with pytest.raises(ValueError, match="Failed to load tasks"):
            load_tasks("test-org/test-dataset")

    @patch("httpx.Client")
    @patch("hud.settings.settings")
    def test_load_tasks_empty(self, mock_settings: MagicMock, mock_client_class: MagicMock) -> None:
        """load_tasks() handles empty dataset."""
        mock_settings.hud_api_url = "https://api.hud.ai"
        mock_settings.api_key = "test_key"

        mock_response = MagicMock()
        mock_response.json.return_value = {"tasks": {}}
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None
        mock_client_class.return_value = mock_client

        tasks = load_tasks("test-org/test-dataset")

        assert len(tasks) == 0

    @patch("httpx.Client")
    @patch("hud.settings.settings")
    def test_load_tasks_missing_fields(
        self, mock_settings: MagicMock, mock_client_class: MagicMock
    ) -> None:
        """load_tasks() handles tasks with missing optional fields (but env is required)."""
        mock_settings.hud_api_url = "https://api.hud.ai"
        mock_settings.api_key = "test_key"

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "tasks": {"task-1": {"env": {"name": "test-env"}, "scenario": "test"}},
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None
        mock_client_class.return_value = mock_client

        tasks = load_tasks("test-org/test-dataset")

        assert len(tasks) == 1
        assert tasks[0].scenario == "test"
        assert tasks[0].id == "task-1"
        assert tasks[0].args == {}
