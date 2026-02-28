"""Tests for hud.datasets.loader module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from hud.datasets.loader import load_tasks, save_tasks
from hud.eval.task import Task


class TestLoadTasks:
    """Tests for load_tasks() function."""

    @patch("hud.datasets.loader.httpx.Client")
    @patch("hud.datasets.loader.settings")
    def test_load_tasks_success(
        self, mock_settings: MagicMock, mock_client_class: MagicMock
    ) -> None:
        """load_tasks() successfully loads tasks from API."""
        mock_settings.hud_api_url = "https://api.hud.ai"
        mock_settings.api_key = "test_key"

        mock_response = MagicMock()
        # EvalsetTasksResponse format: tasks keyed by task ID
        mock_response.json.return_value = {
            "evalset_id": "evalset-123",
            "evalset_name": "test-dataset",
            "tasks": {
                "task-1": {
                    "env": {"name": "test"},
                    "scenario": "checkout",
                    "external_id": "checkout-smoke",
                    "args": {"user": "alice"},
                },
                "task-2": {
                    "env": {"name": "test"},
                    "scenario": "login",
                    "external_id": "login-smoke",
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
        task_slugs = {t.slug for t in tasks}
        assert task_slugs == {"checkout-smoke", "login-smoke"}
        # Platform IDs are internal and should not be inferred from dict keys
        assert all(t.id is None for t in tasks)
        mock_client.get.assert_called_once_with(
            "https://api.hud.ai/tasks/evalset/test-org/test-dataset",
            headers={"Authorization": "Bearer test_key"},
            params={"all": "true"},
        )

    @patch("hud.datasets.loader.httpx.Client")
    @patch("hud.datasets.loader.settings")
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
                    "external_id": "checkout-smoke",
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
        assert tasks[0].slug == "checkout-smoke"
        assert tasks[0].id is None

    @patch("hud.datasets.loader.httpx.Client")
    @patch("hud.datasets.loader.settings")
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

    @patch("hud.datasets.loader.httpx.Client")
    @patch("hud.datasets.loader.settings")
    def test_load_tasks_taskset_not_found(
        self, mock_settings: MagicMock, mock_client_class: MagicMock
    ) -> None:
        """load_tasks() raises HTTPStatusError when taskset doesn't exist."""
        import httpx

        mock_settings.hud_api_url = "https://api.hud.ai"
        mock_settings.api_key = "test_key"

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found", request=MagicMock(), response=mock_response
        )

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None
        mock_client_class.return_value = mock_client

        with pytest.raises(httpx.HTTPStatusError):
            load_tasks("nonexistent-taskset")

    @patch("hud.datasets.loader.httpx.Client")
    @patch("hud.datasets.loader.settings")
    def test_load_tasks_network_error(
        self, mock_settings: MagicMock, mock_client_class: MagicMock
    ) -> None:
        """load_tasks() raises ConnectError when API is unreachable."""
        import httpx

        mock_settings.hud_api_url = "https://api.hud.ai"
        mock_settings.api_key = "test_key"

        mock_client = MagicMock()
        mock_client.get.side_effect = httpx.ConnectError("Connection refused")
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None
        mock_client_class.return_value = mock_client

        with pytest.raises(httpx.ConnectError):
            load_tasks("my-taskset")

    @patch("hud.datasets.loader.httpx.Client")
    @patch("hud.datasets.loader.settings")
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

    @patch("hud.datasets.loader.httpx.Client")
    @patch("hud.datasets.loader.settings")
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
        assert tasks[0].slug is None
        assert tasks[0].id is None
        assert tasks[0].args == {}


class TestSaveTasks:
    """Tests for save_tasks() function."""

    @patch("hud.datasets.loader.httpx.Client")
    @patch("hud.datasets.loader.settings")
    def test_save_tasks_posts_to_upload_and_omits_id(
        self, mock_settings: MagicMock, mock_client_class: MagicMock
    ) -> None:
        """save_tasks() uses /tasks/upload and relies on slug instead of id."""
        mock_settings.hud_api_url = "https://api.hud.ai"
        mock_settings.api_key = "test_key"

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "evalset_id": "evalset-123",
            "tasks_created": 1,
            "tasks_updated": 0,
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None
        mock_client_class.return_value = mock_client

        taskset_id = save_tasks(
            "test-org/test-dataset",
            [
                Task(
                    env={"name": "test-env"},
                    scenario="checkout",
                    args={"user": "alice"},
                    slug="checkout-smoke",
                    id="internal-id-should-not-upload",
                )
            ],
        )

        assert taskset_id == "evalset-123"
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args.args[0] == "https://api.hud.ai/tasks/upload"
        payload = call_args.kwargs["json"]
        assert payload["name"] == "test-org/test-dataset"
        assert payload["tasks"][0]["slug"] == "checkout-smoke"
        assert "id" not in payload["tasks"][0]
