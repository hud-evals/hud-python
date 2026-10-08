"""The featured workflow changes only the flag on existing public environments."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, call
from uuid import uuid4

import pytest

SPEC = importlib.util.spec_from_file_location(
    "mark_featured", Path(__file__).resolve().parents[1] / "mark_featured.py"
)
assert SPEC is not None and SPEC.loader is not None
pipeline = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pipeline
SPEC.loader.exec_module(pipeline)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    return MagicMock(spec=pipeline.PlatformClient)


@pytest.mark.parametrize("name", list(pipeline.catalog()))
def test_cli_only_marks_existing_catalog_environment_featured(name, client, monkeypatch, capsys):
    example = pipeline.catalog()[name]
    registry_id = str(example.registry_id)
    client.get.return_value = {"id": registry_id, "name": name, "public": True}
    client.put.return_value = {"id": registry_id, "featured": True}
    monkeypatch.setattr(pipeline.PlatformClient, "from_settings", lambda: client)
    monkeypatch.setattr(sys, "argv", ["mark_featured.py", name])

    pipeline.main()

    assert client.mock_calls == [
        call.get(f"/registry/{registry_id}"),
        call.put(f"/admin/registry/{registry_id}/featured", json={"featured": True}),
    ]
    assert f"https://hud.ai/environments/{registry_id}" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("field", "value"), [("id", str(uuid4())), ("name", "other-env"), ("public", False)]
)
def test_wrong_or_private_environment_is_not_featured(client, field, value):
    example = pipeline.catalog()["flask-coding"]
    client.get.return_value = {
        "id": str(example.registry_id),
        "name": "flask-coding",
        "public": True,
        field: value,
    }
    with pytest.raises(ValueError, match="existing public environment"):
        pipeline.mark_featured("flask-coding", example, client)
    client.put.assert_not_called()


@pytest.mark.parametrize(("field", "value"), [("id", str(uuid4())), ("featured", False)])
def test_unconfirmed_featured_update_fails(client, field, value):
    example = pipeline.catalog()["flask-coding"]
    client.get.return_value = {
        "id": str(example.registry_id),
        "name": "flask-coding",
        "public": True,
    }
    client.put.return_value = {"id": str(example.registry_id), "featured": True, field: value}
    with pytest.raises(ValueError, match="did not confirm"):
        pipeline.mark_featured("flask-coding", example, client)


@pytest.mark.parametrize(
    ("variable", "value"), [("GITHUB_ACTIONS", "false"), ("GITHUB_REF", "refs/heads/feature")]
)
def test_only_main_branch_workflow_can_change_featured_status(client, monkeypatch, variable, value):
    monkeypatch.setenv(variable, value)
    with pytest.raises(RuntimeError, match="workflow"):
        pipeline.mark_featured("flask-coding", pipeline.catalog()["flask-coding"], client)
    assert client.mock_calls == []
