"""Platform persistence: diff plans, record mapping, and the upload payload."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from hud.eval import Task, Taskset
from hud.eval.runtime import RuntimeConfig
from hud.eval.sync import (
    diff,
    fetch_taskset_tasks,
    resolve_taskset_id,
    task_upload_payload,
    upload_taskset,
)
from hud.utils.platform import PlatformClient

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _row(slug: str, n: object) -> Task:
    return Task(env="e", id="solve", args={"n": n}, slug=slug)


def test_diff_classifies_create_update_unchanged_and_remote_only() -> None:
    local_a = _row("a", 1)
    local_b = _row("b", 2)
    local_c = _row("c", 3)
    remote_a = Task.model_validate(local_a.model_dump())
    remote_b = _row("b", 99)
    remote_old = _row("old", 0)

    plan = diff(
        Taskset("demo", [local_a, local_b, local_c]),
        Taskset("demo", [remote_a, remote_b, remote_old]),
    )

    assert [t.slug for t in plan.to_create] == ["c"]
    assert [t.slug for t in plan.to_update] == ["b"]
    assert [t.slug for t in plan.unchanged] == ["a"]
    assert [t.slug for t in plan.remote_only] == ["old"]
    assert plan.to_apply == [local_c, local_b]
    assert "Create: 1" in plan.summary()


def test_fetched_tasks_map_canonical_export_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The CP export emits the canonical {env, scenario} pair (any legacy v5 env
    # qualifier is stripped server-side), so the SDK maps the fields straight
    # onto Task without re-deriving anything from the names.
    requested: dict[str, str] = {}
    payload = {
        "taskset_id": "ts-id",
        "name": "demo",
        "tasks": [
            {"scenario": "solve", "env": "myenv", "name": "a", "args": {"n": 1}},
            {"scenario": "fix_bug", "env": "other", "name": "b"},
        ],
    }

    def fake_request(method: str, url: str, **kwargs: object) -> dict[str, Any]:
        requested.update(method=method, url=url)
        return payload

    monkeypatch.setattr("hud.utils.platform.make_request_sync", fake_request)

    name, tasks = fetch_taskset_tasks(PlatformClient("https://api.example", "token"), "ts-id")

    assert requested == {"method": "GET", "url": "https://api.example/v2/tasksets/ts-id/export"}
    assert name == "demo"
    assert [(t.env, t.id, t.slug) for t in tasks] == [
        ("myenv", "solve", "a"),
        ("other", "fix_bug", "b"),
    ]


def test_resolve_taskset_id_looks_up_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    requested: dict[str, str] = {}

    def fake_request(method: str, url: str, **kwargs: object) -> dict[str, Any]:
        requested.update(method=method, url=url)
        return {"taskset_id": "ts-id", "name": "demo", "tasks": []}

    monkeypatch.setattr("hud.utils.platform.make_request_sync", fake_request)

    resolved = resolve_taskset_id(PlatformClient("https://api.example", "token"), "My Demo")

    assert requested == {
        "method": "GET",
        "url": "https://api.example/v2/tasksets/by-name/My%20Demo",
    }
    assert resolved == ("ts-id", "demo")


def test_resolve_taskset_id_passes_uuids_through() -> None:
    platform = PlatformClient("https://api.example", "token")
    raw = "8f4e0d62-4a3e-4f63-9c5d-1f2a3b4c5d6e"
    assert resolve_taskset_id(platform, raw) == (raw, raw)


def test_upload_taskset_posts_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    upload = Task(env="e", id="solve", args={"n": 1}, slug="solve-one")
    posted: dict[str, object] = {}

    def fake_request(
        method: str, url: str, json: object = None, **kwargs: object
    ) -> dict[str, Any]:
        posted.update(method=method, url=url, json=json, api_key=kwargs.get("api_key"))
        return {"ok": True}

    monkeypatch.setattr("hud.utils.platform.make_request_sync", fake_request)

    platform = PlatformClient("https://api.example", "token")
    result = upload_taskset(platform, "demo", [upload])

    assert result == {"ok": True}
    assert posted["method"] == "POST"
    assert posted["url"] == "https://api.example/v2/tasks/upload"
    assert posted["api_key"] == "token"
    assert posted["json"] == {
        "taskset_name": "demo",
        "tasks": [
            {
                "name": "solve-one",
                "env": {"name": "e"},
                "task_id": "solve",
                "args": {"n": 1},
            },
        ],
    }


def test_upload_taskset_places_a_new_taskset(monkeypatch: pytest.MonkeyPatch) -> None:
    """`project_id` reaches the platform so a created taskset lands in that Project."""
    posted: dict[str, Any] = {}

    def fake_request(method: str, url: str, json: object = None, **kwargs: Any) -> dict[str, Any]:
        posted.update(json=json)
        return {}

    monkeypatch.setattr("hud.utils.platform.make_request_sync", fake_request)

    upload_taskset(
        PlatformClient("https://api.example", "token"),
        "demo",
        [],
        project_id="project-1",
    )

    assert posted["json"]["project_id"] == "project-1"


def test_upload_taskset_omits_project_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a project the payload is unchanged, so the default Project applies."""
    posted: dict[str, Any] = {}

    def fake_request(method: str, url: str, json: object = None, **kwargs: Any) -> dict[str, Any]:
        posted.update(json=json)
        return {}

    monkeypatch.setattr("hud.utils.platform.make_request_sync", fake_request)

    upload_taskset(PlatformClient("https://api.example", "token"), "demo", [])

    assert "project_id" not in posted["json"]


def test_task_upload_payload_sends_env_and_bare_task_id() -> None:
    payload = task_upload_payload(Task(env="e", id="solve", args={"n": 1}))

    assert payload["env"] == {"name": "e"}
    assert payload["task_id"] == "solve"
    assert "scenario" not in payload


def test_task_upload_payload_includes_runtime_config() -> None:
    task = Task(
        env="e",
        id="solve",
        runtime_config=RuntimeConfig(image="img:tag"),
    )

    payload = task_upload_payload(task)

    assert payload["runtime_config"] == {"image": "img:tag"}


def test_task_upload_payload_embeds_compose_document(tmp_path: Path) -> None:
    compose = tmp_path / "compose.json"
    compose.write_text(
        json.dumps(
            {
                "services": {
                    "main": {"image": "registry.example/hud-main:latest"},
                    "database": {"image": "postgres:16"},
                }
            }
        ),
        encoding="utf-8",
    )
    task = Task(
        env="e",
        id="solve",
        runtime_config=RuntimeConfig(compose=compose, compose_service_access=True),
    )

    payload = task_upload_payload(task)

    assert payload["runtime_config"]["compose"]["services"]["database"]["image"] == "postgres:16"
    assert payload["runtime_config"]["compose_service_access"] is True
    assert str(compose) not in json.dumps(payload)


def test_task_upload_payload_preserves_runtime_config_null_override() -> None:
    task = Task(
        env="e",
        id="solve",
        runtime_config=RuntimeConfig(resources=None),
    )

    payload = task_upload_payload(task)

    assert payload["runtime_config"] == {"resources": None}


def test_task_upload_payload_includes_verifier_task() -> None:
    task = Task(
        env="actor",
        id="solve",
        verifier=Task(
            env="judge",
            id="verify",
            runtime_config=RuntimeConfig(image="judge:latest"),
        ),
    )

    payload = task_upload_payload(task)

    assert payload["verifier"] == {
        "env": "judge",
        "id": "verify",
        "args": {},
        "slug": "verify",
        "runtime_config": {"image": "judge:latest"},
    }
