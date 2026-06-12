"""Platform persistence: diff plans, record mapping, and the upload payload."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hud.eval import Task, Taskset
from hud.eval.sync import (
    diff,
    fetch_taskset_tasks,
    resolve_taskset_id,
    task_upload_payload,
    upload_taskset,
)
from hud.utils.platform import PlatformClient

if TYPE_CHECKING:
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


def test_fetched_tasks_strip_env_prefix_to_runnable_local_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The platform may store scenario names env-prefixed ("e:solve"); locally a
    # Task.id must stay env-local ("solve") so start_task resolves against the
    # env's unprefixed scenario registry. The prefix recovers env when the
    # record omits the env field.
    requested: dict[str, str] = {}
    payload = {
        "taskset_id": "ts-id",
        "name": "demo",
        "tasks": [
            {"scenario": "e:solve", "env": "myenv", "name": "a", "args": {"n": 1}},
            {"scenario": "e:solve", "name": "b"},
        ],
    }

    def fake_request(method: str, url: str, **kwargs: object) -> dict:
        requested.update(method=method, url=url)
        return payload

    monkeypatch.setattr("hud.utils.platform.make_request_sync", fake_request)

    name, tasks = fetch_taskset_tasks(PlatformClient("https://api.example", "token"), "ts-id")

    assert requested == {"method": "GET", "url": "https://api.example/tasksets/ts-id/export"}
    assert name == "demo"
    assert [(t.env, t.id) for t in tasks] == [("myenv", "solve"), ("e", "solve")]
    # Round-trip: a fetched task diffs as unchanged against its local twin.
    plan = diff(Taskset("d", [_row("a", 1)]), Taskset("d", [tasks[0]]))
    assert [t.slug for t in plan.unchanged] == ["a"]


def test_resolve_taskset_id_looks_up_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    requested: dict[str, str] = {}

    def fake_request(method: str, url: str, **kwargs: object) -> dict:
        requested.update(method=method, url=url)
        return {"taskset_id": "ts-id", "name": "demo", "tasks": []}

    monkeypatch.setattr("hud.utils.platform.make_request_sync", fake_request)

    resolved = resolve_taskset_id(PlatformClient("https://api.example", "token"), "My Demo")

    assert requested == {"method": "GET", "url": "https://api.example/tasksets/by-name/My%20Demo"}
    assert resolved == ("ts-id", "demo")


def test_resolve_taskset_id_passes_uuids_through() -> None:
    platform = PlatformClient("https://api.example", "token")
    raw = "8f4e0d62-4a3e-4f63-9c5d-1f2a3b4c5d6e"
    assert resolve_taskset_id(platform, raw) == (raw, raw)


def test_upload_taskset_posts_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    upload = Task(env="e", id="solve", args={"n": 1}, slug="solve-one")
    posted: dict[str, object] = {}

    def fake_request(method: str, url: str, json: object = None, **kwargs: object) -> dict:
        posted.update(method=method, url=url, json=json, api_key=kwargs.get("api_key"))
        return {"ok": True}

    monkeypatch.setattr("hud.utils.platform.make_request_sync", fake_request)

    platform = PlatformClient("https://api.example", "token")
    result = upload_taskset(platform, "demo", [upload])

    assert result == {"ok": True}
    assert posted["method"] == "POST"
    assert posted["url"] == "https://api.example/tasks/upload"
    assert posted["api_key"] == "token"
    assert posted["json"] == {
        "taskset_name": "demo",
        "tasks": [
            {
                "name": "solve-one",
                "env": {"name": "e"},
                "scenario": "e:solve",
                "args": {"n": 1},
            },
        ],
    }


def test_task_upload_payload_prefixes_task_id_with_env_name() -> None:
    assert task_upload_payload(Task(env="e", id="solve", args={"n": 1}))["scenario"] == "e:solve"
