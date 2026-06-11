"""Platform persistence: diff plans, record mapping, and the upload payload."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hud.eval import Task, Taskset
from hud.eval.sync import (
    diff,
    fetch_taskset_tasks,
    resolve_taskset_id,
    task_upload_payload,
    taskset_column_definitions,
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
    # Platform records key tasks as env-prefixed "e:solve"; locally a Task.id
    # must stay env-local ("solve") so start_task resolves against the env's
    # unprefixed scenario registry. The prefix recovers env when the record
    # omits the env block.
    payload = {
        "evalset_name": "demo",
        "tasks": {
            "1": {"scenario": "e:solve", "env": {"name": "myenv"}, "slug": "a", "args": {"n": 1}},
            "2": {"scenario": "e:solve", "slug": "b"},
        },
    }

    def fake_request(method: str, url: str, **kwargs: object) -> dict:
        return payload

    monkeypatch.setattr("hud.utils.platform.make_request_sync", fake_request)

    _, tasks = fetch_taskset_tasks(PlatformClient("https://api.example", "token"), "ts-id")

    assert [(t.env, t.id) for t in tasks] == [("myenv", "solve"), ("e", "solve")]
    # Round-trip: a fetched task diffs as unchanged against its local twin.
    plan = diff(Taskset("d", [_row("a", 1)]), Taskset("d", [tasks[0]]))
    assert [t.slug for t in plan.unchanged] == ["a"]


def test_resolve_taskset_id_passes_uuids_through() -> None:
    platform = PlatformClient("https://api.example", "token")
    raw = "8f4e0d62-4a3e-4f63-9c5d-1f2a3b4c5d6e"
    assert resolve_taskset_id(platform, raw) == (raw, raw)


def test_upload_taskset_posts_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    upload = Task(env="e", id="solve", args={"n": 1}, slug="solve-one", columns={"tier": "easy"})
    posted: dict[str, object] = {}

    def fake_request(method: str, url: str, json: object = None, **kwargs: object) -> dict:
        posted.update(method=method, url=url, json=json, api_key=kwargs.get("api_key"))
        return {"ok": True}

    monkeypatch.setattr("hud.utils.platform.make_request_sync", fake_request)

    platform = PlatformClient("https://api.example", "token")
    result = upload_taskset(
        platform, "demo", [upload], columns=taskset_column_definitions([upload])
    )

    assert result == {"ok": True}
    assert posted["method"] == "POST"
    assert posted["url"] == "https://api.example/tasks/upload"
    assert posted["api_key"] == "token"
    assert posted["json"] == {
        "name": "demo",
        "tasks": [
            {
                "slug": "solve-one",
                "env": {"name": "e"},
                "scenario": "e:solve",
                "args": {"n": 1},
                "column_values": {"tier": "easy"},
            },
        ],
        "columns": {"tier": {"type": "text"}},
    }


def test_task_upload_payload_prefixes_task_id_with_env_name() -> None:
    assert task_upload_payload(Task(env="e", id="solve", args={"n": 1}))["scenario"] == "e:solve"


def test_taskset_column_definitions_infer_types() -> None:
    tasks = [
        Task(env="e", id="t", slug="a", columns={"tier": "easy", "score": 1, "tags": ["x"]}),
        Task(env="e", id="t", slug="b", columns={"tier": "hard", "score": 2.5, "tags": ["y", "z"]}),
    ]

    definitions = taskset_column_definitions(tasks)

    assert definitions == {
        "tier": {"type": "text"},
        "score": {"type": "number"},
        "tags": {"type": "multi-select", "options": ["x", "y", "z"]},
    }
    assert taskset_column_definitions([Task(env="e", id="t", slug="c")]) is None
