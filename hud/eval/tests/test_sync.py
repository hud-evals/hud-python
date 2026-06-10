"""Platform persistence: diff plans, record mapping, and the upload payload."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hud.environment import Environment
from hud.eval import Task, Taskset, task
from hud.eval.sync import (
    diff,
    resolve_taskset_id,
    task_upload_payload,
    taskset_column_definitions,
    upload_taskset,
)
from hud.utils.platform import PlatformClient

if TYPE_CHECKING:
    import pytest


def test_diff_classifies_create_update_unchanged_and_remote_only() -> None:
    env = Environment("e")
    local_a = task(env, "solve", slug="a", n=1)
    local_b = task(env, "solve", slug="b", n=2)
    local_c = task(env, "solve", slug="c", n=3)
    remote_a = Task.from_dict(local_a.to_dict())
    remote_b = task(env, "solve", slug="b", n=99)
    remote_old = task(env, "solve", slug="old", n=0)

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


def test_diff_treats_platform_prefixed_task_ids_as_equal() -> None:
    # Platform records come back env-prefixed ("e:solve"); a local "solve"
    # with identical content must diff as unchanged, not an update.
    env = Environment("e")
    local = task(env, "solve", slug="a", n=1)
    remote = Task(env=Environment("e"), id="e:solve", args={"n": 1}, slug="a")

    plan = diff(Taskset("d", [local]), Taskset("d", [remote]))

    assert [t.slug for t in plan.unchanged] == ["a"]


def test_resolve_taskset_id_passes_uuids_through() -> None:
    platform = PlatformClient("https://api.example", "token")
    raw = "8f4e0d62-4a3e-4f63-9c5d-1f2a3b4c5d6e"
    assert resolve_taskset_id(platform, raw) == (raw, raw)


def test_upload_taskset_posts_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    env = Environment("e")
    upload = task(env, "solve", slug="solve-one", columns={"tier": "easy"}, n=1)
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
    env = Environment("e")
    assert task_upload_payload(task(env, "solve", n=1))["scenario"] == "e:solve"
    assert task_upload_payload(Task(env=env, id="e:solve"))["scenario"] == "e:solve"


def test_taskset_column_definitions_infer_types() -> None:
    env = Environment("e")
    tasks = [
        task(env, "t", slug="a", columns={"tier": "easy", "score": 1, "tags": ["x"]}),
        task(env, "t", slug="b", columns={"tier": "hard", "score": 2.5, "tags": ["y", "z"]}),
    ]

    definitions = taskset_column_definitions(tasks)

    assert definitions == {
        "tier": {"type": "text"},
        "score": {"type": "number"},
        "tags": {"type": "multi-select", "options": ["x", "y", "z"]},
    }
    assert taskset_column_definitions([task(env, "t", slug="c")]) is None
