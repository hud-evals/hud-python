"""``Task`` construction, the portable row shape, and taskset collection.

``to_dict``/``from_dict`` are the portable identity used by ``hud sync`` and the
JSON/JSONL taskset path: env serializes as a bare name reference and
deserializes to a declarative ``Environment(name)``. Placement is never part of
the row — without an ``on=`` provider, execution defaults to the (not yet
wired) HUD-hosted provisioner, which raises a precise error.
"""

from __future__ import annotations

import json

import pytest

from hud.environment import Environment
from hud.eval import Task, Taskset, task


def test_task_helper_collects_args_and_metadata() -> None:
    env = Environment("e")
    v = task(env, "task", slug="my-slug", validation=[{"name": "submit"}], x=1, y=2)
    assert v.id == "task"
    assert v.args == {"x": 1, "y": 2}
    assert v.slug == "my-slug"
    assert v.validation == [{"name": "submit"}]


def test_env_task_call_returns_public_task() -> None:
    env = Environment("e")

    @env.task()
    async def solve(n: int):
        yield f"solve:{n}"
        yield 1.0

    runnable = solve(n=3)
    assert isinstance(runnable, Task)
    assert runnable.id == "solve"
    assert runnable.args == {"n": 3}
    assert runnable.env is env


def test_default_slug_is_task_id_without_args() -> None:
    v = Task(env=Environment("e"), id="solve")
    assert v.default_slug() == "solve"


def test_default_slug_is_deterministic_with_args() -> None:
    env = Environment("e")
    a = Task(env=env, id="solve", args={"b": 2, "a": 1})
    b = Task(env=env, id="solve", args={"a": 1, "b": 2})  # key order differs
    assert a.default_slug() == b.default_slug()  # stable: keys sorted
    assert a.default_slug().startswith("solve-")
    assert a.default_slug() != Task(env=env, id="solve", args={"a": 9}).default_slug()


# ─── the portable row shape ────────────────────────────────────────────


def test_env_serializes_as_name_reference() -> None:
    v = task(Environment("team-intel"), "ask", x=1)
    data = v.to_dict()
    assert data["env"] == {"name": "team-intel"}
    assert data["task"] == "ask"
    assert data["args"] == {"x": 1}


def test_to_dict_only_includes_set_metadata() -> None:
    data = Task(env=Environment("e"), id="t").to_dict()
    assert set(data) == {"env", "task", "args"}  # no None slug/validation/etc.

    data2 = task(Environment("e"), "t", slug="s", columns={"tier": "easy"}).to_dict()
    assert data2["slug"] == "s"
    assert data2["columns"] == {"tier": "easy"}


def test_roundtrip_is_stable_through_from_dict() -> None:
    original = task(
        Environment("team-intel"),
        "ask",
        slug="ask-v1",
        validation=[{"name": "submit", "arguments": {"answer": "x"}}],
        agent_config={"system_prompt": "be precise"},
        columns={"tier": "hard"},
        difficulty=3,
    ).to_dict()

    rebuilt = Task.from_dict(original)

    assert isinstance(rebuilt.env, Environment)  # bare declarative reference
    assert rebuilt.env.name == "team-intel"
    assert rebuilt.id == "ask"
    assert rebuilt.args == {"difficulty": 3}
    assert rebuilt.slug == "ask-v1"
    assert rebuilt.validation == original["validation"]
    assert rebuilt.agent_config == {"system_prompt": "be precise"}
    assert rebuilt.columns == {"tier": "hard"}
    # ...and re-serializing yields the same portable dict.
    assert rebuilt.to_dict() == original


def test_from_dict_validates_shape() -> None:
    with pytest.raises(ValueError, match="env"):
        Task.from_dict({"task": "t"})
    with pytest.raises(ValueError, match="task id"):
        Task.from_dict({"env": {"name": "e"}})
    with pytest.raises(ValueError, match="args"):
        Task.from_dict({"env": {"name": "e"}, "task": "t", "args": "nope"})


# ─── placement ─────────────────────────────────────────────────────────


async def test_no_placement_defaults_to_provision_stub_with_precise_error() -> None:
    v = task(Environment("hosted-env"), "solve", n=1)
    with pytest.raises(NotImplementedError, match=r"'hosted-env'.*on=spawn") as err:
        async with v.session():
            pass
    assert "Runtime(url)" in str(err.value)


# ─── taskset collection ────────────────────────────────────────────────


def test_taskset_is_ordered_and_keyed_by_slug() -> None:
    env = Environment("e")
    first = task(env, "solve", slug="first", n=1)
    second = task(env, "solve", slug="second", n=2)

    tasks = Taskset("demo", [first, second])

    assert list(tasks) == [first, second]
    assert tasks["first"] is first
    assert list(tasks.filter(["second"])) == [second]
    assert list(tasks.exclude(["first"])) == [second]
    assert list(tasks.items()) == [("first", first), ("second", second)]
    assert tasks.environment_names() == {"e"}


def test_taskset_from_file_loads_json_and_jsonl(tmp_path) -> None:
    env = Environment("e")
    entries = [
        task(env, "solve", slug="one", n=1).to_dict(),
        task(env, "solve", slug="two", n=2).to_dict(),
    ]

    json_path = tmp_path / "tasks.json"
    json_path.write_text(json.dumps(entries), encoding="utf-8")
    jsonl_path = tmp_path / "tasks.jsonl"
    jsonl_path.write_text("\n".join(json.dumps(entry) for entry in entries), encoding="utf-8")

    assert [t.slug for t in Taskset.from_file(json_path)] == ["one", "two"]
    assert [t.slug for t in Taskset.from_file(jsonl_path)] == ["one", "two"]


def test_file_roundtrip_keeps_rows_and_env_names(tmp_path) -> None:
    env = Environment("authored")
    authored = [task(env, "solve", slug="one", n=1), task(env, "solve", slug="two", n=2)]
    out = Taskset("demo", authored).to_file(tmp_path / "tasks.json")

    loaded = Taskset.from_file(out)

    assert [t.slug for t in loaded] == ["one", "two"]
    # Rows come back with bare name-reference envs, not the authored object.
    assert all(t.env.name == "authored" and t.env is not env for t in loaded)
    assert [t.to_dict() for t in loaded] == [t.to_dict() for t in authored]


def test_taskset_to_file_writes_json_and_jsonl(tmp_path) -> None:
    env = Environment("e")
    taskset = Taskset(
        "demo",
        [
            task(env, "solve", slug="one", columns={"tier": "easy"}, n=1),
            task(env, "solve", slug="two", columns={"tier": "hard"}, n={"x": 2}),
        ],
    )

    json_path = taskset.to_file(tmp_path / "tasks.json")
    jsonl_path = taskset.to_file(tmp_path / "tasks.jsonl")

    assert [entry["slug"] for entry in json.loads(json_path.read_text())] == ["one", "two"]
    assert [json.loads(line)["slug"] for line in jsonl_path.read_text().splitlines()] == [
        "one",
        "two",
    ]
    with pytest.raises(ValueError, match=r"use \.json or \.jsonl"):
        taskset.to_file(tmp_path / "tasks.txt")


def test_taskset_from_module_collects_public_tasks(tmp_path) -> None:
    module = tmp_path / "local_tasks.py"
    module.write_text(
        """
from hud import Environment, task

env = Environment("module-env")
local = task(env, "solve", slug="local", n=1)
""".strip(),
        encoding="utf-8",
    )

    assert Taskset.from_module(module)["local"].args == {"n": 1}


def test_taskset_from_api_uses_remote_records(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_request(method: str, url: str, **kwargs: object) -> dict[str, object]:
        assert method == "GET"
        if url.endswith("/tasks/evalset/demo"):
            return {"evalset_id": "ts_123", "evalset_name": "Demo"}
        if url.endswith("/tasks/evalsets/ts_123/tasks-by-id"):
            return {
                "evalset_name": "Demo",
                "tasks": {
                    "1": {
                        "env": {"name": "e"},
                        "scenario": "e:solve",
                        "args": {"n": 1},
                        "slug": "one",
                        "column_values": {"tier": "easy"},
                    }
                },
            }
        raise AssertionError(url)

    monkeypatch.setattr("hud.utils.platform.make_request_sync", fake_request)
    monkeypatch.setattr("hud.settings.settings.api_key", "test-key")

    taskset = Taskset.from_api("demo")

    assert taskset.name == "Demo"
    assert taskset["one"].id == "e:solve"
    assert taskset["one"].env.name == "e"
    assert taskset["one"].args == {"n": 1}
    assert taskset["one"].columns == {"tier": "easy"}
