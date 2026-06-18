"""``Task`` construction, the portable row shape, and taskset collection.

The model is the row: plain pydantic (``model_validate``/``model_dump``) is the
whole codec for ``hud sync`` and the JSON/JSONL taskset path. ``env`` is carried
as its name, the join key to whatever placement can bring that environment up.
Placement is never part of the row — without an ``runtime=`` provider, execution
defaults to the (not yet wired) HUD-hosted provisioner, which raises a precise
error.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

import pytest

from hud.environment import Environment
from hud.eval import (
    RuntimeConfig,
    RuntimeGPU,
    RuntimeResources,
    Task,
    Taskset,
)

if TYPE_CHECKING:
    from hud.agents.base import Agent


def test_env_task_call_returns_public_task() -> None:
    env = Environment("e")

    @env.template()
    async def solve(n: int):
        yield f"solve:{n}"
        yield 1.0

    runnable = solve(n=3)
    assert isinstance(runnable, Task)
    assert runnable.id == "solve"
    assert runnable.args == {"n": 3}
    assert runnable.env == "e"  # the row carries the env's name, not the object


def test_default_slug_is_task_id_without_args() -> None:
    v = Task(env="e", id="solve")
    assert v.default_slug() == "solve"


def test_default_slug_is_deterministic_with_args() -> None:
    a = Task(env="e", id="solve", args={"b": 2, "a": 1})
    b = Task(env="e", id="solve", args={"a": 1, "b": 2})  # key order differs
    assert a.default_slug() == b.default_slug()  # stable: keys sorted
    assert a.default_slug().startswith("solve-")
    assert a.default_slug() != Task(env="e", id="solve", args={"a": 9}).default_slug()


# ─── the portable row shape ────────────────────────────────────────────


def test_env_serializes_as_name_reference() -> None:
    v = Task(env="team-intel", id="ask", args={"x": 1})
    data = v.model_dump(exclude_none=True)
    assert data["env"] == "team-intel"
    assert data["id"] == "ask"
    assert data["args"] == {"x": 1}


def test_compact_dump_omits_unset_metadata() -> None:
    data = Task(env="e", id="t").model_dump(exclude_none=True)
    assert set(data) == {"env", "id", "args"}  # no None slug/validation/etc.

    data2 = Task(env="e", id="t", slug="s").model_dump(exclude_none=True)
    assert data2["slug"] == "s"


def test_roundtrip_is_stable_through_plain_pydantic() -> None:
    original = Task(
        env="team-intel",
        id="ask",
        args={"difficulty": 3},
        slug="ask-v1",
        validation=[{"name": "submit", "arguments": {"answer": "x"}}],
        agent_config={"system_prompt": "be precise"},
    ).model_dump(exclude_none=True)

    rebuilt = Task.model_validate(original)

    assert rebuilt.env == "team-intel"  # the name is the reference
    assert rebuilt.id == "ask"
    assert rebuilt.args == {"difficulty": 3}
    assert rebuilt.slug == "ask-v1"
    assert rebuilt.validation == original["validation"]
    assert rebuilt.agent_config == {"system_prompt": "be precise"}
    # ...and re-serializing yields the same portable dict.
    assert rebuilt.model_dump(exclude_none=True) == original


def test_runtime_config_roundtrips_as_part_of_task_row() -> None:
    original = Task(
        env="browser",
        id="checkout",
        runtime_config=RuntimeConfig(
            image="hud-browser:firefox",
            resources=RuntimeResources(cpu=2, memory_mb=4096, gpu=RuntimeGPU()),
        ),
    ).model_dump(exclude_none=True)

    rebuilt = Task.model_validate(original)

    assert rebuilt.runtime_config == RuntimeConfig(
        image="hud-browser:firefox",
        resources=RuntimeResources(cpu=2, memory_mb=4096, gpu=RuntimeGPU()),
    )
    assert rebuilt.model_dump(exclude_none=True) == original


def test_runtime_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="Extra inputs"):
        RuntimeConfig.model_validate({"image": "img:tag", "provider_config": {}})


def test_row_validation_rejects_malformed_entries() -> None:
    # pydantic.ValidationError is a ValueError: callers catch one exception type.
    with pytest.raises(ValueError, match="env"):
        Task.model_validate({"id": "t"})
    with pytest.raises(ValueError, match="env"):
        Task.model_validate({"env": {"name": "e"}, "id": "t"})  # an object is not a name
    with pytest.raises(ValueError, match="id"):
        Task.model_validate({"env": "e"})
    with pytest.raises(ValueError, match="args"):
        Task.model_validate({"env": "e", "id": "t", "args": "nope"})


# ─── placement ─────────────────────────────────────────────────────────


async def test_no_placement_defaults_to_hosted_execution() -> None:
    v = Task(env="hosted-env", id="solve", args={"n": 1})
    # No placement means HUD-hosted execution, which serializes the agent
    # spec before submitting anything; a non-gateway agent therefore fails
    # before launch as an isolated failed Run carrying the precise error.
    job = await v.run(cast("Agent", object()))
    (run,) = job.runs
    assert run.trace.is_error
    assert "gateway agent" in (run.trace.error or "")


# ─── taskset collection ────────────────────────────────────────────────


def test_taskset_is_ordered_and_keyed_by_slug() -> None:
    first = Task(env="e", id="solve", args={"n": 1}, slug="first")
    second = Task(env="e", id="solve", args={"n": 2}, slug="second")

    tasks = Taskset("demo", [first, second])

    assert list(tasks) == [first, second]
    assert tasks["first"] is first
    assert list(tasks.filter(["second"])) == [second]
    assert list(tasks.exclude(["first"])) == [second]
    assert list(tasks.items()) == [("first", first), ("second", second)]
    assert tasks.environment_names() == {"e"}


def test_taskset_from_file_loads_json_and_jsonl(tmp_path) -> None:
    entries = [
        Task(env="e", id="solve", args={"n": 1}, slug="one").model_dump(exclude_none=True),
        Task(env="e", id="solve", args={"n": 2}, slug="two").model_dump(exclude_none=True),
    ]

    json_path = tmp_path / "tasks.json"
    json_path.write_text(json.dumps(entries), encoding="utf-8")
    jsonl_path = tmp_path / "tasks.jsonl"
    jsonl_path.write_text("\n".join(json.dumps(entry) for entry in entries), encoding="utf-8")

    assert [t.slug for t in Taskset.from_file(json_path)] == ["one", "two"]
    assert [t.slug for t in Taskset.from_file(jsonl_path)] == ["one", "two"]


def test_file_roundtrip_keeps_rows_and_env_names(tmp_path) -> None:
    authored = [
        Task(env="authored", id="solve", args={"n": 1}, slug="one"),
        Task(env="authored", id="solve", args={"n": 2}, slug="two"),
    ]
    out = Taskset("demo", authored).to_file(tmp_path / "tasks.json")

    loaded = Taskset.from_file(out)

    assert [t.slug for t in loaded] == ["one", "two"]
    assert all(t.env == "authored" for t in loaded)
    assert list(loaded) == authored  # rows survive the file intact (value equality)


def test_taskset_to_file_writes_json_and_jsonl(tmp_path) -> None:
    taskset = Taskset(
        "demo",
        [
            Task(env="e", id="solve", args={"n": 1}, slug="one"),
            Task(env="e", id="solve", args={"n": {"x": 2}}, slug="two"),
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
from hud import Task

local = Task(env="module-env", id="solve", args={"n": 1}, slug="local")
""".strip(),
        encoding="utf-8",
    )

    assert Taskset.from_module(module)["local"].args == {"n": 1}


def test_taskset_from_api_uses_remote_records(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_request(method: str, url: str, **kwargs: object) -> dict[str, object]:
        assert method == "GET"
        if url.endswith("/tasksets/by-name/demo"):
            return {"taskset_id": "ts_123", "name": "Demo"}
        if url.endswith("/tasksets/ts_123/export"):
            return {
                "name": "Demo",
                "tasks": [
                    {
                        # CP export shape: the legacy env qualifier is stripped
                        # server-side, so env + bare scenario arrive already split.
                        "env": "e",
                        "scenario": "solve",
                        "args": {"n": 1},
                        "name": "one",
                    }
                ],
            }
        raise AssertionError(url)

    monkeypatch.setattr("hud.utils.platform.make_request_sync", fake_request)
    monkeypatch.setattr("hud.settings.settings.api_key", "test-key")

    taskset = Taskset.from_api("demo")

    assert taskset.name == "Demo"
    assert taskset["one"].id == "solve"
    assert taskset["one"].env == "e"
    assert taskset["one"].args == {"n": 1}
