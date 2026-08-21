"""``Task`` construction, the portable row shape, and taskset collection.

The model is the row: plain pydantic (``model_validate``/``model_dump``) is the
whole codec for ``hud sync`` and the JSON/JSONL taskset path. ``env`` is carried
as its name, the join key to whatever placement can bring that environment up.
Placement is never part of the row. Factory-created tasks can run locally;
portable rows require a provider or source placement.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

import pytest

from hud.environment import Environment
from hud.eval import (
    HUDRuntime,
    Run,
    RuntimeConfig,
    RuntimeGPU,
    RuntimeResources,
    RuntimeTPU,
    Task,
    Taskset,
)
from hud.eval.runtime.compose import ComposeConfig, ComposeProject, ComposeProjectRef

if TYPE_CHECKING:
    from pathlib import Path

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


def test_slug_defaults_to_task_id_without_args() -> None:
    v = Task(env="e", id="solve")
    assert v.slug == "solve"


def test_slug_default_is_deterministic_with_args() -> None:
    a = Task(env="e", id="solve", args={"b": 2, "a": 1})
    b = Task(env="e", id="solve", args={"a": 1, "b": 2})  # key order differs
    assert a.slug == b.slug  # stable: keys sorted
    assert a.slug.startswith("solve-")
    assert a.slug != Task(env="e", id="solve", args={"a": 9}).slug


def test_slug_rejects_none() -> None:
    with pytest.raises(ValueError, match="slug"):
        Task.model_validate({"env": "e", "id": "solve", "slug": None})


def test_slug_rejects_empty_string() -> None:
    with pytest.raises(ValueError, match="slug"):
        Task(env="e", id="solve", slug="")


def test_slug_rejects_empty_assignment() -> None:
    task = Task(env="e", id="solve", slug="valid")

    with pytest.raises(ValueError, match="slug"):
        task.slug = ""

    assert task.slug == "valid"


# ─── the portable row shape ────────────────────────────────────────────


def test_env_serializes_as_name_reference() -> None:
    v = Task(env="team-intel", id="ask", args={"x": 1})
    data = v.model_dump(exclude_none=True)
    assert data["env"] == "team-intel"
    assert data["id"] == "ask"
    assert data["args"] == {"x": 1}


def test_compact_dump_omits_unset_metadata() -> None:
    data = Task(env="e", id="t").model_dump(exclude_none=True)
    assert set(data) == {"env", "id", "args", "slug"}
    assert data["slug"] == "t"

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
    resources = RuntimeResources(
        cpu=2,
        memory_mb=4096,
        storage_mb=16384,
        gpu=RuntimeGPU(type=["H100", "A100"]),
        os="windows",
        tpu=RuntimeTPU(type="v5", topology="2x2"),
    )
    original = Task(
        env="browser",
        id="checkout",
        runtime_config=RuntimeConfig(
            image="hud-browser:firefox",
            resources=resources,
        ),
    ).model_dump(exclude_none=True)

    rebuilt = Task.model_validate(original)

    assert rebuilt.runtime_config == RuntimeConfig(
        image="hud-browser:firefox",
        resources=resources,
    )
    assert rebuilt.model_dump(exclude_none=True) == original


def test_verifier_roundtrips_as_an_ordinary_nested_task() -> None:
    original = Task(
        env="actor",
        id="solve",
        verifier=Task(
            env="judge",
            id="verify",
            args={"suite": "hidden"},
            runtime_config=RuntimeConfig(image="judge:latest"),
        ),
    ).model_dump(exclude_none=True)

    rebuilt = Task.model_validate(original)

    assert rebuilt.verifier == Task(
        env="judge",
        id="verify",
        args={"suite": "hidden"},
        runtime_config=RuntimeConfig(image="judge:latest"),
    )
    assert rebuilt.model_dump(exclude_none=True) == original


def test_verifier_rejects_another_verifier() -> None:
    with pytest.raises(ValueError, match="nested verifier tasks are not supported"):
        Task(
            env="actor",
            id="solve",
            verifier=Task(
                env="judge",
                id="verify",
                verifier=Task(env="final-judge", id="verify-final"),
            ),
        )


def test_verifier_rejects_another_verifier_on_assignment() -> None:
    task = Task(env="actor", id="solve")

    with pytest.raises(ValueError, match="nested verifier tasks are not supported"):
        task.verifier = Task(
            env="judge",
            id="verify",
            verifier=Task(env="final-judge", id="verify-final"),
        )

    assert task.verifier is None


def test_runtime_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="Extra inputs"):
        RuntimeConfig.model_validate({"image": "img:tag", "provider_config": {}})


def test_runtime_config_omits_null_resource_fields() -> None:
    task = Task(
        env="database",
        id="cutover",
        runtime_config=RuntimeConfig(
            resources=RuntimeResources(
                cpu=1,
                memory_mb=None,
                storage_mb=None,
                gpu=None,
                os=None,
                tpu=None,
            ),
        ),
    )

    assert task.model_dump(mode="json", exclude_none=True)["runtime_config"] == {
        "resources": {"cpu": 1.0}
    }


def test_compose_runtime_config_serializes_by_task_dump_mode(tmp_path: Path) -> None:
    compose = tmp_path / "compose.json"
    compose.write_text(
        json.dumps({"services": {"main": {"image": "hud-env:latest"}}}),
        encoding="utf-8",
    )
    config = RuntimeConfig(compose=ComposeProject(document=compose))
    task = Task(env="database", id="cutover", runtime_config=config)

    payload = task.model_dump(exclude_none=True)
    rebuilt = Task.model_validate(payload)

    assert payload["runtime_config"] == {"compose": {"document": compose}}
    assert rebuilt.runtime_config is not None
    assert rebuilt.runtime_config.compose is not None
    assert rebuilt.runtime_config.compose.document == compose

    payload = task.model_dump(mode="json", exclude_none=True)
    rebuilt = Task.model_validate(payload)

    assert payload["runtime_config"] == {
        "compose": {
            "document": {
                "services": {
                    "main": {
                        "image": "hud-env:latest",
                        "environment": {},
                        "expose": [],
                        "ports": [],
                        "volumes": [],
                    }
                },
                "networks": {},
            },
        }
    }
    assert rebuilt.runtime_config is not None
    assert rebuilt.runtime_config.compose is not None
    assert isinstance(rebuilt.runtime_config.compose.document, ComposeConfig)
    assert str(compose) not in json.dumps(payload)


def test_compose_project_serializes_document_location_not_author_path(tmp_path: Path) -> None:
    project = tmp_path / "artifact"
    project.mkdir()
    compose = project / "compose-project" / "compose.json"
    compose.parent.mkdir()
    compose.write_text(
        json.dumps(
            {
                "services": {
                    "main": {
                        "image": "hud-harbor:local",
                        "build": {"context": "./main"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    payload = RuntimeConfig(compose=ComposeProject(document=compose, root=project)).model_dump(
        mode="json",
        exclude_unset=True,
    )

    assert payload["compose"]["root"] == {"compose_path": "compose-project/compose.json"}
    assert str(tmp_path) not in json.dumps(payload)


def test_compose_runtime_config_round_trips_platform_records() -> None:
    record = {
        "compose": {
            "document": {
                "services": {"main": {"image": "hud-harbor:local"}},
                "networks": {},
            },
            "root": {"compose_path": "compose-project/compose.json"},
        },
    }

    config = RuntimeConfig.model_validate(record)

    assert config.compose is not None
    assert isinstance(config.compose.document, ComposeConfig)
    assert isinstance(config.compose.root, ComposeProjectRef)
    payload = config.model_dump(mode="json", exclude_unset=True)
    assert payload["compose"]["document"]["services"]["main"]["image"] == "hud-harbor:local"
    assert payload["compose"]["root"] == {"compose_path": "compose-project/compose.json"}
    assert RuntimeConfig.model_validate(payload) == config


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


async def test_platform_taskset_defaults_to_hud_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    import hud.eval.taskset as taskset_mod

    seen: dict[str, object] = {}

    async def fake_rollout(task: Task, agent: Agent, **kwargs: object) -> Run:
        seen.update(kwargs)
        run = Run(None, task.id, {})
        run.trace.status = "completed"
        return run

    monkeypatch.setattr(taskset_mod, "rollout", fake_rollout)

    task = Task(env="hosted-env", id="solve", args={"n": 1})
    taskset = taskset_mod.Taskset("hosted", [task], taskset_id="ts_123")
    job = await taskset.run(cast("Agent", object()))

    (run,) = job.runs
    assert run.trace.status == "completed"
    assert isinstance(seen["runtime"], HUDRuntime)
    assert seen["rollout_timeout"] is None


# ─── taskset collection ────────────────────────────────────────────────


def test_taskset_is_ordered_and_keyed_by_slug() -> None:
    first = Task(env="e", id="solve", args={"n": 1}, slug="first")
    second = Task(
        env="e",
        id="solve",
        args={"n": 2},
        slug="second",
        verifier=Task(env="judge", id="verify"),
    )

    tasks = Taskset("demo", [first, second], taskset_id="ts_123")

    assert list(tasks) == [first, second]
    assert tasks["first"] is first
    filtered = tasks.filter(["second"])
    excluded = tasks.exclude(["first"])
    assert list(filtered) == [second]
    assert list(excluded) == [second]
    assert filtered.taskset_id == "ts_123"
    assert excluded.taskset_id == "ts_123"
    assert list(tasks.items()) == [("first", first), ("second", second)]
    assert tasks.environment_names() == {"e", "judge"}


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


def test_taskset_file_preserves_local_compose_project(tmp_path: Path) -> None:
    compose = tmp_path / "compose.yaml"
    compose.write_text("services:\n  main:\n    image: alpine:3.21\n", encoding="utf-8")
    output = Taskset(
        "compose",
        [
            Task(
                env="compose",
                id="solve",
                runtime_config=RuntimeConfig(
                    compose=ComposeProject(document=compose, root=tmp_path),
                ),
            ),
        ],
    ).to_file(tmp_path / "tasks.json")

    row = json.loads(output.read_text(encoding="utf-8"))[0]
    assert row["runtime_config"]["compose"] == {
        "document": "compose.yaml",
        "root": ".",
    }
    loaded = next(iter(Taskset.from_file(output)))
    assert loaded.runtime_config is not None
    assert loaded.runtime_config.compose == ComposeProject(
        document=compose.resolve(),
        root=tmp_path.resolve(),
    )


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
    assert taskset.taskset_id == "ts_123"
    assert taskset["one"].id == "solve"
    assert taskset["one"].env == "e"
    assert taskset["one"].args == {"n": 1}
