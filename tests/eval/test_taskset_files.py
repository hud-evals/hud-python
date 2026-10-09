"""Taskset files and platform tasksets: the portable row format a taskset is
written to, read from, and fetched as."""

from __future__ import annotations

import json
import textwrap
import uuid
from typing import TYPE_CHECKING, Any

import pytest
from inline_snapshot import snapshot

from hud import Environment
from hud.eval import (
    ComposeProject,
    RuntimeConfig,
    RuntimeGPU,
    RuntimeLimits,
    RuntimeResources,
    RuntimeTPU,
    Task,
    Taskset,
)
from hud.eval.runtime import DockerBindMount

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from tests.harness import FakeServices, HudEnv


def authored_rows(root: Path) -> list[Task]:
    """One row per shape a taskset file carries."""
    compose = root / "project" / "compose.yaml"
    compose.parent.mkdir(parents=True)
    compose.write_text("services:\n  main:\n    image: lab:${TAG}\n")
    (compose.parent / ".env").write_text("TAG=1\n")
    env = Environment("lab")

    @env.template()
    async def add(a: int, b: int):
        yield f"add {a} {b}"
        yield 1.0

    return [
        add(a=2, b=3),
        Task(
            env="lab",
            id="add",
            args={"a": 1, "b": 1},
            slug="one-plus-one",
            validation=[{"name": "answer", "arguments": {"expected": 2}}],
            agent_config={"timeout_seconds": 30.0},
            columns={"difficulty": "easy"},
        ),
        Task(
            env="gpu",
            id="train",
            slug="train-on-gpus",
            runtime_config=RuntimeConfig(
                image="registry.test/trainer:1",
                resources=RuntimeResources(
                    cpu=8,
                    memory_mb=32768,
                    storage_mb=102400,
                    gpu=RuntimeGPU(type=["H100", "A100"], count=2),
                    os="linux",
                ),
                limits=RuntimeLimits(startup_timeout_s=600, run_timeout_s=7200),
            ),
        ),
        Task(
            env="tpu",
            id="train",
            slug="train-on-tpus",
            runtime_config=RuntimeConfig(
                resources=RuntimeResources(tpu=RuntimeTPU(type="v5e", topology="2x4"))
            ),
        ),
        Task(
            env="actor",
            id="solve",
            runtime_config=RuntimeConfig(image="actor:1"),
            verifier=Task(
                env="judge",
                id="verify",
                args={"expected": "secret"},
                runtime_config=RuntimeConfig(resources=None),
            ),
        ),
        Task(
            env="compose",
            id="solve",
            slug="solve-in-compose",
            runtime_config=RuntimeConfig(
                compose=ComposeProject(document=compose, root=compose.parent, service_access=True)
            ),
        ),
    ]


async def test_a_taskset_written_as_json_reads_back_row_for_row(tmp_path: Path) -> None:
    rows = authored_rows(tmp_path)

    written = Taskset("demo", rows).to_file(tmp_path / "out" / "tasks.json")

    assert json.loads(written.read_text()) == snapshot(
        [
            {"env": "lab", "id": "add", "args": {"a": 2, "b": 3}, "slug": "add-001a8858"},
            {
                "env": "lab",
                "id": "add",
                "args": {"a": 1, "b": 1},
                "slug": "one-plus-one",
                "validation": [{"name": "answer", "arguments": {"expected": 2}}],
                "agent_config": {"timeout_seconds": 30.0},
                "columns": {"difficulty": "easy"},
            },
            {
                "env": "gpu",
                "id": "train",
                "args": {},
                "slug": "train-on-gpus",
                "runtime_config": {
                    "image": "registry.test/trainer:1",
                    "resources": {
                        "cpu": 8.0,
                        "memory_mb": 32768,
                        "storage_mb": 102400,
                        "gpu": {"type": ["H100", "A100"], "count": 2},
                        "os": "linux",
                    },
                    "limits": {"startup_timeout_s": 600, "run_timeout_s": 7200},
                },
            },
            {
                "env": "tpu",
                "id": "train",
                "args": {},
                "slug": "train-on-tpus",
                "runtime_config": {"resources": {"tpu": {"type": "v5e", "topology": "2x4"}}},
            },
            {
                "env": "actor",
                "id": "solve",
                "args": {},
                "slug": "solve",
                "runtime_config": {"image": "actor:1"},
                "verifier": {
                    "env": "judge",
                    "id": "verify",
                    "args": {"expected": "secret"},
                    "slug": "verify-9c4c5d01",
                    "runtime_config": {"resources": None},
                },
            },
            {
                "env": "compose",
                "id": "solve",
                "args": {},
                "slug": "solve-in-compose",
                "runtime_config": {
                    "compose": {
                        "document": "../project/compose.yaml",
                        "root": "../project",
                        "service_access": True,
                    }
                },
            },
        ]
    )
    loaded = Taskset.from_file(written)
    assert loaded.name == "tasks"
    assert [row.model_dump() for row in loaded] == [row.model_dump() for row in rows]


async def test_a_taskset_written_as_jsonl_holds_one_row_per_line(tmp_path: Path) -> None:
    rows = authored_rows(tmp_path)

    written = Taskset("demo", rows).to_file(tmp_path / "tasks.jsonl")

    lines = written.read_text().splitlines()
    assert [json.loads(line)["slug"] for line in lines] == [row.slug for row in rows]
    assert [row.model_dump() for row in Taskset.from_file(written)] == [
        row.model_dump() for row in rows
    ]


def test_a_rows_slug_names_its_task_and_args() -> None:
    row = Task(env="lab", id="add", args={"a": 2, "b": 3})

    assert Task(env="lab", id="add").slug == "add"
    assert row.slug == "add-001a8858"
    assert Task(env="lab", id="add", args={"b": 3, "a": 2}).slug == row.slug
    assert Task(env="lab", id="add", args={"a": 3, "b": 3}).slug != row.slug


def write(path: Path, text: str) -> Path:
    path.write_text(textwrap.dedent(text))
    return path


READS: dict[str, tuple[str, str, list[str]]] = {
    "a single JSON object is a taskset of one": (
        "tasks.json",
        '{"env": "lab", "id": "add"}',
        ["add"],
    ),
    "blank JSONL lines are skipped": (
        "tasks.jsonl",
        '{"env": "lab", "id": "a"}\n\n{"env": "lab", "id": "b"}\n',
        ["a", "b"],
    ),
}


@pytest.mark.parametrize(("name", "text", "slugs"), READS.values(), ids=READS.keys())
def test_a_data_file_reads_into_rows(
    name: str, text: str, slugs: list[str], tmp_path: Path
) -> None:
    loaded = Taskset.from_file(write(tmp_path / name, text))

    assert [row.slug for row in loaded] == slugs


def test_a_module_contributes_its_public_tasks_lists_and_tasksets(tmp_path: Path) -> None:
    source = write(
        tmp_path / "tasks.py",
        """
        from hud import Task, Taskset

        single = Task(env="lab", id="single")
        listed = [Task(env="lab", id="first"), Task(env="lab", id="second"), "not a task"]
        grouped = Taskset("group", [Task(env="lab", id="grouped")])
        _private = Task(env="lab", id="private")
        """,
    )

    assert sorted(row.slug for row in Taskset.from_module(source)) == [
        "first",
        "grouped",
        "second",
        "single",
    ]


def test_a_taskset_is_an_ordered_collection_keyed_by_slug() -> None:
    rows = [
        Task(env="lab", id="a"),
        Task(env="lab", id="b", verifier=Task(env="judge", id="verify")),
        Task(env="other", id="c"),
    ]
    taskset = Taskset("demo", rows, taskset_id="ts-1")

    assert (len(taskset), list(taskset), taskset["b"]) == (3, rows, rows[1])
    assert list(taskset.items()) == [(row.slug, row) for row in rows]
    assert [row.slug for row in taskset.filter(["c", "a"])] == ["a", "c"]
    assert [row.slug for row in taskset.exclude(["a"])] == ["b", "c"]
    assert taskset.filter(["a"]).taskset_id == taskset.exclude(["a"]).taskset_id == "ts-1"
    assert taskset.environment_names() == {"lab", "judge", "other"}


def nested_verifier() -> Task:
    inner = Task(env="judge", id="verify", verifier=Task(env="judge", id="again"))
    return Task(env="lab", id="add", verifier=inner)


def assign(row: Task, field: str, value: Any) -> None:
    setattr(row, field, value)


def compose_outside_root(root: Path) -> Any:
    (root / "a").mkdir()
    (root / "b").mkdir()
    document = write(root / "b" / "compose.yaml", "services: {main: {image: lab}}\n")
    return ComposeProject(document=document, root=root / "a")


INVALID: dict[str, tuple[Callable[[Path], Any], str]] = {
    "a row without an env": (
        lambda tmp: Task.model_validate({"id": "add"}),
        r"env\n  Field required",
    ),
    "an env given as an object": (
        lambda tmp: Task.model_validate({"env": {"name": "lab"}, "id": "add"}),
        r"env\n  Input should be a valid string",
    ),
    "a row without an id": (
        lambda tmp: Task.model_validate({"env": "lab"}),
        r"id\n  Field required",
    ),
    "args that are not an object": (
        lambda tmp: Task.model_validate({"env": "lab", "id": "add", "args": "a=1"}),
        r"args\n  Input should be a valid dictionary",
    ),
    "a null slug": (
        lambda tmp: Task.model_validate({"env": "lab", "id": "add", "slug": None}),
        r"slug\n  Input should be a valid string",
    ),
    "an empty slug": (
        lambda tmp: Task(env="lab", id="add", slug=""),
        r"slug\n  String should have at least 1 character",
    ),
    "an empty slug assigned later": (
        lambda tmp: assign(Task(env="lab", id="add"), "slug", ""),
        r"slug\n  String should have at least 1 character",
    ),
    "a verifier with a verifier": (
        lambda tmp: nested_verifier(),
        "nested verifier tasks are not supported",
    ),
    "a verifier with a verifier assigned later": (
        lambda tmp: assign(Task(env="lab", id="add"), "verifier", nested_verifier().verifier),
        "nested verifier tasks are not supported",
    ),
    "an unknown runtime field": (
        lambda tmp: RuntimeConfig.model_validate({"image": "lab", "provider_config": {}}),
        r"provider_config\n  Extra inputs are not permitted",
    ),
    "both an image and a Compose project": (
        lambda tmp: RuntimeConfig(
            image="lab", compose=ComposeProject(document=write(tmp / "c.yaml", "services: {}\n"))
        ),
        "runtime_config accepts either image or compose, not both",
    ),
    "a TPU topology that is not a grid": (
        lambda tmp: RuntimeTPU(type="v5e", topology="eight"),
        r"topology\n  String should match pattern",
    ),
    "a Compose file outside its project root": (
        compose_outside_root,
        "Compose file must be inside its project root",
    ),
    "a local Compose file under a platform root": (
        lambda tmp: ComposeProject.model_validate(
            {
                "document": str(write(tmp / "c.yaml", "services: {}\n")),
                "root": {"compose_path": "c"},
            }
        ),
        "Compose source and project root must use the same form",
    ),
    "a relative bind mount source": (
        lambda tmp: DockerBindMount("data", "/data"),
        "Docker bind mount source must be absolute",
    ),
    "a relative bind mount target": (
        lambda tmp: DockerBindMount("/data", "data"),
        "Docker bind mount target must be absolute",
    ),
    "a bind mount path with a comma": (
        lambda tmp: DockerBindMount("/data,x", "/data"),
        "Docker bind mount paths cannot contain commas",
    ),
    "a file that is not JSON, JSONL or Python": (
        lambda tmp: Taskset.from_file(write(tmp / "tasks.yaml", "- env: lab\n")),
        r"unsupported taskset source: .*tasks\.yaml",
    ),
    "a JSON document that is a scalar": (
        lambda tmp: Taskset.from_file(write(tmp / "tasks.json", "42")),
        r".*tasks\.json: expected a JSON object, list, or JSONL file",
    ),
    "an entry that is not an object": (
        lambda tmp: Taskset.from_file(write(tmp / "tasks.json", '[{"env": "lab", "id": "a"}, 3]')),
        r".*tasks\.json: each task entry must be an object",
    ),
    "a JSONL line that is not an object": (
        lambda tmp: Taskset.from_file(write(tmp / "tasks.jsonl", '"add"\n')),
        r".*tasks\.jsonl: each task entry must be an object",
    ),
    "duplicate slugs in a file": (
        lambda tmp: Taskset.from_file(
            write(tmp / "tasks.json", '[{"env": "lab", "id": "a"}, {"env": "other", "id": "a"}]')
        ),
        "duplicate task slugs: a",
    ),
    "writing to an unsupported format": (
        lambda tmp: Taskset("demo", [Task(env="lab", id="a")]).to_file(tmp / "tasks.csv"),
        r"unsupported taskset export format: \.csv; use \.json or \.jsonl",
    ),
}


@pytest.mark.parametrize(("build", "message"), INVALID.values(), ids=INVALID.keys())
def test_a_malformed_row_or_file_is_rejected_with_its_reason(
    build: Callable[[Path], Any], message: str, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match=message):
        build(tmp_path)


def test_a_platform_compose_record_round_trips_without_local_paths() -> None:
    record = {
        "compose": {
            "document": {"services": {"main": {"image": "lab:1"}}, "networks": {}},
            "root": {"compose_path": "recipe/compose.yaml"},
        }
    }

    config = RuntimeConfig.model_validate(record)

    assert config.model_dump(mode="json", exclude_unset=True) == snapshot(
        {
            "compose": {
                "document": {
                    "services": {
                        "main": {
                            "image": "lab:1",
                            "environment": {},
                            "expose": [],
                            "ports": [],
                            "volumes": [],
                        }
                    },
                    "networks": {},
                },
                "root": {"compose_path": "recipe/compose.yaml"},
            }
        }
    )
    assert (
        RuntimeConfig.model_validate(config.model_dump(mode="json", exclude_unset=True)) == config
    )


def test_a_local_compose_project_serializes_as_its_document_for_the_platform(
    tmp_path: Path,
) -> None:
    project = tmp_path / "artifact"
    compose = project / "recipe" / "compose.yaml"
    compose.parent.mkdir(parents=True)
    compose.write_text("services:\n  main:\n    image: lab:1\n    build: {context: ./main}\n")
    config = RuntimeConfig(compose=ComposeProject(document=compose, root=project))

    assert config.model_dump(mode="json", exclude_unset=True) == snapshot(
        {
            "compose": {
                "document": {
                    "services": {
                        "main": {
                            "image": "lab:1",
                            "build": {"context": "./main"},
                            "environment": {},
                            "expose": [],
                            "ports": [],
                            "volumes": [],
                        }
                    },
                    "networks": {},
                },
                "root": {"compose_path": "recipe/compose.yaml"},
            }
        }
    )


EXPORT = {
    "name": "Demo Set",
    "tasks": [
        {"env": "lab", "scenario": "add", "args": {"a": 2, "b": 3}, "name": "two-plus-three"},
        {
            "env": "actor",
            "scenario": "solve",
            "name": "solve",
            "columns": {"difficulty": "hard"},
            "runtime_config": {"image": "actor:1"},
            "verifier": {"env": "judge", "id": "verify"},
        },
        "not a record",
    ],
}
TASKSET_ID = str(uuid.UUID(int=7))
FETCHES: dict[str, tuple[str, list[Any], tuple[str, str | None, list[str]], list[str]]] = {
    "a name resolves to its taskset, then its rows": (
        "Demo Set",
        [("by-name", {"taskset_id": TASKSET_ID, "name": "Demo Set"}), ("export", EXPORT)],
        ("Demo Set", TASKSET_ID, ["two-plus-three", "solve"]),
        ["/v2/tasksets/by-name/Demo Set", f"/v2/tasksets/{TASKSET_ID}/export"],
    ),
    "an id is fetched directly": (
        TASKSET_ID,
        [("export", EXPORT)],
        ("Demo Set", TASKSET_ID, ["two-plus-three", "solve"]),
        [f"/v2/tasksets/{TASKSET_ID}/export"],
    ),
    "a taskset whose export is gone is empty": (
        TASKSET_ID,
        [("export", None)],
        (TASKSET_ID, TASKSET_ID, []),
        [f"/v2/tasksets/{TASKSET_ID}/export"],
    ),
}


@pytest.mark.parametrize(
    ("name", "replies", "fetched", "paths"), FETCHES.values(), ids=FETCHES.keys()
)
def test_a_platform_taskset_is_fetched_by_name_or_id(
    name: str,
    replies: list[tuple[str, Any]],
    fetched: tuple[str, str | None, list[str]],
    paths: list[str],
    services: FakeServices,
    hud_env: HudEnv,
) -> None:
    hud_env.set(HUD_API_KEY="k")
    for endpoint, body in replies:
        path = (
            "/v2/tasksets/by-name/{name}" if endpoint == "by-name" else "/v2/tasksets/{id}/export"
        )
        if body is None:
            services.route("api", "GET", path, json={"detail": "not found"}, status=404)
        else:
            services.route("api", "GET", path, json=body)

    taskset = Taskset.from_api(name)

    assert (taskset.name, taskset.taskset_id, [row.slug for row in taskset]) == fetched
    assert [request.path for request in services.requests()] == paths
    assert {request.bearer for request in services.requests()} == {"k"}


def test_platform_export_records_become_portable_rows(
    services: FakeServices, hud_env: HudEnv
) -> None:
    hud_env.set(HUD_API_KEY="k")
    services.route("api", "GET", "/v2/tasksets/{id}/export", json=EXPORT)

    taskset = Taskset.from_api(TASKSET_ID)

    assert list(taskset) == [
        Task(env="lab", id="add", args={"a": 2, "b": 3}, slug="two-plus-three"),
        Task(
            env="actor",
            id="solve",
            slug="solve",
            columns={"difficulty": "hard"},
            runtime_config=RuntimeConfig(image="actor:1"),
            verifier=Task(env="judge", id="verify"),
        ),
    ]


def test_an_unknown_platform_taskset_name_is_an_error(
    services: FakeServices, hud_env: HudEnv
) -> None:
    hud_env.set(HUD_API_KEY="k")
    services.route("api", "GET", "/v2/tasksets/by-name/{name}", json={}, status=404)

    with pytest.raises(ValueError, match="taskset not found: missing"):
        Taskset.from_api("missing")
