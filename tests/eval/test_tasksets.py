"""Tasksets as collections: the rows a source, a data file or the platform holds, how
each row is keyed, and what a sync would change."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from hud.eval import ComposeProject, RuntimeConfig, Task, Taskset

if TYPE_CHECKING:
    from pathlib import Path

    from tests.support import FakeServices, HudEnv


@pytest.mark.parametrize("suffix", [".json", ".jsonl"])
def test_a_taskset_written_to_a_data_file_reads_back_row_for_row(
    suffix: str, tmp_path: Path
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "compose.yaml").write_text("services:\n  main:\n    image: lab\n")
    rows = [
        Task(env="lab", id="add", args={"a": 2, "b": 3}),
        Task.model_validate(
            {
                "env": "lab",
                "id": "add",
                "args": {"a": 1, "b": 1},
                "slug": "one-plus-one",
                "validation": [{"name": "answer", "arguments": {"expected": 2}}],
                "agent_config": {"timeout_seconds": 30.0},
                "columns": {"difficulty": "easy"},
                "runtime_config": {
                    "image": "registry.test/trainer:1",
                    "resources": {
                        "cpu": 8,
                        "gpu": {"type": ["H100", "A100"], "count": 2},
                        "tpu": {"type": "v5e", "topology": "2x4"},
                    },
                    "limits": {"startup_timeout_s": 600},
                },
                "verifier": {"env": "judge", "id": "verify", "args": {"expected": "2"}},
            }
        ),
        Task(
            env="compose",
            id="solve",
            runtime_config=RuntimeConfig(
                compose=ComposeProject(document=project / "compose.yaml", service_access=True)
            ),
        ),
    ]

    written = Taskset("demo", rows).to_file(tmp_path / "out" / f"tasks{suffix}")
    loaded = Taskset.from_file(written)

    assert loaded.name == "tasks"
    assert [row.model_dump() for row in loaded] == [row.model_dump() for row in rows]
    assert '"document": "../project/compose.yaml"' in written.read_text()


SOURCE = """
from hud import Environment, Taskset

env = Environment("lab")


@env.template()
async def solve(n: int):
    yield "go"
    yield 1.0


single = solve(n=1)
several = [solve(n=2), solve(n=3)]
collected = Taskset("inner", [solve(n=4)])
_private = solve(n=5)
"""


def test_a_python_source_holds_the_tasks_and_tasksets_it_defines(tmp_path: Path) -> None:
    (tmp_path / "tasks.py").write_text(SOURCE)

    loaded = Taskset.from_file(tmp_path / "tasks.py")

    assert loaded.name == "tasks"
    assert sorted(row.args["n"] for row in loaded) == [1, 2, 3, 4]


def test_a_directory_holds_the_tasks_of_every_source_in_it(tmp_path: Path) -> None:
    root = tmp_path / "suite"
    root.mkdir()
    (root / "first.py").write_text(SOURCE)
    (root / "second.py").write_text(SOURCE.replace("n=", "n=1"))
    (root / "conftest.py").write_text("raise RuntimeError('not a task source')\n")

    loaded = Taskset.from_file(root)

    assert loaded.name == "suite"
    assert len(loaded) == 8


@pytest.mark.parametrize("reference", ["SheetBench-50", "eeeeeeee-0000-4000-8000-000000000001"])
def test_a_platform_taskset_loads_by_name_or_id(
    reference: str, services: FakeServices, hud_env: HudEnv
) -> None:
    hud_env.set(HUD_API_KEY="k")
    taskset_id = "eeeeeeee-0000-4000-8000-000000000001"
    services.route(
        "api",
        "GET",
        "/v2/tasksets/by-name/SheetBench-50",
        json={"taskset_id": taskset_id, "name": "SheetBench-50"},
    )
    services.route(
        "api",
        "GET",
        f"/v2/tasksets/{taskset_id}/export",
        json={
            "name": "SheetBench-50",
            "tasks": [
                {"name": "sum-a", "env": "sheets", "scenario": "sum", "args": {"cell": "A1"}},
                {"name": "sum-b", "env": "sheets", "scenario": "sum", "columns": {"hard": True}},
            ],
        },
    )

    loaded = Taskset.from_api(reference)

    assert (loaded.name, loaded.taskset_id) == ("SheetBench-50", taskset_id)
    assert [(row.slug, row.env, row.id, row.args) for row in loaded] == [
        ("sum-a", "sheets", "sum", {"cell": "A1"}),
        ("sum-b", "sheets", "sum", {}),
    ]


def test_a_platform_taskset_that_does_not_exist_is_refused(
    services: FakeServices, hud_env: HudEnv
) -> None:
    hud_env.set(HUD_API_KEY="k")
    services.route("api", "GET", "/v2/tasksets/by-name/{name}", status=404, json={})

    with pytest.raises(ValueError, match="taskset not found: missing"):
        Taskset.from_api("missing")
