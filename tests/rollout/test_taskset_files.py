"""Taskset files: the portable row format a taskset is written to and read back from,
with every field a row can carry."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from pathlib import Path


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

    compose = json.loads(written.read_text())[-1]["runtime_config"]["compose"]
    assert (compose["document"], compose["root"]) == ("../project/compose.yaml", "../project")
    loaded = Taskset.from_file(written)
    assert loaded.name == "tasks"
    assert [row.model_dump() for row in loaded] == [row.model_dump() for row in rows]
