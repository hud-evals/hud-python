"""File tracking in a trace: a rollout on a tracked workspace exports what the environment
set up and what the agent changed as ``filetracking`` spans."""

from __future__ import annotations

import asyncio
import itertools
from typing import TYPE_CHECKING, Any

import pytest

from hud import Environment
from hud.agents.base import Agent
from hud.capabilities import Capability
from hud.eval import LocalRuntime, Task, rollout
from tests.conftest import spans

if TYPE_CHECKING:
    from pathlib import Path

    from hud.eval import Run
    from tests.conftest import HudEnv

SCHEMA = "hud.filetracking.v1"


@pytest.fixture(autouse=True)
def tracking_often(hud_env: HudEnv, tmp_path: Path) -> None:
    hud_env.set(
        HUD_TELEMETRY_ENABLED="1",
        HUD_TELEMETRY_LOCAL_DIR=str(tmp_path / "spans"),
        HUD_FILE_TRACKING_INTERVAL="0.05",
    )


class Writer(Agent):
    """Writes ``files`` into the workspace, then waits long enough to be sampled."""

    def __init__(self, root: Path, files: dict[str, bytes]) -> None:
        super().__init__()
        self.root = root
        self.files = files

    async def __call__(self, run: Run) -> None:
        for name, data in self.files.items():
            (self.root / name).write_bytes(data)
        await asyncio.sleep(0.2)
        run.trace.content = "done"


def tracked(root: Path, *, track_files: bool = True, seed: int = 0) -> Environment:
    """A workspace whose task writes ``setup.txt``, and ``seed`` more files, as it starts."""
    env = Environment("tracked")
    env.workspace(root, track_files=track_files)

    @env.template()
    async def write():
        (root / "setup.txt").write_text("from setup\n")
        for index in range(seed):
            (root / f"seed-{index:04d}-{'x' * 40}.txt").write_text("seed\n")
        yield "write your files"
        yield 1.0

    return env


def exported(run: Run) -> list[tuple[str, dict[str, Any]]]:
    """The run's file-tracking spans, in order, as (name, payload)."""
    return [
        (span["name"], span["attributes"]["hud.payload"])
        for span in spans(run.trace_id)
        if span["attributes"].get("hud.schema") == SCHEMA
    ]


async def test_a_tracked_rollout_exports_its_setup_a_snapshot_its_diffs_and_deliverables(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ws"
    files = {"notes.md": b"# notes\n", "report.pdf": b"%PDF-1.4 report"}

    run = await rollout(
        Task(env="tracked", id="write"), Writer(root, files), runtime=LocalRuntime(tracked(root))
    )

    tracking = exported(run)
    payloads = dict(tracking)
    assert [name for name, _ in itertools.groupby(name for name, _ in tracking)] == [
        "filetracking.setup",
        "filetracking.snapshot",
        "filetracking.diff",
        "filetracking.capture",
    ]
    assert [p["path"] for p in payloads["filetracking.setup"]["patches"]] == ["setup.txt"]
    assert sorted(p["path"] for p in payloads["filetracking.diff"]["patches"]) == [
        "notes.md",
        "report.pdf",
    ]
    assert [f["path"] for f in payloads["filetracking.capture"]["files"]] == ["report.pdf"]


async def test_a_large_workspace_snapshot_arrives_whole(tmp_path: Path) -> None:
    root = tmp_path / "ws"

    run = await rollout(
        Task(env="tracked", id="write"),
        Writer(root, {}),
        runtime=LocalRuntime(tracked(root, seed=1500)),
    )

    payloads = dict(exported(run))
    assert len(payloads["filetracking.snapshot"]["files"]) == 1501
    assert len(payloads["filetracking.setup"]["patches"]) == 1501


async def test_an_untracked_workspace_exports_no_file_tracking(tmp_path: Path) -> None:
    root = tmp_path / "ws"

    run = await rollout(
        Task(env="tracked", id="write"),
        Writer(root, {"notes.md": b"# notes\n"}),
        runtime=LocalRuntime(tracked(root, track_files=False)),
    )

    assert (run.reward, exported(run)) == (1.0, [])


async def test_a_tracker_that_cannot_be_reached_exports_nothing_and_the_run_goes_on(
    tmp_path: Path,
) -> None:
    env = Environment("tracked")
    env.add_capability(
        Capability(name="filetracking", protocol="filetracking/1", url="tcp://127.0.0.1:9")
    )

    @env.template()
    async def write():
        yield "write your files"
        yield 1.0

    (run,) = (await write().run(Writer(tmp_path, {"notes.md": b"# notes\n"}))).runs

    assert (run.reward, exported(run)) == (1.0, [])
