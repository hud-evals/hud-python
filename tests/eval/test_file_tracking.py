"""File tracking: a rollout on a tracked workspace exports what the environment set
up and what the agent changed as ``filetracking`` spans in its trace."""

from __future__ import annotations

import asyncio
import itertools
from typing import TYPE_CHECKING, Any

import pytest

from hud import Environment
from hud.agents.base import Agent
from hud.capabilities import Capability
from hud.environment.file_tracker import FileTracker, serve_file_tracking
from hud.eval import LocalRuntime, Task, rollout
from tests.harness import spans

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from hud.eval import Run
    from tests.harness import HudEnv

SCHEMA = "hud.filetracking.v1"


class Writer(Agent):
    """Writes ``files`` into the workspace, then lingers so the tracker can sample it."""

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
    """A workspace env whose task writes ``setup.txt``, and ``seed`` more files, as it starts."""
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


def legacy(root: Path) -> Environment:
    """A workspace tracked by a server that predates setup diffs."""
    env = Environment("tracked")
    root.mkdir(parents=True, exist_ok=True)

    @env.initialize
    async def track() -> None:
        tracker = FileTracker(root)
        tracker.take_baseline()
        server = await serve_file_tracking(tracker)
        port = server.sockets[0].getsockname()[1]
        env.add_capability(
            Capability(
                name="filetracking", protocol="filetracking/1", url=f"tcp://127.0.0.1:{port}"
            )
        )

    @env.template()
    async def write():
        (root / "setup.txt").write_text("from setup\n")
        yield "write your files"
        yield 1.0

    return env


def unreachable(root: Path) -> Environment:
    env = Environment("tracked")
    env.add_capability(
        Capability(name="filetracking", protocol="filetracking/1", url="tcp://127.0.0.1:9")
    )

    @env.template()
    async def write():
        yield "write your files"
        yield 1.0

    del root
    return env


NOTES = {"notes.md": b"# notes\n", "report.pdf": b"%PDF-1.4 report"}
STREAMED = {
    "setup": ["setup.txt"],
    "snapshot": 1,
    "changed": ["notes.md", "report.pdf"],
    "captured": ["report.pdf"],
    "skipped": 0,
}
CASES: dict[str, tuple[Callable[[Path], Environment], dict[str, bytes], str, dict[str, Any]]] = {
    "a tracked workspace streams setup, a snapshot, diffs and its deliverables": (
        tracked,
        NOTES,
        "1",
        STREAMED,
    ),
    "a server without setup diffs starts from a snapshot": (
        legacy,
        NOTES,
        "1",
        {key: value for key, value in STREAMED.items() if key != "setup"},
    ),
    "a deliverable too large to capture is still reported": (
        tracked,
        {"huge.pdf": b"%PDF" + b"0" * (6 * 1024 * 1024)},
        "1",
        STREAMED | {"changed": ["huge.pdf"], "captured": [], "skipped": 1},
    ),
    "a large workspace snapshot arrives whole": (
        lambda root: tracked(root, seed=1500),
        {},
        "1",
        {"setup": 1501, "snapshot": 1501},
    ),
    "an untracked workspace exports nothing": (
        lambda root: tracked(root, track_files=False),
        NOTES,
        "1",
        {},
    ),
    "a tracker that cannot be reached exports nothing": (unreachable, NOTES, "1", {}),
    "nothing is exported with telemetry off": (tracked, NOTES, "0", {}),
}
ORDER = ["setup", "snapshot", "changed", "captured"]
SPAN_NAMES = {
    "setup": "filetracking.setup",
    "snapshot": "filetracking.snapshot",
    "changed": "filetracking.diff",
    "captured": "filetracking.capture",
}


@pytest.mark.parametrize(
    ("build", "files", "telemetry", "summary"), CASES.values(), ids=CASES.keys()
)
async def test_a_tracked_rollout_exports_what_changed_in_its_workspace(
    build: Callable[[Path], Environment],
    files: dict[str, bytes],
    telemetry: str,
    summary: dict[str, Any],
    hud_env: HudEnv,
    tmp_path: Path,
) -> None:
    root = tmp_path / "ws"
    hud_env.set(
        HUD_TELEMETRY_ENABLED=telemetry,
        HUD_TELEMETRY_LOCAL_DIR=str(tmp_path / "spans"),
        HUD_FILE_TRACKING_INTERVAL="0.05",
    )

    run = await rollout(
        Task(env="tracked", id="write"), Writer(root, files), runtime=LocalRuntime(build(root))
    )

    tracking = [
        span for span in spans(run.trace_id) if span["attributes"].get("hud.schema") == SCHEMA
    ]
    assert run.reward == 1.0
    assert [name for name, _ in itertools.groupby(span["name"] for span in tracking)] == [
        SPAN_NAMES[key] for key in ORDER if key in summary
    ]
    assert {span["attributes"]["hud.task_run_id"] for span in tracking} <= {run.trace_id}
    assert described({span["name"]: span["attributes"]["hud.payload"] for span in tracking}) == (
        summary
    )


def described(payloads: dict[str, Any]) -> dict[str, Any]:
    """What each kind of span reports, as a viewer shows it."""
    summary: dict[str, Any] = {}
    if setup := payloads.get("filetracking.setup"):
        paths = [patch["path"] for patch in setup["patches"]]
        summary["setup"] = paths if len(paths) == 1 else len(paths)
    if snapshot := payloads.get("filetracking.snapshot"):
        summary["snapshot"] = len(snapshot["files"])
    if diff := payloads.get("filetracking.diff"):
        summary["changed"] = sorted(patch["path"] for patch in diff["patches"])
    if capture := payloads.get("filetracking.capture"):
        summary["captured"] = sorted(file["path"] for file in capture.get("files", []))
        summary["skipped"] = capture["files_skipped"]
    return summary
