"""File tracking for a directory in a box, against a box that is this machine."""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Any, cast

import pytest

from hud.environment.box import BoxFileTracker

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from hud.runtime.core import Box

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the box runs POSIX sh")

# Long ago, as a file copied or moved in with its times kept would have it.
OLD = 1_000_000_000


class _LocalBox:
    """Runs each command on this machine, in the directory it asks for.

    ``COPYFILE_DISABLE`` keeps macOS's tar from archiving AppleDouble ``._`` entries that
    a Linux box's tar never writes.
    """

    def exec_argv(
        self,
        argv: Sequence[str],
        *,
        user: str | None = None,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        tty: bool = False,
    ) -> list[str]:
        return [
            "env",
            "COPYFILE_DISABLE=1",
            "sh",
            "-c",
            'cd "$1" && shift && exec "$@"',
            "sh",
            cwd or ".",
            *argv,
        ]


def _changes(diff: dict[str, Any]) -> dict[str, str]:
    return {patch["path"]: patch["status"] for patch in diff["patches"]}


@pytest.fixture
async def tracked(tmp_path: Path) -> Any:
    workdir = tmp_path / "app"
    (workdir / "src").mkdir(parents=True)
    (workdir / "src" / "main.py").write_text("print('one')\n")
    (workdir / "notes.txt").write_text("keep\n")
    (workdir / "node_modules").mkdir()
    (workdir / "node_modules" / "dep.js").write_text("ignored\n")
    tracker = BoxFileTracker(cast("Box", _LocalBox()), str(workdir))
    try:
        yield workdir, tracker
    finally:
        tracker.close()


async def test_a_box_file_tracker_follows_what_changes_in_the_box(tracked: Any) -> None:
    workdir, tracker = tracked

    assert (await tracker.call("setup"))["files_changed"] == 0
    manifest = await tracker.call("snapshot")
    assert [row["path"] for row in manifest["files"]] == ["notes.txt", "src/main.py"]

    (workdir / "src" / "main.py").write_text("print('two')\n")
    (workdir / "report.html").write_text("<p>done</p>\n")
    (workdir / "notes.txt").unlink()
    (workdir / "node_modules" / "other.js").write_text("ignored\n")
    moved = workdir / "moved.txt"
    moved.write_text("from elsewhere\n")
    os.utime(moved, (OLD, OLD))

    diff = await tracker.call("diff")

    assert _changes(diff) == {
        "src/main.py": "modified",
        "report.html": "added",
        "notes.txt": "deleted",
        "moved.txt": "added",
    }
    patch = next(item for item in diff["patches"] if item["path"] == "src/main.py")
    assert "-print('one')" in patch["patch"]
    assert "+print('two')" in patch["patch"]

    # The same size again, at once: caught though its size and time may match the last read.
    (workdir / "src" / "main.py").write_text("print('six')\n")
    assert _changes(await tracker.call("diff")) == {"src/main.py": "modified"}
    assert _changes(await tracker.call("diff")) == {}

    flushed = await tracker.call("flush")
    assert flushed["diff"]["files_changed"] == 0
    assert [file["path"] for file in flushed["capture"]["files"]] == ["report.html"]


async def test_a_change_that_kept_an_older_time_is_reported_by_the_flush(tracked: Any) -> None:
    workdir, tracker = tracked
    await tracker.call("setup")

    replaced = workdir / "src" / "main.py"
    replaced.write_text("print('replaced')\n")
    os.utime(replaced, (OLD, OLD))
    live = _changes(await tracker.call("diff"))
    final = _changes((await tracker.call("flush"))["diff"])

    assert {**live, **final} == {"src/main.py": "modified"}
