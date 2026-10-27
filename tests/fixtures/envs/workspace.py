"""A file to fix in a real workspace: a reward of 1 proves the agent's own tools edited it."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from hud import Environment

ROOT = Path(tempfile.mkdtemp(prefix="hud-fixture-workspace-"))

env = Environment("workspace")
env.workspace(ROOT, guest_path=str(ROOT))


@env.shutdown
async def remove_workspace() -> None:
    shutil.rmtree(ROOT, ignore_errors=True)


@env.template()
async def fix_the_file():
    notes = ROOT / "notes.txt"
    notes.write_text("one\ntwo\nthree\n")
    yield (
        f"The file {notes} has a line that reads 'two'. Change that line to '2' "
        "and leave the other lines as they are."
    )
    yield 1.0 if notes.read_text() == "one\n2\nthree\n" else 0.0


tasks = [fix_the_file()]
