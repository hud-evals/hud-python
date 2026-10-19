"""``filetracking/1``: what a served workspace reports about its files, reply by reply.

A workspace with ``track_files=True`` publishes a ``filetracking`` capability.
These scenarios serve one over a fixture tree holding build output, VCS
metadata, ignored files and credentials, edit the tree the way an agent would,
and check what each reply reports: which files changed and how, and what never leaves.
A bad request is a case in ``test_error_replies.py``.
"""

from __future__ import annotations

import base64
import hashlib
from typing import TYPE_CHECKING, Any

from hud.environment import Environment
from tests.fixtures.environments import served
from tests.support import wire

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#: Larger than the per-file diff limit (1 MiB).
BIG_TEXT = 1024 * 1024 + 1
#: Larger than the per-file capture limit (5 MiB).
BIG_PDF = 5 * 1024 * 1024 + 1

TREE = {
    "a.txt": "line1\nline2\n",
    "keep.txt": "keep\n",
    "src.py": "x = 1\n",
    "go-build.go": "package main\n",
    "data.txt": "root data\n",
    "pkg/data.txt": "package data\n",
    "pkg/.gitignore": "data.txt\n",
    ".gitignore": "ignored.txt\n",
    "ignored.txt": "ignored\n",
    "node_modules/dep.js": "module.exports = 1;\n",
    ".tmp/go-build123/_pkg_.a": "compiled\n",
    ".git/config": '[remote "origin"]\nurl = https://token@example.com/repo.git\n',
    ".gitmodules": '[submodule "private"]\nurl = https://token@example.com/private.git\n',
    ".env": "API_KEY=supersecretvalue\n",
    ".env.xlsx": "API_KEY=before-setup",
    "setup.xlsx": "setup workbook",
    "big.txt": "a" * BIG_TEXT,
}

#: An agent's edits: modified, deleted, added, secret, excluded and oversized files.
EDITS = {
    "a.txt": "line1\nCHANGED\n",
    "keep.txt": None,
    "new.txt": "hello\n",
    ".env": "API_KEY=supersecretvalue\nDB_PASSWORD=hunter2\n",
    ".gitmodules": '[submodule "private"]\nurl = https://new-token@example.com/private.git\n',
    ".git/config": '[remote "origin"]\nurl = https://new-token@example.com/repo.git\n',
    "ignored.txt": "still ignored, longer\n",
    "node_modules/dep.js": "module.exports = 2; // edited\n",
    "big.txt": "b" * (BIG_TEXT + 1),
}

#: A rollout's deliverables, written after setup.
DELIVERABLES = {
    "analysis.py": "print('tracked by diff')\n",
    "deliverable.xlsx": "workbook bytes",
    "report.html": "<html><body>report</body></html>\n",
    ".env.xlsx": "API_KEY=after",
    "large.pdf": "%" * BIG_PDF,
}


def write(root: Path, files: Mapping[str, str | None]) -> None:
    for name, content in files.items():
        path = root / name
        if content is None:
            path.unlink()
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def changed(reply: dict[str, Any]) -> dict[str, str]:
    """A diff reply's changes as ``{path: status}``."""
    return {patch["path"]: patch["status"] for patch in reply["result"]["patches"]}


async def test_file_tracking_reports_setup_agent_edits_and_deliverables(tmp_path: Path) -> None:
    root = tmp_path / "root"
    write(root, TREE)
    env = Environment("tracked")
    env.workspace(root, track_files=True)

    async with served(env) as client, wire(client.binding("filetracking").url) as tracking:
        initial = await tracking.call("snapshot")
        unchanged = await tracking.call("diff")
        write(root, EDITS)
        edited = await tracking.call("diff")
        repeated = await tracking.call("diff")
        write(root, {"generated.txt": "setup\n", "src.py": "x = 2  # setup\n"})
        setup = await tracking.call("setup")
        after_setup = await tracking.call("diff")
        write(root, {"legacy.txt": "legacy\n"})
        await tracking.call("advance")
        after_advance = await tracking.call("diff")
        write(root, DELIVERABLES)
        flushed = await tracking.call("flush")

    # VCS metadata, ignored files, dependencies, build output and secrets are not tracked.
    manifest = {entry["path"]: entry["content_hash"] for entry in initial["result"]["files"]}
    assert sorted(manifest) == [
        ".gitignore",
        "a.txt",
        "big.txt",
        "data.txt",
        "go-build.go",
        "keep.txt",
        "pkg/.gitignore",
        "pkg/data.txt",
        "setup.xlsx",
        "src.py",
    ]
    assert manifest["a.txt"] == hashlib.sha256(b"line1\nline2\n").hexdigest()

    # Each diff starts where the last one, or setup, or advance, left off.
    for reply in (unchanged, repeated, after_setup, after_advance):
        assert changed(reply) == {}
    assert changed(edited) == {
        "keep.txt": "deleted",
        "new.txt": "added",
        "a.txt": "modified",
        ".env": "modified",
        ".gitmodules": "modified",
        "big.txt": "modified",
    }
    patches = {patch["path"]: patch["patch"] for patch in edited["result"]["patches"]}
    assert "+CHANGED" in patches["a.txt"]
    for secret in ("supersecretvalue", "hunter2", "new-token"):
        assert secret not in str(edited)
    assert len(patches["big.txt"]) < 200  # over the diff limit: reported, not diffed
    assert changed(setup) == {"generated.txt": "added", "src.py": "modified"}

    # Flush diffs the deliverables and captures the eligible ones under the size limit.
    diff, capture = flushed["result"]["diff"], flushed["result"]["capture"]
    assert changed({"result": diff}) == {
        "deliverable.xlsx": "added",
        ".env.xlsx": "modified",
        "analysis.py": "added",
        "report.html": "added",
        "large.pdf": "added",
    }
    captured = {file["path"]: base64.b64decode(file["file"]["data"]) for file in capture["files"]}
    assert captured == {
        "deliverable.xlsx": b"workbook bytes",
        "report.html": b"<html><body>report</body></html>\n",
    }
    assert capture["files"][0]["content_type"] == XLSX
    assert (capture["files_skipped"], capture["truncated"]) == (1, True)
    assert "API_KEY=after" not in str(flushed)
