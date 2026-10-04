"""The control channel's ``files.*`` calls: what a served workspace with ``track_files=True``
reports about its files: what it tracks, the deliverables it captures, and what never
leaves the workspace."""

from __future__ import annotations

import base64
import hashlib
from typing import TYPE_CHECKING

from hud.environment import Environment
from tests.conftest import write_files
from tests.environment.conftest import session
from tests.serving import serving

if TYPE_CHECKING:
    from pathlib import Path

#: Larger than the per-file capture limit (5 MiB).
BIG_PDF = 5 * 1024 * 1024 + 1


def tracked(root: Path) -> Environment:
    env = Environment("tracked")
    env.workspace(root, track_files=True)
    return env


async def test_a_snapshot_lists_the_files_it_tracks_leaving_out_what_is_not_the_work(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    write_files(
        root,
        {
            "a.txt": "line1\n",
            "pkg/data.txt": "package data\n",
            "pkg/.gitignore": "data.txt\n",
            ".gitignore": "ignored.txt\n",
            "ignored.txt": "ignored\n",
            "node_modules/dep.js": "module.exports = 1;\n",
            ".tmp/go-build123/_pkg_.a": "compiled\n",
            ".git/config": "[core]\n",
            ".gitmodules": "[submodule]\n",
            ".env": "API_KEY=secret\n",
        },
    )

    env = tracked(root)
    async with (
        serving(env) as runtime,
        session(runtime.url) as peer,
    ):
        snapshot = await peer.call("files.snapshot")

    files = {entry["path"]: entry["content_hash"] for entry in snapshot["result"]["files"]}
    assert sorted(files) == [".gitignore", "a.txt", "pkg/.gitignore", "pkg/data.txt"]
    assert files["a.txt"] == hashlib.sha256(b"line1\n").hexdigest()


async def test_secrets_never_leave_in_a_diff(tmp_path: Path) -> None:
    root = tmp_path / "root"
    write_files(
        root,
        {".env": "API_KEY=supersecretvalue\n", ".gitmodules": "url = https://token@x/p.git\n"},
    )

    env = tracked(root)
    async with (
        serving(env) as runtime,
        session(runtime.url) as peer,
    ):
        await peer.call("files.snapshot")
        write_files(
            root,
            {
                ".env": "API_KEY=supersecretvalue\nDB_PASSWORD=hunter2\n",
                ".gitmodules": "url = https://new-token@x/p.git\n",
                ".git/config": "url = https://new-token@x/repo.git\n",
            },
        )
        edited = await peer.call("files.diff")

    assert {patch["path"] for patch in edited["result"]["patches"]} == {".env", ".gitmodules"}
    for secret in ("supersecretvalue", "hunter2", "new-token"):
        assert secret not in str(edited)


async def test_a_flush_diffs_the_deliverables_and_captures_the_documents_under_the_limit(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    write_files(root, {".env.xlsx": "API_KEY=before"})

    env = tracked(root)
    async with (
        serving(env) as runtime,
        session(runtime.url) as peer,
    ):
        await peer.call("files.snapshot")
        write_files(
            root,
            {
                "analysis.py": "print('tracked by diff')\n",
                "deliverable.xlsx": "workbook bytes",
                "report.html": "<html><body>report</body></html>\n",
                ".env.xlsx": "API_KEY=after",
                "large.pdf": "%" * BIG_PDF,
            },
        )
        flushed = (await peer.call("files.flush"))["result"]

    assert {p["path"]: p["status"] for p in flushed["diff"]["patches"]} == {
        "analysis.py": "added",
        "deliverable.xlsx": "added",
        "report.html": "added",
        ".env.xlsx": "modified",
        "large.pdf": "added",
    }
    capture = flushed["capture"]
    captured = {f["path"]: base64.b64decode(f["file"]["data"]) for f in capture["files"]}
    assert captured == {
        "deliverable.xlsx": b"workbook bytes",
        "report.html": b"<html><body>report</body></html>\n",
    }
    assert (capture["files_skipped"], capture["truncated"]) == (1, True)
    assert "API_KEY=after" not in str(flushed)
