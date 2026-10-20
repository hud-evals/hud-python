"""``filetracking/1``: what a served workspace with ``track_files=True`` reports about its
files: what it tracks, what changed since the last baseline, the deliverables it captures,
and what never leaves the workspace."""

from __future__ import annotations

import base64
import hashlib
from typing import TYPE_CHECKING

from hud.environment import Environment
from tests.conftest import write_files
from tests.environment.conftest import wire
from tests.fixtures.environments import served

if TYPE_CHECKING:
    from pathlib import Path

#: Larger than the per-file diff limit (1 MiB).
BIG_TEXT = 1024 * 1024 + 1
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

    async with served(tracked(root)) as client, wire(client.binding("filetracking").url) as peer:
        snapshot = await peer.call("snapshot")

    files = {entry["path"]: entry["content_hash"] for entry in snapshot["result"]["files"]}
    assert sorted(files) == [".gitignore", "a.txt", "pkg/.gitignore", "pkg/data.txt"]
    assert files["a.txt"] == hashlib.sha256(b"line1\n").hexdigest()


async def test_a_diff_reports_what_changed_since_the_last_one(tmp_path: Path) -> None:
    root = tmp_path / "root"
    write_files(root, {"edited.txt": "line1\nline2\n", "removed.txt": "gone\n"})

    async with served(tracked(root)) as client, wire(client.binding("filetracking").url) as peer:
        await peer.call("snapshot")
        before = await peer.call("diff")
        write_files(root, {"edited.txt": "line1\nCHANGED\n", "added.txt": "hello\n"})
        (root / "removed.txt").unlink()
        edited = await peer.call("diff")
        again = await peer.call("diff")

    patches = {patch["path"]: patch for patch in edited["result"]["patches"]}
    assert (before["result"]["patches"], again["result"]["patches"]) == ([], [])
    assert {path: patch["status"] for path, patch in patches.items()} == {
        "edited.txt": "modified",
        "added.txt": "added",
        "removed.txt": "deleted",
    }
    assert "+CHANGED" in patches["edited.txt"]["patch"]


async def test_secrets_never_leave_in_a_diff(tmp_path: Path) -> None:
    root = tmp_path / "root"
    write_files(
        root,
        {".env": "API_KEY=supersecretvalue\n", ".gitmodules": "url = https://token@x/p.git\n"},
    )

    async with served(tracked(root)) as client, wire(client.binding("filetracking").url) as peer:
        await peer.call("snapshot")
        write_files(
            root,
            {
                ".env": "API_KEY=supersecretvalue\nDB_PASSWORD=hunter2\n",
                ".gitmodules": "url = https://new-token@x/p.git\n",
                ".git/config": "url = https://new-token@x/repo.git\n",
            },
        )
        edited = await peer.call("diff")

    assert {patch["path"] for patch in edited["result"]["patches"]} == {".env", ".gitmodules"}
    for secret in ("supersecretvalue", "hunter2", "new-token"):
        assert secret not in str(edited)


async def test_a_file_over_the_diff_limit_is_reported_without_its_contents(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    write_files(root, {"big.txt": "a" * BIG_TEXT})

    async with served(tracked(root)) as client, wire(client.binding("filetracking").url) as peer:
        await peer.call("snapshot")
        write_files(root, {"big.txt": "b" * (BIG_TEXT + 1)})
        edited = await peer.call("diff")

    ((path, status, patch),) = [
        (p["path"], p["status"], p["patch"]) for p in edited["result"]["patches"]
    ]
    assert (path, status) == ("big.txt", "modified")
    assert len(patch) < 200


async def test_setup_and_advance_each_move_the_baseline_later_diffs_start_from(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    write_files(root, {"src.py": "x = 1\n"})

    async with served(tracked(root)) as client, wire(client.binding("filetracking").url) as peer:
        await peer.call("snapshot")
        write_files(root, {"generated.txt": "setup\n", "src.py": "x = 2\n"})
        setup = await peer.call("setup")
        after_setup = await peer.call("diff")
        write_files(root, {"legacy.txt": "legacy\n"})
        await peer.call("advance")
        after_advance = await peer.call("diff")

    assert {p["path"]: p["status"] for p in setup["result"]["patches"]} == {
        "generated.txt": "added",
        "src.py": "modified",
    }
    assert (after_setup["result"]["patches"], after_advance["result"]["patches"]) == ([], [])


async def test_a_flush_diffs_the_deliverables_and_captures_the_documents_under_the_limit(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    write_files(root, {".env.xlsx": "API_KEY=before"})

    async with served(tracked(root)) as client, wire(client.binding("filetracking").url) as peer:
        await peer.call("snapshot")
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
        flushed = (await peer.call("flush"))["result"]

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


async def test_an_unknown_request_is_refused_and_tracking_goes_on(tmp_path: Path) -> None:
    root = tmp_path / "root"
    write_files(root, {"a.txt": "one\n"})

    async with served(tracked(root)) as client, wire(client.binding("filetracking").url) as peer:
        refused = await peer.call("rewind")
        snapshot = await peer.call("snapshot")

    assert refused["error"]["code"] == -32000
    assert [entry["path"] for entry in snapshot["result"]["files"]] == ["a.txt"]
