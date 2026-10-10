"""``filetracking/1``: what a served workspace reports about its files, reply by reply.

A workspace with ``track_files=True`` publishes a ``filetracking`` capability.
These scenarios serve one over a fixture tree holding build output, VCS
metadata, ignored files and credentials, edit the tree the way an agent would,
and compare each reply on the wire with a golden copy (timestamps normalized).
"""

from __future__ import annotations

import base64
import hashlib
import re
from typing import TYPE_CHECKING, Any

import pytest
from inline_snapshot import snapshot

from hud.environment import Environment, Workspace
from tests.harness import served

from .conftest import wire

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


def normalized(value: Any) -> Any:
    """Replace clocks and mtimes, which change between runs, and content hashes."""
    if isinstance(value, dict):
        return {
            key: "<clock>"
            if key in {"snapshot_timestamp", "scan_duration_ms"}
            else normalized(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [normalized(item) for item in value]
    if isinstance(value, str):
        value = re.sub(r"^(redacted|overlimit):(\d+):\d+$", r"\1:\2:<mtime>", value)
        value = re.sub(r"^[0-9a-f]{64}$", "<sha256>", value)
        if len(value) > 200:
            return f"<{len(value)} chars>"
    return value


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
        advanced = await tracking.call("advance")
        after_advance = await tracking.call("diff")
        write(root, DELIVERABLES)
        flushed = await tracking.call("flush")

    assert normalized(initial) == snapshot(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "files": [
                    {"path": ".gitignore", "size": 12, "content_hash": "<sha256>"},
                    {"path": "a.txt", "size": 12, "content_hash": "<sha256>"},
                    {
                        "path": "big.txt",
                        "size": 1048577,
                        "content_hash": "overlimit:1048577:<mtime>",
                    },
                    {"path": "data.txt", "size": 10, "content_hash": "<sha256>"},
                    {"path": "go-build.go", "size": 13, "content_hash": "<sha256>"},
                    {"path": "keep.txt", "size": 5, "content_hash": "<sha256>"},
                    {"path": "pkg/.gitignore", "size": 9, "content_hash": "<sha256>"},
                    {"path": "pkg/data.txt", "size": 13, "content_hash": "<sha256>"},
                    {"path": "setup.xlsx", "size": 14, "content_hash": "<sha256>"},
                    {"path": "src.py", "size": 6, "content_hash": "<sha256>"},
                ],
                "files_scanned": 10,
            },
        }
    )
    assert normalized([unchanged, repeated, after_setup, after_advance]) == snapshot(
        [
            {
                "jsonrpc": "2.0",
                "id": 2,
                "result": {
                    "snapshot_timestamp": "<clock>",
                    "scan_duration_ms": "<clock>",
                    "files_scanned": 13,
                    "files_changed": 0,
                    "patches": [],
                },
            },
            {
                "jsonrpc": "2.0",
                "id": 4,
                "result": {
                    "snapshot_timestamp": "<clock>",
                    "scan_duration_ms": "<clock>",
                    "files_scanned": 13,
                    "files_changed": 0,
                    "patches": [],
                },
            },
            {
                "jsonrpc": "2.0",
                "id": 6,
                "result": {
                    "snapshot_timestamp": "<clock>",
                    "scan_duration_ms": "<clock>",
                    "files_scanned": 14,
                    "files_changed": 0,
                    "patches": [],
                },
            },
            {
                "jsonrpc": "2.0",
                "id": 8,
                "result": {
                    "snapshot_timestamp": "<clock>",
                    "scan_duration_ms": "<clock>",
                    "files_scanned": 15,
                    "files_changed": 0,
                    "patches": [],
                },
            },
        ]
    )
    assert normalized(edited) == snapshot(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "result": {
                "snapshot_timestamp": "<clock>",
                "scan_duration_ms": "<clock>",
                "files_scanned": 13,
                "files_changed": 6,
                "patches": [
                    {
                        "path": "keep.txt",
                        "status": "deleted",
                        "patch": """\
--- a/keep.txt
+++ b/keep.txt
@@ -1 +0,0 @@
-keep\
""",
                        "size_before": 5,
                        "size_after": 0,
                        "content_hash_before": "<sha256>",
                        "content_hash_after": None,
                    },
                    {
                        "path": "new.txt",
                        "status": "added",
                        "patch": """\
--- a/new.txt
+++ b/new.txt
@@ -0,0 +1 @@
+hello\
""",
                        "size_before": 0,
                        "size_after": 6,
                        "content_hash_before": None,
                        "content_hash_after": "<sha256>",
                    },
                    {
                        "path": "a.txt",
                        "status": "modified",
                        "patch": """\
--- a/a.txt
+++ b/a.txt
@@ -1,2 +1,2 @@
 line1
-line2
+CHANGED\
""",
                        "size_before": 12,
                        "size_after": 14,
                        "content_hash_before": "<sha256>",
                        "content_hash_after": "<sha256>",
                    },
                    {
                        "path": ".env",
                        "status": "modified",
                        "patch": "Secret file changed (content redacted): .env\n",
                        "size_before": 25,
                        "size_after": 45,
                        "content_hash_before": "redacted:25:<mtime>",
                        "content_hash_after": "redacted:45:<mtime>",
                    },
                    {
                        "path": ".gitmodules",
                        "status": "modified",
                        "patch": "Secret file changed (content redacted): .gitmodules\n",
                        "size_before": 66,
                        "size_after": 70,
                        "content_hash_before": "redacted:66:<mtime>",
                        "content_hash_after": "redacted:70:<mtime>",
                    },
                    {
                        "path": "big.txt",
                        "status": "modified",
                        "patch": "File too large to diff (1048577 -> 1048578 bytes): big.txt\n",
                        "size_before": 1048577,
                        "size_after": 1048578,
                        "content_hash_before": "overlimit:1048577:<mtime>",
                        "content_hash_after": "overlimit:1048578:<mtime>",
                    },
                ],
            },
        }
    )
    assert normalized(setup) == snapshot(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "result": {
                "snapshot_timestamp": "<clock>",
                "scan_duration_ms": "<clock>",
                "files_scanned": 14,
                "files_changed": 2,
                "patches": [
                    {
                        "path": "generated.txt",
                        "status": "added",
                        "patch": """\
--- a/generated.txt
+++ b/generated.txt
@@ -0,0 +1 @@
+setup\
""",
                        "size_before": 0,
                        "size_after": 6,
                        "content_hash_before": None,
                        "content_hash_after": "<sha256>",
                    },
                    {
                        "path": "src.py",
                        "status": "modified",
                        "patch": """\
--- a/src.py
+++ b/src.py
@@ -1 +1 @@
-x = 1
+x = 2  # setup\
""",
                        "size_before": 6,
                        "size_after": 15,
                        "content_hash_before": "<sha256>",
                        "content_hash_after": "<sha256>",
                    },
                ],
            },
        }
    )
    assert advanced == snapshot({"jsonrpc": "2.0", "id": 7, "result": {"advanced": True}})
    assert normalized(flushed) == snapshot(
        {
            "jsonrpc": "2.0",
            "id": 9,
            "result": {
                "diff": {
                    "snapshot_timestamp": "<clock>",
                    "scan_duration_ms": "<clock>",
                    "files_scanned": 19,
                    "files_changed": 5,
                    "patches": [
                        {
                            "path": "deliverable.xlsx",
                            "status": "added",
                            "patch": """\
--- a/deliverable.xlsx
+++ b/deliverable.xlsx
@@ -0,0 +1 @@
+workbook bytes\
""",
                            "size_before": 0,
                            "size_after": 14,
                            "content_hash_before": None,
                            "content_hash_after": "<sha256>",
                        },
                        {
                            "path": ".env.xlsx",
                            "status": "modified",
                            "patch": "Secret file changed (content redacted): .env.xlsx\n",
                            "size_before": 20,
                            "size_after": 13,
                            "content_hash_before": "redacted:20:<mtime>",
                            "content_hash_after": "redacted:13:<mtime>",
                        },
                        {
                            "path": "analysis.py",
                            "status": "added",
                            "patch": """\
--- a/analysis.py
+++ b/analysis.py
@@ -0,0 +1 @@
+print('tracked by diff')\
""",
                            "size_before": 0,
                            "size_after": 25,
                            "content_hash_before": None,
                            "content_hash_after": "<sha256>",
                        },
                        {
                            "path": "report.html",
                            "status": "added",
                            "patch": """\
--- a/report.html
+++ b/report.html
@@ -0,0 +1 @@
+<html><body>report</body></html>\
""",
                            "size_before": 0,
                            "size_after": 33,
                            "content_hash_before": None,
                            "content_hash_after": "<sha256>",
                        },
                        {
                            "path": "large.pdf",
                            "status": "added",
                            "patch": "File too large to diff (0 -> 5242881 bytes): large.pdf\n",
                            "size_before": 0,
                            "size_after": 5242881,
                            "content_hash_before": None,
                            "content_hash_after": "overlimit:5242881:<mtime>",
                        },
                    ],
                },
                "capture": {
                    "snapshot_timestamp": "<clock>",
                    "scan_duration_ms": "<clock>",
                    "files_scanned": 19,
                    "files_changed": 5,
                    "files_eligible": 3,
                    "files_captured": 2,
                    "files_skipped": 1,
                    "files": [
                        {
                            "path": "deliverable.xlsx",
                            "size": 14,
                            "content_hash": "<sha256>",
                            "content_type": XLSX,
                            "file": {
                                "type": "file",
                                "data": "d29ya2Jvb2sgYnl0ZXM=",
                                "media_type": XLSX,
                            },
                            "status": "added",
                        },
                        {
                            "path": "report.html",
                            "size": 33,
                            "content_hash": "<sha256>",
                            "content_type": "text/html",
                            "file": {
                                "type": "file",
                                "data": "PGh0bWw+PGJvZHk+cmVwb3J0PC9ib2R5PjwvaHRtbD4K",
                                "media_type": "text/html",
                            },
                            "status": "added",
                        },
                    ],
                    "truncated": True,
                },
            },
        }
    )
    manifest = {entry["path"]: entry["content_hash"] for entry in initial["result"]["files"]}
    assert manifest["a.txt"] == hashlib.sha256(b"line1\nline2\n").hexdigest()
    captured = {
        file["path"]: base64.b64decode(file["file"]["data"])
        for file in flushed["result"]["capture"]["files"]
    }
    assert captured == {
        "deliverable.xlsx": b"workbook bytes",
        "report.html": b"<html><body>report</body></html>\n",
    }


async def test_a_bad_request_gets_an_error_and_the_connection_keeps_serving(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    write(root, {"a.txt": "one\n"})
    env = Environment("tracked")
    env.workspace(root, track_files=True)

    async with served(env) as client, wire(client.binding("filetracking").url) as tracking:
        refused = await tracking.call("rewind")
        write(root, {"a.txt": "one\ntwo\n"})
        diff = await tracking.call("diff")

    assert refused == snapshot(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": -32000, "message": "unknown filetracking method: 'rewind'"},
        }
    )
    assert [patch["path"] for patch in diff["result"]["patches"]] == ["a.txt"]


@pytest.mark.parametrize("track_files", [True, False])
def test_the_file_tracking_capability_needs_a_started_tracking_workspace(
    tmp_path: Path, track_files: bool
) -> None:
    workspace = Workspace(tmp_path, track_files=track_files)

    with pytest.raises(RuntimeError, match=r"file tracking not started"):
        workspace.file_tracking_capability()
