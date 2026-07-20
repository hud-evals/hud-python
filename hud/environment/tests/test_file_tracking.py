"""filetracking/1 wire roundtrip: serve a FileTracker, drive the wire directly."""

from __future__ import annotations

import asyncio
import base64
from typing import TYPE_CHECKING, Any

from hud.environment.file_tracker import FileTracker, serve_file_tracking
from hud.environment.utils import read_frame, send_frame

if TYPE_CHECKING:
    from pathlib import Path


async def _call(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    method: str,
) -> dict[str, Any]:
    await send_frame(
        writer,
        {"jsonrpc": "2.0", "id": 1, "method": method, "params": {}},
    )
    reply = await read_frame(reader)
    assert reply is not None
    assert "error" not in reply
    result = reply.get("result")
    assert isinstance(result, dict)
    return result


async def test_diff_snapshot_setup_roundtrip(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x\n")
    tracker = FileTracker(tmp_path)
    tracker.take_baseline()

    server = await serve_file_tracking(tracker)
    host, port = server.sockets[0].getsockname()[:2]
    reader, writer = await asyncio.open_connection(host, port)
    try:
        # Nothing changed yet.
        assert (await _call(reader, writer, "diff"))["files_changed"] == 0

        # An edit shows up as a diff on the next pull.
        (tmp_path / "a.txt").write_text("x\ny\n")
        diff = await _call(reader, writer, "diff")
        assert diff["files_changed"] == 1
        assert diff["patches"][0]["path"] == "a.txt"

        # diff() advanced the baseline, so a re-pull is empty.
        assert (await _call(reader, writer, "diff"))["files_changed"] == 0

        # snapshot() returns the full manifest.
        snapshot = await _call(reader, writer, "snapshot")
        assert any(entry["path"] == "a.txt" for entry in snapshot["files"])

        # setup() emits scenario changes and starts a clean agent-edit layer.
        (tmp_path / "b.txt").write_text("z\n")
        setup = await _call(reader, writer, "setup")
        assert setup["files_changed"] == 1
        assert setup["patches"][0]["path"] == "b.txt"
        assert (await _call(reader, writer, "diff"))["files_changed"] == 0

        # flush() returns the trailing diff plus bounded changed deliverables.
        (tmp_path / "report.xlsx").write_bytes(b"spreadsheet")
        flush = await _call(reader, writer, "flush")
        assert flush["diff"]["files_changed"] == 1
        capture = flush["capture"]
        assert capture["files_captured"] == 1
        assert capture["files"][0]["path"] == "report.xlsx"
        assert base64.b64decode(capture["files"][0]["file"]["data"]) == b"spreadsheet"
    finally:
        writer.close()
        await writer.wait_closed()
        server.close()
        await server.wait_closed()
