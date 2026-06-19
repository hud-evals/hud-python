"""FileTracker: snapshot diffing, excludes, gitignore, and the secrets deny-list."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hud.environment.file_tracker import FileTracker

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_modified_file_produces_a_unified_diff(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("line1\nline2\n")
    tracker = FileTracker(tmp_path)
    tracker.take_baseline()

    (tmp_path / "a.txt").write_text("line1\nCHANGED\n")
    diff = tracker.take_snapshot()

    assert diff.files_changed == 1
    patch = diff.patches[0]
    assert patch.rel_path == "a.txt"
    assert patch.status == "modified"
    assert "CHANGED" in patch.patch


def test_added_and_deleted_files(tmp_path: Path) -> None:
    (tmp_path / "keep.txt").write_text("x\n")
    tracker = FileTracker(tmp_path)
    tracker.take_baseline()

    (tmp_path / "new.txt").write_text("hello\n")
    (tmp_path / "keep.txt").unlink()
    diff = tracker.take_snapshot()

    by_path = {p.rel_path: p for p in diff.patches}
    assert by_path["new.txt"].status == "added"
    assert by_path["new.txt"].size_before == 0
    assert by_path["keep.txt"].status == "deleted"


def test_no_changes_yields_empty_diff(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x\n")
    tracker = FileTracker(tmp_path)
    tracker.take_baseline()

    diff = tracker.take_snapshot()

    assert diff.files_changed == 0
    assert diff.patches == []


def test_exclude_patterns_skip_build_output(tmp_path: Path) -> None:
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "dep.js").write_text("module.exports = 1;\n")
    (tmp_path / "src.py").write_text("x = 1\n")
    tracker = FileTracker(tmp_path)
    tracker.take_baseline()

    manifest_paths = {entry["path"] for entry in tracker.current_manifest()}
    assert "src.py" in manifest_paths
    assert not any(p.startswith("node_modules/") for p in manifest_paths)


def test_gitignore_is_honored(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("ignored.txt\n")
    (tmp_path / "ignored.txt").write_text("secret-ish\n")
    (tmp_path / "tracked.txt").write_text("x\n")
    tracker = FileTracker(tmp_path)
    tracker.take_baseline()

    manifest_paths = {entry["path"] for entry in tracker.current_manifest()}
    assert "tracked.txt" in manifest_paths
    assert "ignored.txt" not in manifest_paths


def test_secret_files_are_tracked_but_content_is_never_emitted(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("API_KEY=supersecretvalue\n")
    tracker = FileTracker(tmp_path)
    tracker.take_baseline()

    (tmp_path / ".env").write_text("API_KEY=supersecretvalue\nDB_PASSWORD=hunter2\n")
    diff = tracker.take_snapshot()

    assert diff.files_changed == 1
    patch = diff.patches[0]
    assert patch.rel_path == ".env"
    assert patch.status == "modified"
    # The change is detected, but the content is redacted — never in the patch.
    assert "redacted" in patch.patch.lower()
    assert "supersecretvalue" not in patch.patch
    assert "hunter2" not in patch.patch


def test_per_file_diff_cap_emits_a_placeholder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(FileTracker, "_MAX_DIFF_FILE_BYTES", 4)
    (tmp_path / "big.txt").write_text("aaaaaaaa\n")
    tracker = FileTracker(tmp_path)
    tracker.take_baseline()

    (tmp_path / "big.txt").write_text("bbbbbbbb\n")
    diff = tracker.take_snapshot()

    assert diff.files_changed == 1
    assert "too large to diff" in diff.patches[0].patch


def test_budget_skipped_files_stay_pending_until_emitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "a.txt").write_text("a1\n")
    (tmp_path / "b.txt").write_text("b1\n")
    tracker = FileTracker(tmp_path)
    tracker.take_baseline()

    (tmp_path / "a.txt").write_text("a2\n")
    (tmp_path / "b.txt").write_text("b2\n")

    # Cap so small no patch fits: both changes are skipped this poll.
    monkeypatch.setattr(FileTracker, "_MAX_DIFF_BYTES", 1)
    first = tracker.take_snapshot()
    assert first.truncated
    assert first.patches == []
    assert set(first.skipped_paths) == {"a.txt", "b.txt"}

    # Next poll with headroom: the skipped changes must not be lost — the
    # baseline kept them pending, so they re-diff now.
    monkeypatch.setattr(FileTracker, "_MAX_DIFF_BYTES", 50 * 1024 * 1024)
    second = tracker.take_snapshot()
    by_path = {p.rel_path: p for p in second.patches}
    assert set(by_path) == {"a.txt", "b.txt"}
    assert "a2" in by_path["a.txt"].patch


def test_nested_gitignore_is_not_honored_or_leaked(tmp_path: Path) -> None:
    # Root has no .gitignore; a nested package does. With root-only honoring the
    # nested rule must neither take effect locally nor leak tree-wide (the old
    # basename match would have excluded "data.txt" everywhere).
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / ".gitignore").write_text("data.txt\n")
    (pkg / "data.txt").write_text("local\n")
    (tmp_path / "data.txt").write_text("root\n")
    tracker = FileTracker(tmp_path)
    tracker.take_baseline()

    manifest_paths = {entry["path"] for entry in tracker.current_manifest()}
    assert "data.txt" in manifest_paths
    assert "pkg/data.txt" in manifest_paths


def test_manifest_carries_paths_and_hashes(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x\n")
    tracker = FileTracker(tmp_path)
    tracker.take_baseline()

    manifest = tracker.current_manifest()

    assert len(manifest) == 1
    entry = manifest[0]
    assert entry["path"] == "a.txt"
    assert entry["size"] == (tmp_path / "a.txt").stat().st_size
    assert len(entry["content_hash"]) == 64  # sha256 hex


def test_to_dict_shape_matches_wire_contract(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("1\n")
    tracker = FileTracker(tmp_path)
    tracker.take_baseline()
    (tmp_path / "a.txt").write_text("2\n")

    payload = tracker.take_snapshot().to_dict()

    assert set(payload) >= {
        "snapshot_timestamp",
        "scan_duration_ms",
        "files_scanned",
        "files_changed",
        "patches",
    }
    assert set(payload["patches"][0]) == {"path", "status", "patch", "size_before", "size_after"}
