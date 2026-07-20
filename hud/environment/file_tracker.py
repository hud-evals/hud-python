"""Filesystem change tracker for a workspace directory.

Snapshots a directory tree and produces unified diffs (patches) between
snapshots, so a rollout can record exactly what changed on disk over time. The
tracker is pure computation over a local ``root``; ``serve_file_tracking`` wraps
one tracker in the small ``filetracking/1`` framed-JSON adapter used by
workspaces.

Ported from the orchestrator sidecar's ``/proc``-scanning tracker, with the
Kubernetes coupling removed (it scans an injected ``root`` instead of
``/proc/{pid}/root``) and a non-overridable secrets deny-list added: paths that
look like credentials are tracked at the metadata tier only — their content is
never read, hashed-for-fingerprint only, and never emitted as a diff.
"""

from __future__ import annotations

import asyncio
import base64
import difflib
import fnmatch
import hashlib
import logging
import mimetypes
import os
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from .utils import error, read_frame, reply, send_frame

LOGGER = logging.getLogger("hud.environment.file_tracker")

#: Noise paths excluded from tracking entirely (build output, caches, VCS internals).
DEFAULT_EXCLUDE_PATTERNS: tuple[str, ...] = (
    "node_modules/",
    ".venv/",
    "__pycache__/",
    "*.pyc",
    ".cache/",
    ".npm/",
    ".git",
    ".hg",
    ".svn",
    ".jj",
    # HUD CLI state written into env dirs (e.g. deploy's .hud/config.json).
    ".hud/",
    "*.so",
    "*.o",
    "*.a",
)

#: Credential-shaped paths tracked at the metadata tier only — content is never
#: read or emitted, regardless of any capture policy. Not overridable.
SECRET_DENY_PATTERNS: tuple[str, ...] = (
    ".env",
    ".env.*",
    ".gitconfig",
    ".gitmodules",
    "*.pem",
    "id_*",
    "*_key",
    "*_key.*",
    "credentials",
    "credentials.*",
    ".netrc",
    ".git-credentials",
    ".ssh/",
    ".aws/",
)

#: Skip files larger than this during scanning (default 10 MB).
DEFAULT_MAX_FILE_SIZE: int = 10 * 1024 * 1024

#: Artifacts worth preserving as downloads. Common source edits are already
#: represented by diffs; HTML is included because it is often the rendered result.
DEFAULT_CAPTURE_EXTENSIONS: tuple[str, ...] = (
    ".docx",
    ".htm",
    ".html",
    ".jpeg",
    ".jpg",
    ".pdf",
    ".png",
    ".pptx",
    ".webp",
    ".xlsm",
    ".xlsx",
)

CAPTURE_CONTENT_TYPES: dict[str, str] = {
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".htm": "text/html",
    ".html": "text/html",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".webp": "image/webp",
    ".xlsm": "application/vnd.ms-excel.sheet.macroEnabled.12",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


@dataclass(frozen=True)
class FileEntry:
    """Snapshot of a single file's state."""

    rel_path: str
    size: int
    mtime_ns: int
    content_hash: str
    # Cached text content for diffing. None = binary, unreadable, over-budget, or
    # a redacted secret. Stored as a tuple so unchanged files share it across
    # snapshots without re-reading.
    lines: tuple[str, ...] | None = None
    # A credential-shaped path: tracked for change detection, never for content.
    redacted: bool = False


@dataclass
class ScanBudget:
    """Per-scan counter of new file-content bytes read into memory.

    Threaded through the recursive walk so each ``_scan()`` has its own counter
    (thread-safe when scans run concurrently via ``run_in_executor``).
    """

    bytes_loaded: int = 0


@dataclass
class Snapshot:
    """A point-in-time snapshot of the tracked filesystem."""

    timestamp: float
    files: dict[str, FileEntry] = field(default_factory=dict)
    scan_duration_ms: float = 0.0


@dataclass
class PatchEntry:
    """A single file's diff between two snapshots."""

    rel_path: str
    status: str  # "added", "modified", "deleted"
    patch: str  # unified diff text (placeholder for binary/redacted/over-limit)
    size_before: int = 0
    size_after: int = 0


@dataclass
class DiffResult:
    """Result of diffing two snapshots."""

    patches: list[PatchEntry]
    snapshot_timestamp: float
    scan_duration_ms: float
    files_scanned: int
    files_changed: int
    truncated: bool = False  # True if the diff payload was capped by _MAX_DIFF_BYTES
    # Paths dropped by the byte cap. Internal-only (never serialized): the
    # tracker uses it to keep those files pending so they re-diff on a later
    # poll instead of being silently baselined away.
    skipped_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON transport (the filetracking/1 wire shape)."""
        result: dict[str, Any] = {
            "snapshot_timestamp": self.snapshot_timestamp,
            "scan_duration_ms": round(self.scan_duration_ms, 2),
            "files_scanned": self.files_scanned,
            "files_changed": self.files_changed,
            "patches": [
                {
                    "path": p.rel_path,
                    "status": p.status,
                    "patch": p.patch,
                    "size_before": p.size_before,
                    "size_after": p.size_after,
                }
                for p in self.patches
            ],
        }
        if self.truncated:
            result["truncated"] = True
        return result


class FileTracker:
    """Tracks file changes under a directory ``root`` via snapshot diffing.

    Usage::

        tracker = FileTracker("/workspace")
        tracker.take_baseline()  # at session start
        diff = tracker.take_snapshot()  # later — diff since the last snapshot
    """

    # Maximum bytes of NEW file content to read into memory per scan. Unchanged
    # files reuse the previous snapshot's cached lines (zero cost). Once
    # exhausted, new/modified files are still recorded (hash + metadata) with
    # ``lines=None`` so they show as changed with a placeholder rather than a
    # full diff. 50 MB raw is ~100 MB in Python text objects.
    _MAX_SCAN_CONTENT_BYTES: int = 50 * 1024 * 1024

    # Hard cap on total serialized diff payload. Patches that would push the
    # cumulative total past this are skipped (smaller ones still pack in).
    _MAX_DIFF_BYTES: int = 50 * 1024 * 1024

    # Per-file size cap for diff generation. Larger files get a placeholder
    # instead of a full unified diff, so ``difflib`` never allocates unbounded.
    _MAX_DIFF_FILE_BYTES: int = 1 * 1024 * 1024

    # Artifact capture is stricter than scan/diff limits because it emits base64
    # into telemetry. Keep the raw total comfortably below the default 25 MB span request limit
    # (after base64 overhead).
    _MAX_CAPTURE_FILES: int = 20
    _MAX_CAPTURE_FILE_BYTES: int = 5 * 1024 * 1024
    _MAX_CAPTURE_TOTAL_BYTES: int = 15 * 1024 * 1024

    def __init__(
        self,
        root: Path | str,
        *,
        exclude_patterns: tuple[str, ...] = DEFAULT_EXCLUDE_PATTERNS,
        capture_extensions: tuple[str, ...] = DEFAULT_CAPTURE_EXTENSIONS,
        honor_gitignore: bool = True,
        max_file_size: int = DEFAULT_MAX_FILE_SIZE,
        secret_deny_patterns: tuple[str, ...] = SECRET_DENY_PATTERNS,
    ) -> None:
        self._root = Path(root).resolve()
        self._exclude_patterns = exclude_patterns
        self._capture_extensions = tuple(ext.lower() for ext in capture_extensions)
        self._secret_deny_patterns = secret_deny_patterns
        self._honor_gitignore = honor_gitignore
        self._max_file_size = max_file_size

        self._previous_snapshot: Snapshot | None = None
        self._baseline_snapshot: Snapshot | None = None

        self._gitignore_patterns: list[str] = []
        self._gitignore_loaded = False

        LOGGER.info(
            "FileTracker initialized: root=%s, excludes=%d, max_size=%dMB",
            self._root,
            len(self._exclude_patterns),
            self._max_file_size // (1024 * 1024),
        )

    # ─── public API ───────────────────────────────────────────────────

    def take_baseline(self) -> Snapshot:
        """Take the initial baseline snapshot. Call once at session start."""
        snapshot = self._scan()
        self._baseline_snapshot = snapshot
        self._previous_snapshot = snapshot
        LOGGER.info(
            "Baseline snapshot: %d files in %.1fms",
            len(snapshot.files),
            snapshot.scan_duration_ms,
        )
        return snapshot

    def advance_baseline(self) -> None:
        """Re-scan and update the tracking baseline WITHOUT producing a diff.

        Used after scenario setup (which writes many files that are not agent
        edits), so subsequent diffs and final artifact capture start clean.
        """
        prev_count = len(self._previous_snapshot.files) if self._previous_snapshot else 0
        snapshot = self._scan()
        self._baseline_snapshot = snapshot
        self._previous_snapshot = snapshot
        if len(snapshot.files) != prev_count:
            LOGGER.info(
                "file diff baseline advanced: %d files (was %d)", len(snapshot.files), prev_count
            )

    def take_snapshot(self) -> DiffResult:
        """Scan and diff against the previous snapshot, then advance the baseline."""
        if self._previous_snapshot is None:
            LOGGER.warning("No baseline snapshot; taking one now")
            baseline = self.take_baseline()
            return DiffResult(
                patches=[],
                snapshot_timestamp=baseline.timestamp,
                scan_duration_ms=baseline.scan_duration_ms,
                files_scanned=len(baseline.files),
                files_changed=0,
            )

        return self._diff_current(self._scan())

    def flush_changes(self) -> dict[str, Any]:
        """Return the final diff plus changed artifact payloads in one scan."""
        if self._previous_snapshot is None:
            LOGGER.warning("No baseline snapshot; taking one now")
            baseline = self.take_baseline()
            return {
                "diff": DiffResult(
                    patches=[],
                    snapshot_timestamp=baseline.timestamp,
                    scan_duration_ms=baseline.scan_duration_ms,
                    files_scanned=len(baseline.files),
                    files_changed=0,
                ).to_dict(),
                "capture": self._capture_artifacts(baseline),
            }

        current = self._scan()
        capture = self._capture_artifacts(current)
        return {"diff": self._diff_current(current).to_dict(), "capture": capture}

    def _diff_current(self, current: Snapshot) -> DiffResult:
        if self._previous_snapshot is None:
            raise RuntimeError("cannot diff without a baseline")

        diff = self._diff(self._previous_snapshot, current)
        # Keep budget-skipped files pending: revert each to its previous state in
        # the new baseline (drop added-but-skipped files) so the next scan still
        # sees them as changed and re-diffs them, rather than dropping the change.
        for path in diff.skipped_paths:
            old_entry = self._previous_snapshot.files.get(path)
            if old_entry is None:
                current.files.pop(path, None)
            else:
                current.files[path] = old_entry
        self._previous_snapshot = current

        if diff.files_changed > 0:
            LOGGER.info(
                "Snapshot diff: %d files changed (%d scanned) in %.1fms",
                diff.files_changed,
                diff.files_scanned,
                diff.scan_duration_ms,
            )
        return diff

    def current_manifest(self) -> list[dict[str, Any]]:
        """The latest file manifest: ``[{path, size, content_hash}, ...]``.

        The full-state anchor a ``snapshot`` request returns — paths + hashes,
        never content (so it is safe regardless of capture policy).
        """
        snapshot = self._previous_snapshot or self._baseline_snapshot
        if snapshot is None:
            return []
        return [
            {"path": e.rel_path, "size": e.size, "content_hash": e.content_hash}
            for e in sorted(snapshot.files.values(), key=lambda e: e.rel_path)
        ]

    def _capture_artifacts(self, current: Snapshot) -> dict[str, Any]:
        changed: list[tuple[str, FileEntry]] = []
        if self._baseline_snapshot is not None:
            for path in sorted(current.files):
                entry = current.files[path]
                baseline_entry = self._baseline_snapshot.files.get(path)
                if baseline_entry is None:
                    changed.append(("added", entry))
                elif baseline_entry.content_hash != entry.content_hash:
                    changed.append(("modified", entry))

        captured: list[dict[str, Any]] = []
        files_eligible = 0
        files_skipped = 0
        total_bytes = 0
        truncated = False

        for status, entry in changed:
            suffix = PurePosixPath(entry.rel_path).suffix.lower()
            if entry.redacted or suffix not in self._capture_extensions:
                continue
            files_eligible += 1

            if (
                len(captured) >= self._MAX_CAPTURE_FILES
                or entry.size > self._MAX_CAPTURE_FILE_BYTES
                or total_bytes + entry.size > self._MAX_CAPTURE_TOTAL_BYTES
            ):
                files_skipped += 1
                truncated = True
                continue

            file = self._capture_file(status, entry)
            if file is None:
                files_skipped += 1
                continue

            total_bytes += int(file["size"])
            captured.append(file)

        payload = {
            "snapshot_timestamp": current.timestamp,
            "scan_duration_ms": round(current.scan_duration_ms, 2),
            "files_scanned": len(current.files),
            "files_changed": len(changed),
            "files_eligible": files_eligible,
            "files_captured": len(captured),
            "files_skipped": files_skipped,
            "files": captured,
        }
        if truncated:
            payload["truncated"] = True
        return payload

    def _capture_file(self, status: str, entry: FileEntry) -> dict[str, Any] | None:
        raw_path = self._root / entry.rel_path
        try:
            if raw_path.is_symlink():
                return None
        except OSError:
            return None

        path = raw_path.resolve()
        try:
            path.relative_to(self._root)
        except ValueError:
            return None

        try:
            if not path.is_file():
                return None
            stat = path.stat()
            if stat.st_size != entry.size or stat.st_size > self._MAX_CAPTURE_FILE_BYTES:
                return None
            with path.open("rb") as f:
                data = f.read()
        except (PermissionError, OSError):
            return None

        if len(data) != entry.size:
            return None

        suffix = PurePosixPath(entry.rel_path).suffix.lower()
        content_type = CAPTURE_CONTENT_TYPES.get(
            suffix,
            mimetypes.guess_type(entry.rel_path)[0] or "application/octet-stream",
        )
        return {
            "path": entry.rel_path,
            "status": status,
            "size": entry.size,
            "content_hash": hashlib.sha256(data).hexdigest(),
            "content_type": content_type,
            "file": {
                "type": "file",
                "data": base64.b64encode(data).decode("ascii"),
                "media_type": content_type,
            },
        }

    # ─── scanning ─────────────────────────────────────────────────────

    def _scan(self) -> Snapshot:
        start = time.monotonic()
        files: dict[str, FileEntry] = {}
        budget = ScanBudget()

        if self._honor_gitignore and not self._gitignore_loaded:
            self._gitignore_patterns = self._collect_gitignore_patterns()
            self._gitignore_loaded = True

        exclude_patterns = self._exclude_patterns + tuple(self._gitignore_patterns)
        if self._root.is_dir():
            self._walk_directory(str(self._root), files, budget, exclude_patterns)

        return Snapshot(
            timestamp=time.time(),
            files=files,
            scan_duration_ms=(time.monotonic() - start) * 1000,
        )

    def _walk_directory(
        self,
        abs_dir: str,
        files: dict[str, FileEntry],
        budget: ScanBudget,
        exclude_patterns: tuple[str, ...],
    ) -> None:
        try:
            scanner = os.scandir(abs_dir)
        except (PermissionError, OSError) as exc:
            LOGGER.debug("Cannot scan %s: %s", abs_dir, exc)
            return

        root_str = str(self._root)
        with scanner:
            for entry in scanner:
                try:
                    # Path relative to root, posix-style, no leading slash.
                    rel = os.path.relpath(entry.path, root_str).replace(os.sep, "/")
                    is_dir = entry.is_dir(follow_symlinks=False)

                    if self._matches(rel, is_dir, exclude_patterns):
                        continue

                    if is_dir:
                        self._walk_directory(entry.path, files, budget, exclude_patterns)
                        continue
                    if not entry.is_file(follow_symlinks=False):
                        continue

                    try:
                        stat = entry.stat(follow_symlinks=False)
                    except (PermissionError, OSError):
                        continue
                    if stat.st_size > self._max_file_size:
                        continue

                    files[rel] = self._build_entry(entry.path, rel, stat, budget)
                except (PermissionError, OSError, ValueError):
                    continue

    def _build_entry(
        self, abs_path: str, rel: str, stat: os.stat_result, budget: ScanBudget
    ) -> FileEntry:
        """Build a ``FileEntry``, reusing cached content when unchanged."""
        prev = self._previous_snapshot.files.get(rel) if self._previous_snapshot else None
        if prev is not None and prev.mtime_ns == stat.st_mtime_ns and prev.size == stat.st_size:
            # Unchanged — reuse cached hash + lines (no allocation).
            return FileEntry(
                rel_path=rel,
                size=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
                content_hash=prev.content_hash,
                lines=prev.lines,
                redacted=prev.redacted,
            )

        if self._matches(rel, False, self._secret_deny_patterns):
            # Credential-shaped: detect change via fingerprint, never read content.
            return FileEntry(
                rel_path=rel,
                size=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
                content_hash=f"redacted:{stat.st_size}:{stat.st_mtime_ns}",
                lines=None,
                redacted=True,
            )
        if stat.st_size > self._MAX_DIFF_FILE_BYTES:
            content_hash = f"overlimit:{stat.st_size}:{stat.st_mtime_ns}"
            lines = None
        elif budget.bytes_loaded + stat.st_size > self._MAX_SCAN_CONTENT_BYTES:
            content_hash = f"budget_exceeded:{stat.st_size}:{stat.st_mtime_ns}"
            lines = None
        else:
            content_hash = self._hash_file(abs_path, stat.st_size)
            lines = self._read_lines(abs_path)
            budget.bytes_loaded += stat.st_size

        return FileEntry(
            rel_path=rel,
            size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            content_hash=content_hash,
            lines=lines,
        )

    @staticmethod
    def _matches(rel: str, is_dir: bool, patterns: tuple[str, ...]) -> bool:
        path = f"/{rel}"
        name = PurePosixPath(path).name
        for pattern in patterns:
            if pattern.endswith("/"):
                dir_name = pattern.rstrip("/")
                if (is_dir and name == dir_name) or f"/{dir_name}/" in path:
                    return True
            elif fnmatch.fnmatch(name, pattern):
                return True
        return False

    def _collect_gitignore_patterns(self) -> list[str]:
        """Read the root ``.gitignore`` only.

        Patterns are matched tree-wide (see :meth:`_matches`), which is only
        sound for ignore rules that are themselves root-scoped. Nested
        per-directory ``.gitignore`` files are intentionally not loaded: applying
        a subdirectory's rules globally would wrongly exclude (or fail to
        exclude) paths elsewhere in the tree. The built-in
        ``DEFAULT_EXCLUDE_PATTERNS`` already drop the dominant noise
        (``node_modules/``, ``.venv/``, ``__pycache__/``, build artifacts), so a
        root-only honor is enough for a telemetry noise filter.
        """
        root_gitignore = self._root / ".gitignore"
        if not root_gitignore.is_file():
            return []
        patterns: list[str] = []
        try:
            with root_gitignore.open("r", encoding="utf-8", errors="replace") as f:
                for raw in f:
                    line = raw.strip()
                    # Skip comments, blanks, and negations (unsupported).
                    if not line or line.startswith(("#", "!")):
                        continue
                    patterns.append(line.lstrip("/"))
        except (PermissionError, OSError) as exc:
            LOGGER.debug("Cannot read gitignore %s: %s", root_gitignore, exc)
        if patterns:
            LOGGER.info("Loaded %d gitignore patterns", len(patterns))
        return patterns

    @staticmethod
    def _read_lines(path: str) -> tuple[str, ...] | None:
        """Read a file as text lines; None if binary/unreadable."""
        try:
            with open(path, encoding="utf-8", errors="strict") as f:
                return tuple(f.read().splitlines())
        except UnicodeDecodeError:
            return None
        except (PermissionError, OSError, FileNotFoundError):
            return None

    @staticmethod
    def _hash_file(path: str, size: int) -> str:
        """SHA-256 of a file's content (a content-address; the diff dedup key)."""
        h = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
        except (PermissionError, OSError):
            return f"unreadable:{size}"
        return h.hexdigest()

    # ─── diffing ──────────────────────────────────────────────────────

    def _diff(self, old: Snapshot, new: Snapshot) -> DiffResult:
        """Unified diffs between two snapshots, smallest-first within a byte cap."""
        changed: list[tuple[str, FileEntry | None, FileEntry | None, str]] = []
        for path in set(old.files) | set(new.files):
            old_entry = old.files.get(path)
            new_entry = new.files.get(path)
            if old_entry is None and new_entry is not None:
                changed.append((path, old_entry, new_entry, "added"))
            elif old_entry is not None and new_entry is None:
                changed.append((path, old_entry, new_entry, "deleted"))
            elif (
                old_entry is not None
                and new_entry is not None
                and old_entry.content_hash != new_entry.content_hash
            ):
                changed.append((path, old_entry, new_entry, "modified"))

        # Smallest first so many small agent edits pack in before the budget is
        # eaten by a few large files.
        changed.sort(key=lambda c: max(c[1].size if c[1] else 0, c[2].size if c[2] else 0))

        patches: list[PatchEntry] = []
        total_bytes = 0
        skipped_paths: list[str] = []
        for path, old_entry, new_entry, status in changed:
            size_before = old_entry.size if old_entry else 0
            size_after = new_entry.size if new_entry else 0
            patch_text = self._patch_text(path, old_entry, new_entry, size_before, size_after)

            patch_bytes = len(patch_text.encode("utf-8", errors="replace"))
            if total_bytes + patch_bytes > self._MAX_DIFF_BYTES:
                skipped_paths.append(path)
                continue
            total_bytes += patch_bytes
            patches.append(
                PatchEntry(
                    rel_path=path,
                    status=status,
                    patch=patch_text,
                    size_before=size_before,
                    size_after=size_after,
                )
            )

        total_changed = len(patches) + len(skipped_paths)
        if skipped_paths:
            LOGGER.warning(
                "Diff size cap (%d MB): %d/%d changed files included, %d skipped",
                self._MAX_DIFF_BYTES // (1024 * 1024),
                len(patches),
                total_changed,
                len(skipped_paths),
            )
        return DiffResult(
            patches=patches,
            snapshot_timestamp=new.timestamp,
            scan_duration_ms=new.scan_duration_ms,
            files_scanned=len(new.files),
            files_changed=total_changed,
            truncated=len(skipped_paths) > 0,
            skipped_paths=skipped_paths,
        )

    def _patch_text(
        self,
        path: str,
        old_entry: FileEntry | None,
        new_entry: FileEntry | None,
        size_before: int,
        size_after: int,
    ) -> str:
        if (old_entry and old_entry.redacted) or (new_entry and new_entry.redacted):
            return f"Secret file changed (content redacted): {path}\n"
        if size_before > self._MAX_DIFF_FILE_BYTES or size_after > self._MAX_DIFF_FILE_BYTES:
            return f"File too large to diff ({size_before} -> {size_after} bytes): {path}\n"
        old_lines = old_entry.lines if old_entry else ()
        new_lines = new_entry.lines if new_entry else ()
        if old_lines is None or new_lines is None:
            return f"Binary file changed: {path}\n"
        return "\n".join(
            difflib.unified_diff(
                list(old_lines),
                list(new_lines),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
                lineterm="",
            )
        )


class _FileTrackingHandler:
    """Per-server dispatcher; one tracker shared across connections under a lock."""

    def __init__(self, tracker: FileTracker) -> None:
        self._tracker = tracker
        self._lock = asyncio.Lock()

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        loop = asyncio.get_running_loop()
        try:
            while (msg := await read_frame(reader)) is not None:
                msg_id = msg.get("id")
                method = msg.get("method", "")
                try:
                    async with self._lock:
                        if method == "diff":
                            diff = await loop.run_in_executor(None, self._tracker.take_snapshot)
                            result = diff.to_dict()
                        elif method == "snapshot":
                            manifest = self._tracker.current_manifest()
                            result = {"files": manifest, "files_scanned": len(manifest)}
                        elif method == "advance":
                            await loop.run_in_executor(None, self._tracker.advance_baseline)
                            result = {"advanced": True}
                        elif method == "flush":
                            result = await loop.run_in_executor(None, self._tracker.flush_changes)
                        else:
                            raise ValueError(f"unknown filetracking method: {method!r}")
                except Exception as exc:  # never tear the connection down on one bad call
                    LOGGER.debug("filetracking %s failed: %s", method, exc)
                    if isinstance(msg_id, int):
                        await send_frame(writer, error(msg_id, -32000, str(exc)))
                    continue
                if isinstance(msg_id, int):
                    await send_frame(writer, reply(msg_id, result))
        except (ConnectionError, OSError):
            pass
        finally:
            writer.close()


async def serve_file_tracking(
    tracker: FileTracker, host: str = "127.0.0.1", port: int = 0
) -> asyncio.Server:
    """Bind a ``filetracking/1`` server for ``tracker``. Caller drives the port."""
    handler = _FileTrackingHandler(tracker)
    server = await asyncio.start_server(handler.handle, host, port)
    sock = server.sockets[0].getsockname()
    LOGGER.info("filetracking bound on %s:%s", sock[0], sock[1])
    return server


__all__ = [
    "DEFAULT_CAPTURE_EXTENSIONS",
    "DEFAULT_EXCLUDE_PATTERNS",
    "DEFAULT_MAX_FILE_SIZE",
    "SECRET_DENY_PATTERNS",
    "DiffResult",
    "FileEntry",
    "FileTracker",
    "PatchEntry",
    "Snapshot",
    "serve_file_tracking",
]
