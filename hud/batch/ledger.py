"""RowResult and the Ledger: one light record per row, in SQLite, doubling as resume state."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .dataset import Cursor

if TYPE_CHECKING:
    from collections.abc import Iterator

RowStatus = Literal["completed", "error", "cancelled"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS results (
    slug TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL,
    row_id TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    status TEXT NOT NULL,
    reward REAL,
    evaluation TEXT NOT NULL,
    error TEXT,
    trace_id TEXT,
    reported INTEGER NOT NULL,
    steps INTEGER NOT NULL,
    duration_s REAL NOT NULL,
    reader INTEGER NOT NULL,
    file INTEGER NOT NULL,
    row_group INTEGER NOT NULL,
    row INTEGER NOT NULL,
    finished_at TEXT NOT NULL,
    trace TEXT
);
CREATE INDEX IF NOT EXISTS results_batch ON results (batch_id, status);
CREATE TABLE IF NOT EXISTS checkpoints (
    batch_id TEXT NOT NULL,
    reader INTEGER NOT NULL,
    file INTEGER NOT NULL,
    row_group INTEGER NOT NULL,
    row INTEGER NOT NULL,
    PRIMARY KEY (batch_id, reader)
);
"""


class RowResult(BaseModel):
    """What one row produced: its grade, where it came from, and whether a trace exists."""

    model_config = ConfigDict(frozen=True)

    batch_id: str
    row_id: str
    slug: str
    attempt: int = Field(ge=1)
    status: RowStatus
    reward: float | None
    evaluation: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    #: The run's trace id. It names a platform trace only when ``reported``.
    trace_id: str | None = None
    reported: bool = False
    steps: int = 0
    duration_s: float = 0.0
    reader: int = 0
    cursor: Cursor
    finished_at: str
    #: The full local trace of a failed row, when the batch keeps error traces.
    trace: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return self.status == "completed" and self.error is None


def _row_result(record: sqlite3.Row) -> RowResult:
    return RowResult(
        batch_id=record["batch_id"],
        row_id=record["row_id"],
        slug=record["slug"],
        attempt=record["attempt"],
        status=record["status"],
        reward=record["reward"],
        evaluation=json.loads(record["evaluation"]),
        error=record["error"],
        trace_id=record["trace_id"],
        reported=bool(record["reported"]),
        steps=record["steps"],
        duration_s=record["duration_s"],
        reader=record["reader"],
        cursor=Cursor(file=record["file"], row_group=record["row_group"], row=record["row"]),
        finished_at=record["finished_at"],
        trace=json.loads(record["trace"]) if record["trace"] is not None else None,
    )


class Ledger:
    """Per-row results and per-reader checkpoints for batch jobs, in one SQLite file.

    The slug (``batch_id:row_id``) is the primary key, so recording a retry
    supersedes the earlier attempt and a resumed job skips what is already
    done. ``":memory:"`` keeps a ledger for the life of the process.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = path
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(_SCHEMA)

    def close(self) -> None:
        self._db.close()

    # ─── results ─────────────────────────────────────────────────────────

    def record(self, result: RowResult) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO results VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                result.slug,
                result.batch_id,
                result.row_id,
                result.attempt,
                result.status,
                result.reward,
                json.dumps(result.evaluation, default=str),
                result.error,
                result.trace_id,
                int(result.reported),
                result.steps,
                result.duration_s,
                result.reader,
                result.cursor.file,
                result.cursor.row_group,
                result.cursor.row,
                result.finished_at,
                json.dumps(result.trace, default=str) if result.trace is not None else None,
            ),
        )

    def get(self, slug: str) -> RowResult | None:
        record = self._db.execute("SELECT * FROM results WHERE slug = ?", (slug,)).fetchone()
        return _row_result(record) if record is not None else None

    def results(self, batch_id: str, *, status: RowStatus | None = None) -> Iterator[RowResult]:
        query = "SELECT * FROM results WHERE batch_id = ?"
        params: tuple[Any, ...] = (batch_id,)
        if status is not None:
            query += " AND status = ?"
            params += (status,)
        query += " ORDER BY file, row_group, row"
        for record in self._db.execute(query, params):
            yield _row_result(record)

    def attempts(self, batch_id: str) -> dict[str, tuple[int, RowStatus]]:
        """``slug -> (attempt, status)`` for every recorded row of the batch."""
        return {
            record["slug"]: (record["attempt"], record["status"])
            for record in self._db.execute(
                "SELECT slug, attempt, status FROM results WHERE batch_id = ?", (batch_id,)
            )
        }

    def summary(self, batch_id: str) -> dict[str, Any]:
        """Row counts by status and the mean reward of completed rows."""
        counts = {
            record["status"]: record["n"]
            for record in self._db.execute(
                "SELECT status, COUNT(*) AS n FROM results WHERE batch_id = ? GROUP BY status",
                (batch_id,),
            )
        }
        mean = self._db.execute(
            "SELECT AVG(reward) FROM results WHERE batch_id = ? AND status = 'completed'",
            (batch_id,),
        ).fetchone()[0]
        return {
            "completed": counts.get("completed", 0),
            "error": counts.get("error", 0),
            "cancelled": counts.get("cancelled", 0),
            "mean_reward": mean,
        }

    # ─── checkpoints ─────────────────────────────────────────────────────

    def checkpoints(self, batch_id: str) -> dict[int, Cursor]:
        return {
            record["reader"]: Cursor(
                file=record["file"], row_group=record["row_group"], row=record["row"]
            )
            for record in self._db.execute(
                "SELECT reader, file, row_group, row FROM checkpoints WHERE batch_id = ?",
                (batch_id,),
            )
        }

    def checkpoint(self, batch_id: str, reader: int, cursor: Cursor) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO checkpoints VALUES (?, ?, ?, ?, ?)",
            (batch_id, reader, cursor.file, cursor.row_group, cursor.row),
        )

    # ─── export ──────────────────────────────────────────────────────────

    def export(self, batch_id: str, path: str | Path) -> Path:
        """Write the batch's results to ``.jsonl`` or ``.parquet``."""
        target = Path(path)
        rows = (
            result.model_dump(mode="json", exclude={"trace"}) for result in self.results(batch_id)
        )
        if target.suffix == ".jsonl":
            with target.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, default=str) + "\n")
        elif target.suffix == ".parquet":
            import pyarrow as pa
            import pyarrow.parquet as pq

            flat = [
                {**row, "evaluation": json.dumps(row["evaluation"], default=str)} for row in rows
            ]
            pq.write_table(pa.Table.from_pylist(flat), target)
        else:
            raise ValueError(f"unsupported export format {target.suffix!r}; use .jsonl or .parquet")
        return target


__all__ = ["Ledger", "RowResult", "RowStatus"]
