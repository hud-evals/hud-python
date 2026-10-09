"""Read what the SDK's telemetry exported, from the local span files."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from hud.settings import settings

STEP_SCHEMA = "hud.step.v1"


def spans(trace_id: str | None, span_dir: Path | None = None) -> list[dict[str, Any]]:
    """Every span exported for the run ``trace_id``, in start order.

    Spans land in ``<span_dir>/<trace id as 32 hex>.jsonl``; the span directory
    is the SDK's own (``HUD_TELEMETRY_LOCAL_DIR``, else ``~/.hud/spans`` while
    uploads are off).
    """
    if trace_id is None:
        raise ValueError("the run has no trace id, so it exported no spans")
    directory = span_dir or Path(settings.span_dir or "")
    path = directory / f"{uuid.UUID(trace_id).hex}.jsonl"
    if not path.exists():
        return []
    records = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]
    return sorted(records, key=lambda span: span.get("start_time", ""))


def steps(trace_id: str | None, span_dir: Path | None = None) -> list[dict[str, Any]]:
    """The step payloads of a run, in order."""
    return [
        span["attributes"]["hud.payload"]
        for span in spans(trace_id, span_dir)
        if span.get("attributes", {}).get("hud.schema") == STEP_SCHEMA
    ]
