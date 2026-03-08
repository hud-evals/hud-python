"""Helpers for transporting structured chat reply metadata over A2A."""

from __future__ import annotations

import json
import uuid
from typing import Any

from a2a.types import Artifact, Part, TaskArtifactUpdateEvent, TextPart

from hud.types import Trace

REPLY_METADATA_TYPE = "hud_reply_metadata"


def build_reply_metadata(trace: Trace) -> dict[str, Any] | None:
    """Build a structured metadata envelope from a chat trace."""
    if not trace.citations:
        return None

    return {
        "type": REPLY_METADATA_TYPE,
        "citations": trace.citations,
        "data": None,
    }


def build_reply_metadata_event(
    *,
    context_id: str,
    task_id: str,
    trace: Trace,
) -> TaskArtifactUpdateEvent | None:
    """Convert chat trace metadata into a single A2A artifact event."""
    payload = build_reply_metadata(trace)
    if payload is None:
        return None

    return TaskArtifactUpdateEvent(
        context_id=context_id,
        task_id=task_id,
        append=False,
        last_chunk=True,
        artifact=Artifact(
            artifact_id=str(uuid.uuid4()),
            name=REPLY_METADATA_TYPE,
            parts=[Part(root=TextPart(text=json.dumps(payload)))],
        ),
    )
