"""OpenTelemetry-shaped span types emitted by the SDK.

Spans are the single wire format for platform ingest. The envelope carries
domain data as a pair of namespaced attributes: ``hud.schema`` tags how to
decode and ``hud.payload`` holds the one domain object the tag describes;
``@instrument`` debug spans carry neither, only generic diagnostics. Domain
schemas (e.g. the step stream) own their tag values and payload shapes —
this module knows none of them.
"""

from __future__ import annotations

import uuid
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from hud.utils.serialization import JsonObject

SpanKind: TypeAlias = Literal["INTERNAL", "SERVER", "CLIENT", "PRODUCER", "CONSUMER"]
SpanStatusCode: TypeAlias = Literal["UNSET", "OK", "ERROR"]

#: Attribute carrying the schema tag a backend serializer dispatches on.
SCHEMA_ATTRIBUTE = "hud.schema"
#: Attribute carrying the schema-tagged domain object — one payload per span.
#: Also the payload key on ``@instrument`` request/result span events.
PAYLOAD_ATTRIBUTE = "hud.payload"
#: Attribute carrying the task-run id the exporter groups uploads by.
TASK_RUN_ID_ATTRIBUTE = "hud.task_run_id"


class SpanEvent(BaseModel):
    """OpenTelemetry-style event attached to a span."""

    name: str
    timestamp: str
    attributes: JsonObject = Field(default_factory=dict)


class Span(BaseModel):
    """Fine-grained, OpenTelemetry-shaped span.

    Domain identifiers live in namespaced attributes such as
    ``hud.task_run_id``; ``trace_id``/``span_id`` describe the telemetry graph.
    """

    name: str
    trace_id: str = Field(pattern=r"^[0-9a-fA-F]{32}$")
    span_id: str = Field(pattern=r"^[0-9a-fA-F]{16}$")
    parent_span_id: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{16}$")
    kind: SpanKind = "INTERNAL"

    start_time: str  # ISO format
    end_time: str  # ISO format

    status_code: SpanStatusCode = "UNSET"
    status_message: str | None = None

    attributes: JsonObject = Field(default_factory=dict)
    events: list[SpanEvent] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


def new_span_id() -> str:
    """A fresh 16-hex span id."""
    return uuid.uuid4().hex[:16]


def normalize_trace_id(trace_id: str) -> str:
    """Map an arbitrary run identifier onto a 32-hex OTel trace id."""
    clean = trace_id.replace("-", "")
    if len(clean) == 32:
        try:
            int(clean, 16)
        except ValueError:
            pass
        else:
            return clean.lower()
    return uuid.uuid5(uuid.NAMESPACE_URL, f"hud.task_run:{trace_id}").hex


__all__ = [
    "PAYLOAD_ATTRIBUTE",
    "SCHEMA_ATTRIBUTE",
    "TASK_RUN_ID_ATTRIBUTE",
    "Span",
    "SpanEvent",
    "SpanKind",
    "SpanStatusCode",
    "new_span_id",
    "normalize_trace_id",
]
