"""Custom OpenTelemetry exporter that sends spans to the existing HUD telemetry
HTTP endpoint (/trace/<id>/telemetry-upload).

The exporter groups spans by ``hud.task_run_id`` baggage / attribute so we keep
exactly the same semantics the old async worker in ``hud.telemetry.exporter``
implemented.

This exporter is *synchronous* (derives from :class:`SpanExporter`).  We rely on
``hud.shared.make_request_sync`` which already contains retry & auth logic.
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from collections import defaultdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from mcp.types import ClientRequest, ServerResult
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from pydantic import BaseModel, ConfigDict, Field

from hud.shared import make_request_sync
import httpx
from hud.types import TraceStep as HudSpanAttributes

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import ReadableSpan

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class HudSpan(BaseModel):
    """A telemetry span ready for export."""

    name: str
    trace_id: str = Field(pattern=r"^[0-9a-fA-F]{32}$")
    span_id: str = Field(pattern=r"^[0-9a-fA-F]{16}$")
    parent_span_id: str | None = Field(None, pattern=r"^[0-9a-fA-F]{16}$")

    start_time: str  # ISO format
    end_time: str  # ISO format

    status_code: str  # "UNSET", "OK", "ERROR"
    status_message: str | None = None

    attributes: HudSpanAttributes
    exceptions: list[dict[str, Any]] | None = None

    model_config = ConfigDict(extra="forbid")


def extract_span_attributes(
    attrs: dict[str, Any], method_name: str | None = None, span_name: str | None = None
) -> HudSpanAttributes:
    """Extract and parse span attributes into typed model.

    This handles:
    - Detecting span type (MCP vs Agent)
    - Renaming verbose OpenTelemetry semantic conventions
    - Parsing JSON strings to MCP types
    """
    # Start with core attributes - map to TraceStep field names
    result_attrs = {
        "task_run_id": attrs.get(
            "hud.task_run_id"
        ),  # TraceStep expects task_run_id, not hud.task_run_id
        "job_id": attrs.get("hud.job_id"),  # TraceStep expects job_id, not hud.job_id
        "type": attrs.get("span.kind", "CLIENT"),  # TraceStep expects type, not span.kind
    }

    # Determine span type based on presence of agent or MCP attributes
    # Note: The input attrs might already have "category" set
    existing_category = attrs.get("category")

    if existing_category:
        # Use the explicit category if provided
        result_attrs["category"] = existing_category
    elif span_name and span_name.startswith("agent."):
        # Legacy support for spans named "agent.*"
        result_attrs["category"] = "agent"
    else:
        result_attrs["category"] = "mcp"  # Default to MCP

    # No special processing needed for different categories
    # The backend will handle them based on the category field

    # Add method_name and request_id for MCP spans
    if result_attrs["category"] == "mcp":
        if method_name:
            result_attrs["method_name"] = method_name
        # Check for request_id with and without semconv_ai prefix
        request_id = attrs.get("semconv_ai.mcp.request_id") or attrs.get("mcp.request.id")
        if request_id:
            result_attrs["request_id"] = request_id

    # Parse input/output - check both with and without semconv_ai prefix
    input_str = attrs.get("semconv_ai.traceloop.entity.input") or attrs.get(
        "traceloop.entity.input"
    )
    output_str = attrs.get("semconv_ai.traceloop.entity.output") or attrs.get(
        "traceloop.entity.output"
    )

    logger.debug(
        "Category: %s, has input: %s, has output: %s",
        result_attrs.get("category"),
        bool(input_str),
        bool(output_str),
    )

    # Check for direct request/result attributes first
    if "request" in attrs and not result_attrs.get("request"):
        req = attrs["request"]
        if isinstance(req, str):
            with contextlib.suppress(json.JSONDecodeError):
                req = json.loads(req)
        result_attrs["request"] = req

    if "result" in attrs and not result_attrs.get("result"):
        res = attrs["result"]
        if isinstance(res, str):
            with contextlib.suppress(json.JSONDecodeError):
                res = json.loads(res)
        result_attrs["result"] = res

    # Process input/output from MCP instrumentation
    if input_str and not result_attrs.get("request"):
        try:
            input_data = json.loads(input_str) if isinstance(input_str, str) else input_str

            # For MCP category, try to parse as ClientRequest to extract the root
            if result_attrs["category"] == "mcp" and isinstance(input_data, dict):
                try:
                    if "method" in input_data and "params" in input_data:
                        client_request = ClientRequest.model_validate(input_data)
                        result_attrs["request"] = client_request.root
                    else:
                        result_attrs["request"] = input_data
                except Exception:
                    result_attrs["request"] = input_data
            else:
                # For all other categories, just store the data
                result_attrs["request"] = input_data
        except Exception as e:
            logger.debug("Failed to parse request JSON: %s", e)

    if output_str and not result_attrs.get("result"):
        try:
            output_data = json.loads(output_str) if isinstance(output_str, str) else output_str

            # For MCP category, try to parse as ServerResult to extract the root
            if result_attrs["category"] == "mcp" and isinstance(output_data, dict):
                # Check for error
                if "error" in output_data:
                    result_attrs["mcp_error"] = True
                try:
                    server_result = ServerResult.model_validate(output_data)
                    result_attrs["result"] = server_result.root
                    # Check for isError in the result
                    if getattr(server_result.root, "isError", False):
                        result_attrs["mcp_error"] = True
                except Exception:
                    result_attrs["result"] = output_data
            else:
                # For all other categories, just store the data
                result_attrs["result"] = output_data
        except Exception as e:
            logger.debug("Failed to parse result JSON: %s", e)

    # Don't include the verbose attributes or ones we've already processed
    exclude_keys = {
        "hud.task_run_id",
        "hud.job_id",
        "span.kind",
        "semconv_ai.mcp.method_name",
        "mcp.method.name",  # Also exclude non-prefixed version
        "semconv_ai.mcp.request_id",
        "mcp.request.id",  # Also exclude non-prefixed version
        "semconv_ai.traceloop.entity.input",
        "semconv_ai.traceloop.entity.output",
        "traceloop.entity.input",  # Also exclude non-prefixed versions
        "traceloop.entity.output",
        "mcp_request",  # Exclude to prevent overwriting parsed values
        "mcp_result",  # Exclude to prevent overwriting parsed values
        "request",  # Exclude to prevent overwriting parsed values
        "result",  # Exclude to prevent overwriting parsed values
        "category",  # Already handled above
    }

    # Add any extra attributes
    for key, value in attrs.items():
        if key not in exclude_keys:
            result_attrs[key] = value  # noqa: PERF403

    logger.debug(
        """Final result_attrs before creating HudSpanAttributes:
        request=%s,
        result=%s""",
        result_attrs.get("request"),
        result_attrs.get("result"),
    )
    return HudSpanAttributes(**result_attrs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ts_ns_to_iso(ts_ns: int) -> str:
    """Convert a ``Span`` timestamp (nanoseconds) to ISO-8601 string."""
    # OpenTelemetry times are epoch nanoseconds
    dt = datetime.fromtimestamp(ts_ns / 1_000_000_000, tz=UTC)
    return dt.isoformat().replace("+00:00", "Z")


def _span_to_dict(span: ReadableSpan) -> dict[str, Any]:
    """Convert an OpenTelemetry span to a dict using typed models."""

    attrs = dict(span.attributes or {})

    # Extract method name from span name if not in attributes
    # Check both with and without semconv_ai prefix
    raw_method = attrs.get("semconv_ai.mcp.method_name") or attrs.get("mcp.method.name")
    method_name: str | None = None
    if isinstance(raw_method, str):
        method_name = raw_method
    if method_name is None and isinstance(span.name, str) and span.name.endswith(".mcp"):
        method_name = span.name[:-4]  # Remove .mcp suffix

    # Create typed attributes
    typed_attrs = extract_span_attributes(attrs, method_name, str(span.name))

    # Record span kind as extra attribute (TraceStep allows extras)
    try:
        typed_attrs.span_kind = span.kind.name  # type: ignore[attr-defined]
    except Exception:
        logger.warning("Failed to set span kind attribute")

    # Build typed span
    # Guard context/parent/timestamps
    context = getattr(span, "context", None)
    trace_id_hex = (
        format(context.trace_id, "032x") if context and hasattr(context, "trace_id") else "0" * 32
    )
    span_id_hex = (
        format(context.span_id, "016x") if context and hasattr(context, "span_id") else "0" * 16
    )
    parent = getattr(span, "parent", None)
    parent_id_hex = (
        format(parent.span_id, "016x") if parent and hasattr(parent, "span_id") else None
    )
    start_ns = span.start_time or 0
    end_ns = span.end_time or start_ns

    typed_span = HudSpan(
        name=span.name,
        trace_id=trace_id_hex,
        span_id=span_id_hex,
        parent_span_id=parent_id_hex,
        start_time=_ts_ns_to_iso(int(start_ns)),
        end_time=_ts_ns_to_iso(int(end_ns)),
        status_code=span.status.status_code.name if span.status else "UNSET",
        status_message=span.status.description if span.status else None,
        attributes=typed_attrs,
        exceptions=None,
    )

    # Add error information if present
    if span.events:
        exceptions = []
        exceptions = [
            {
                "timestamp": _ts_ns_to_iso(event.timestamp),
                "attributes": dict(event.attributes or {}),
            }
            for event in span.events
        ]
        if exceptions:
            typed_span.exceptions = exceptions

    # Convert to dict for export
    return typed_span.model_dump(mode="json", by_alias=True, exclude_none=True)


# ---------------------------------------------------------------------------
# Exporter
# ---------------------------------------------------------------------------


class HudSpanExporter(SpanExporter):
    """Exporter that forwards spans to HUD backend using existing endpoint."""

    def __init__(self, *, telemetry_url: str, api_key: str) -> None:
        super().__init__()
        self._telemetry_url = telemetry_url.rstrip("/")
        self._api_key = api_key
        # Use a short-timeout client so exporter never blocks shutdown for long
        # These limits are generous but bounded; telemetry payloads are small
        self._client = httpx.Client(
            timeout=15.0,
            limits=httpx.Limits(
                max_connections=128, max_keepalive_connections=64, keepalive_expiry=5.0
            ),
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def export(self, spans: list[ReadableSpan]) -> SpanExportResult:  # type: ignore[override]
        if not spans:
            return SpanExportResult.SUCCESS

        # Group spans by hud.task_run_id attribute
        grouped: dict[str, list[ReadableSpan]] = defaultdict(list)
        for span in spans:
            run_id = span.attributes.get("hud.task_run_id") if span.attributes else None
            if not run_id:
                # Skip spans that are outside HUD traces
                continue
            grouped[str(run_id)].append(span)

        # Send each group synchronously (retry inside make_request_sync)
        for run_id, span_batch in grouped.items():
            try:
                url = f"{self._telemetry_url}/trace/{run_id}/telemetry-upload"
                telemetry_spans = [_span_to_dict(s) for s in span_batch]
                # Include current step count in metadata
                metadata = {}
                # Get the HIGHEST step count from the batch (most recent)
                step_count = 0
                for span in span_batch:
                    if span.attributes and "hud.step_count" in span.attributes:
                        current_step = span.attributes["hud.step_count"]
                        if isinstance(current_step, int) and current_step > step_count:
                            step_count = current_step

                payload = {
                    "metadata": metadata,
                    "telemetry": telemetry_spans,
                }

                # Only include step_count if we found any steps
                if step_count > 0:
                    payload["step_count"] = step_count

                logger.debug("HUD exporter sending %d spans to %s", len(span_batch), url)
                make_request_sync(
                    method="POST",
                    url=url,
                    json=payload,
                    api_key=self._api_key,
                    max_retries=1,
                    retry_delay=0.5,
                    client=self._client,
                )
            except Exception as exc:
                logger.exception("HUD exporter failed to send spans for task %s: %s", run_id, exc)
                # If *any* group fails we return FAILURE so the OTEL SDK can retry
                return SpanExportResult.FAILURE

        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:  # type: ignore[override]
        # Close HTTP client
        try:
            self._client.close()
        except Exception:
            pass

    def force_flush(self, timeout_millis: int | None = None) -> bool:  # type: ignore[override]
        # No-op and return immediately. Export is synchronous and handled by
        # BatchSpanProcessor; this exporter does not buffer internally.
        return True
