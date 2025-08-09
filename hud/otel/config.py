from __future__ import annotations

"""Central configuration for OpenTelemetry inside HUD SDK.

This file is responsible for
1. creating the global ``TracerProvider``
2. attaching span processors (HUD enrichment, batch + exporter)
3. activating the community MCP instrumentation so that *every* MCP
   request/response/notification is traced automatically.

It is *idempotent*: calling :func:`configure_telemetry` more than once
returns the same provider and does nothing.
"""

import logging
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from hud.settings import settings

from .collector import enable_trace_collection, install_collector
from .exporters import HudSpanExporter
from .instrumentation import auto_instrument_agents, install_mcp_instrumentation
from .processors import HudEnrichmentProcessor

logger = logging.getLogger(__name__)

# Global singleton provider so multiple calls do not create duplicates
_TRACER_PROVIDER: TracerProvider | None = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def configure_telemetry(
    *,
    service_name: str = "hud-sdk",
    service_version: str | None = None,
    environment: str | None = None,
    extra_resource_attributes: dict[str, Any] | None = None,
    enable_otlp: bool = False,
    otlp_endpoint: str | None = None,
    otlp_headers: dict[str, str] | None = None,
    enable_collection: bool = True,
) -> TracerProvider:
    """Initialise OpenTelemetry for the current Python process.

    It is safe to call this in every entry-point; the provider will only
    be created once.
    """
    global _TRACER_PROVIDER

    if _TRACER_PROVIDER is not None:
        return _TRACER_PROVIDER

    # ------------------------------------------------------------------
    # 1. Resource (identity of this service)
    # ------------------------------------------------------------------
    res_attrs: dict[str, Any] = {
        "service.name": service_name,
        "telemetry.sdk.name": "hud-otel",
        "telemetry.sdk.language": "python",
    }
    if service_version:
        res_attrs["service.version"] = service_version
    if environment:
        res_attrs["deployment.environment"] = environment
    if extra_resource_attributes:
        res_attrs.update(extra_resource_attributes)

    resource = Resource.create(res_attrs)

    # ------------------------------------------------------------------
    # 2. Provider
    # ------------------------------------------------------------------
    provider = TracerProvider(resource=resource)
    _TRACER_PROVIDER = provider

    # ------------------------------------------------------------------
    # 3. Processors / exporters
    # ------------------------------------------------------------------
    provider.add_span_processor(HudEnrichmentProcessor())

    # HUD exporter (default)
    if settings.telemetry_enabled:
        if not settings.api_key:
            raise ValueError("API key is required for telemetry")
        exporter = HudSpanExporter(base_url=settings.base_url, api_key=settings.api_key)
        provider.add_span_processor(BatchSpanProcessor(exporter))
    else:
        logger.info("Telemetry disabled via settings, spans will not be exported")

    # OTLP exporter (optional - for standard OTel viewers)
    if enable_otlp:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

            otlp_config = {}
            if otlp_endpoint:
                otlp_config["endpoint"] = otlp_endpoint
            if otlp_headers:
                otlp_config["headers"] = otlp_headers

            otlp_exporter = OTLPSpanExporter(**otlp_config)
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            logger.info(f"OTLP exporter enabled - endpoint: {otlp_endpoint or 'localhost:4317'}")
        except ImportError:
            logger.warning(
                "OTLP export requested but opentelemetry-exporter-otlp-proto-grpc not installed. "
                "Install with: pip install opentelemetry-exporter-otlp-proto-grpc"
            )

    # ------------------------------------------------------------------
    # 4. Activate provider and instrumentation
    # ------------------------------------------------------------------
    trace.set_tracer_provider(provider)
    install_mcp_instrumentation(provider)

    # Install in-memory collector if requested
    if enable_collection:
        install_collector()
        enable_trace_collection(True)
        logger.debug("In-memory trace collection enabled")

    # Auto-instrument agent classes
    auto_instrument_agents()
    logger.debug("Agent auto-instrumentation completed")

    logger.debug("OpenTelemetry configured (provider id=%s)", id(provider))
    return provider


def shutdown_telemetry() -> None:
    """Flush and shutdown the global provider (if configured)."""
    global _TRACER_PROVIDER
    if _TRACER_PROVIDER is None:
        return
    _TRACER_PROVIDER.shutdown()  # type: ignore[arg-type]
    _TRACER_PROVIDER = None
    logger.debug("OpenTelemetry shutdown complete")
