"""Central configuration for OpenTelemetry inside HUD SDK.

This file is responsible for
1. creating the global ``TracerProvider``
2. attaching span processors (HUD enrichment, batch + exporter)
3. activating the community MCP instrumentation so that *every* MCP
   request/response/notification is traced automatically.

It is *idempotent*: calling :func:`configure_telemetry` more than once
returns the same provider and does nothing.
"""

from __future__ import annotations

import logging
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from hud.settings import settings

from .collector import enable_trace_collection, install_collector
from .exporters import HudSpanExporter
from .instrumentation import install_mcp_instrumentation
from .processors import HudEnrichmentProcessor

logger = logging.getLogger(__name__)

# Global singleton provider so multiple calls do not create duplicates
_TRACER_PROVIDER: TracerProvider | None = None


def is_telemetry_configured() -> bool:
    """Check if telemetry has been configured."""
    return _TRACER_PROVIDER is not None


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

    # HUD exporter (only if enabled and API key is available)
    if settings.telemetry_enabled and settings.api_key:
        # Use the HudSpanExporter directly (it now handles async context internally)
        exporter = HudSpanExporter(
            telemetry_url=settings.hud_telemetry_url, api_key=settings.api_key
        )

        # Batch exports for efficiency while maintaining reasonable real-time visibility
        provider.add_span_processor(
            BatchSpanProcessor(
                exporter,
                schedule_delay_millis=1000,  # Export every 5 seconds (less frequent)
                max_queue_size=16384,  # Larger queue for high-volume scenarios
                max_export_batch_size=512,  # Larger batches (fewer uploads)
                export_timeout_millis=30000,
            )
        )
    elif settings.telemetry_enabled and not settings.api_key and not enable_otlp:
        # Error if no exporters are configured
        raise ValueError(
            "No telemetry backend configured. Either:\n"
            "1. Set HUD_API_KEY environment variable for HUD telemetry (https://hud.ai)\n"
            "2. Use enable_otlp=True with configure_telemetry() for alternative backends (e.g., Jaeger)\n"  # noqa: E501
        )
    elif not settings.telemetry_enabled:
        logger.info("HUD telemetry disabled via HUD_TELEMETRY_ENABLED=false")

    # OTLP exporter (optional - for standard OTel viewers)
    if enable_otlp:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            otlp_config = {}
            if otlp_endpoint:
                otlp_config["endpoint"] = otlp_endpoint
                # Default to HTTP endpoint if not specified
                if not otlp_endpoint.startswith(("http://", "https://")):
                    otlp_config["endpoint"] = f"http://{otlp_endpoint}/v1/traces"
            else:
                # Default HTTP endpoint
                otlp_config["endpoint"] = "http://localhost:4318/v1/traces"

            if otlp_headers:
                otlp_config["headers"] = otlp_headers

            otlp_exporter = OTLPSpanExporter(**otlp_config)
            provider.add_span_processor(
                BatchSpanProcessor(
                    otlp_exporter,
                    schedule_delay_millis=1000,
                    max_queue_size=16384,
                    max_export_batch_size=512,
                    export_timeout_millis=30000,
                )
            )
            logger.info("OTLP HTTP exporter enabled - endpoint: %s", otlp_config["endpoint"])
        except ImportError:
            logger.warning(
                "OTLP export requested but opentelemetry-exporter-otlp-proto-http not installed. "
                "Install with: pip install 'hud-python[agent]'"
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

    # Agent instrumentation now handled by @hud.instrument decorators
    logger.debug("OpenTelemetry configuration completed")

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
