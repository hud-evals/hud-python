"""General-purpose instrumentation decorator for HUD telemetry.

This module provides the instrument() decorator that users can use
to instrument any function with OpenTelemetry spans.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import logging
from typing import TYPE_CHECKING, Any, TypeVar, overload

import pydantic_core
from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from hud.otel import configure_telemetry, is_telemetry_configured
from hud.otel.context import get_current_task_run_id

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from typing import ParamSpec

    P = ParamSpec("P")
    R = TypeVar("R")

logger = logging.getLogger(__name__)


def _serialize_value(value: Any, max_items: int = 10) -> Any:
    """Serialize a value for span attributes.

    Uses pydantic_core.to_json for robust serialization of complex objects.

    Args:
        value: The value to serialize
        max_items: Maximum number of items for collections

    Returns:
        JSON-serializable version of the value
    """
    # Simple types pass through
    if isinstance(value, str | int | float | bool | type(None)):
        return value

    # For collections, we need to limit size first
    if isinstance(value, list | tuple):
        value = value[:max_items] if len(value) > max_items else value
    elif isinstance(value, dict) and len(value) > max_items:
        value = dict(list(value.items())[:max_items])

    # Use pydantic_core for serialization - it handles:
    # - Pydantic models (via model_dump)
    # - Dataclasses (via asdict)
    # - Bytes (encodes to string)
    # - Custom objects (via __dict__ or repr)
    # - Complex nested structures
    try:
        # Convert to JSON bytes then back to Python objects
        # This ensures we get JSON-serializable types
        json_bytes = pydantic_core.to_json(value, fallback=str)
        return json.loads(json_bytes)
    except Exception:
        # Fallback if pydantic_core fails somehow
        return f"<{type(value).__name__}>"


@overload
def instrument(
    func: None = None,
    *,
    name: str | None = None,
    span_type: str = "function",
    attributes: dict[str, Any] | None = None,
    record_args: bool = True,
    record_result: bool = True,
    span_kind: SpanKind = SpanKind.INTERNAL,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...


@overload
def instrument(
    func: Callable[P, R],
    *,
    name: str | None = None,
    span_type: str = "function",
    attributes: dict[str, Any] | None = None,
    record_args: bool = True,
    record_result: bool = True,
    span_kind: SpanKind = SpanKind.INTERNAL,
) -> Callable[P, R]: ...


@overload
def instrument(
    func: Callable[P, Awaitable[R]],
    *,
    name: str | None = None,
    span_type: str = "function",
    attributes: dict[str, Any] | None = None,
    record_args: bool = True,
    record_result: bool = True,
    span_kind: SpanKind = SpanKind.INTERNAL,
) -> Callable[P, Awaitable[R]]: ...


def instrument(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    span_type: str = "function",
    attributes: dict[str, Any] | None = None,
    record_args: bool = True,
    record_result: bool = True,
    span_kind: SpanKind = SpanKind.INTERNAL,
) -> Callable[..., Any]:
    """Instrument a function to emit OpenTelemetry spans.

    This decorator wraps any function to automatically create spans for
    observability. It works with both sync and async functions.

    Args:
        func: The function to instrument (when used without parentheses)
        name: Custom span name (defaults to fully qualified function name)
        span_type: The category for this span (e.g., "agent", "mcp", "database", "validation")
        attributes: Additional attributes to attach to every span
        record_args: Whether to record function arguments in the request field
        record_result: Whether to record function result in the result field
        span_kind: OpenTelemetry span kind (INTERNAL, CLIENT, SERVER, etc.)

    Returns:
        The instrumented function that emits spans

    Examples:
        # Basic usage - defaults to category="function"
        @hud.instrument
        async def process_data(items: list[str]) -> dict:
            return {"count": len(items)}

        # Custom category
        @hud.instrument(
            span_type="database",  # This becomes category="database"
            record_args=True,
            record_result=True
        )
        async def query_users(filter: dict) -> list[User]:
            return await db.find(filter)

        # Agent instrumentation
        @hud.instrument(
            span_type="agent",  # category="agent" gets special handling
            record_args=False,  # Don't record large message arrays
            record_result=True
        )
        async def get_model_response(self, messages: list) -> Response:
            return await self.model.complete(messages)

        # Instrument third-party functions
        import requests
        requests.get = hud.instrument(
            span_type="http",  # category="http"
            span_kind=SpanKind.CLIENT
        )(requests.get)

        # Conditional instrumentation
        if settings.enable_db_tracing:
            db.query = hud.instrument(db.query)
    """
    # Don't configure telemetry at decoration time - wait until first call
    # This allows users to configure alternative backends before importing agents

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Check if already instrumented
        if hasattr(func, "_hud_instrumented"):
            logger.debug("Function %s already instrumented, skipping", func.__name__)
            return func

        # Get function metadata
        func_module = getattr(func, "__module__", "unknown")
        func_name = getattr(func, "__name__", "unknown")
        func_qualname = getattr(func, "__qualname__", func_name)

        # Determine span name
        span_name = name or f"{func_module}.{func_qualname}"

        # Get function signature for argument parsing
        try:
            sig = inspect.signature(func)
        except (ValueError, TypeError):
            sig = None

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Ensure telemetry is configured (lazy initialization)
            # Only configure with defaults if user hasn't configured it yet
            if not is_telemetry_configured():
                configure_telemetry()

            tracer = trace.get_tracer("hud-sdk")

            # Build span attributes
            span_attrs = {
                "category": span_type,  # span_type IS the category
                "function.module": func_module,
                "function.name": func_name,
                "function.qualname": func_qualname,
            }

            # Add custom attributes
            if attributes:
                span_attrs.update(attributes)

            # Add current task_run_id if available
            task_run_id = get_current_task_run_id()
            if task_run_id:
                span_attrs["hud.task_run_id"] = task_run_id

            # Record function arguments if requested
            if record_args and sig:
                try:
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()

                    # Serialize arguments (with safety limits)
                    args_dict = {}
                    for param_name, value in bound_args.arguments.items():
                        try:
                            # Skip 'self' and 'cls' parameters
                            if param_name in ("self", "cls"):
                                continue

                            args_dict[param_name] = _serialize_value(value)
                        except Exception:
                            args_dict[param_name] = "<serialization_error>"

                    if args_dict:
                        args_json = json.dumps(args_dict)
                        span_attrs["function.arguments"] = args_json
                        # Always set generic request field for consistency
                        span_attrs["request"] = args_json
                except Exception as e:
                    logger.debug("Failed to record function arguments: %s", e)

            with tracer.start_as_current_span(
                span_name,
                kind=span_kind,
                attributes=span_attrs,
            ) as span:
                try:
                    # Execute the function
                    result = await func(*args, **kwargs)

                    # Record result if requested
                    if record_result:
                        try:
                            serialized = _serialize_value(result)
                            result_json = json.dumps(serialized)
                            span.set_attribute("function.result", result_json)
                            # Always set generic result field for consistency
                            span.set_attribute("result", result_json)

                            # Also set result type for complex objects
                            if not isinstance(
                                result, str | int | float | bool | type(None) | list | tuple | dict
                            ):
                                span.set_attribute("function.result_type", type(result).__name__)
                        except Exception as e:
                            logger.debug("Failed to record function result: %s", e)

                    span.set_status(Status(StatusCode.OK))
                    return result

                except Exception as e:
                    # Record exception and set error status
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Ensure telemetry is configured (lazy initialization)
            # Only configure with defaults if user hasn't configured it yet
            if not is_telemetry_configured():
                configure_telemetry()

            tracer = trace.get_tracer("hud-sdk")

            # Build span attributes (same as async)
            span_attrs = {
                "category": span_type,  # span_type IS the category
                "function.module": func_module,
                "function.name": func_name,
                "function.qualname": func_qualname,
            }

            if attributes:
                span_attrs.update(attributes)

            task_run_id = get_current_task_run_id()
            if task_run_id:
                span_attrs["hud.task_run_id"] = task_run_id

            # Record function arguments if requested
            if record_args and sig:
                try:
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()

                    args_dict = {}
                    for param_name, value in bound_args.arguments.items():
                        try:
                            if param_name in ("self", "cls"):
                                continue

                            args_dict[param_name] = _serialize_value(value)
                        except Exception:
                            args_dict[param_name] = "<serialization_error>"

                    if args_dict:
                        args_json = json.dumps(args_dict)
                        span_attrs["function.arguments"] = args_json
                        # Always set generic request field for consistency
                        span_attrs["request"] = args_json
                except Exception as e:
                    logger.debug("Failed to record function arguments: %s", e)

            with tracer.start_as_current_span(
                span_name,
                kind=span_kind,
                attributes=span_attrs,
            ) as span:
                try:
                    # Execute the function
                    result = func(*args, **kwargs)

                    # Record result if requested
                    if record_result:
                        try:
                            serialized = _serialize_value(result)
                            result_json = json.dumps(serialized)
                            span.set_attribute("function.result", result_json)
                            # Always set generic result field for consistency
                            span.set_attribute("result", result_json)

                            # Also set result type for complex objects
                            if not isinstance(
                                result, str | int | float | bool | type(None) | list | tuple | dict
                            ):
                                span.set_attribute("function.result_type", type(result).__name__)
                        except Exception as e:
                            logger.debug("Failed to record function result: %s", e)

                    span.set_status(Status(StatusCode.OK))
                    return result

                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

        # Choose wrapper based on function type
        wrapper = async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

        # Mark as instrumented
        wrapper._hud_instrumented = True  # type: ignore[attr-defined]
        wrapper._hud_original = func  # type: ignore[attr-defined]

        return wrapper

    # Handle usage with or without parentheses
    if func is None:
        # Called with arguments: @instrument(name="foo")
        return decorator
    else:
        # Called without arguments: @instrument
        return decorator(func)
