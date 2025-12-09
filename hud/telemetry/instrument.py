"""Simple instrumentation decorator for HUD tracing.

This module provides a lightweight @instrument decorator that records
function calls within the context of env.trace(). No OpenTelemetry required.

Usage:
    @hud.instrument
    async def my_function(arg1, arg2):
        ...

    # Within a trace context, calls are recorded
    async with env.trace("task") as tc:
        result = await my_function("a", "b")
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import logging
import time
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, TypeVar, overload

import pydantic_core

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from typing import ParamSpec

    P = ParamSpec("P")
    R = TypeVar("R")

logger = logging.getLogger(__name__)


def _serialize_value(value: Any, max_items: int = 10) -> Any:
    """Serialize a value for recording."""
    if isinstance(value, str | int | float | bool | type(None)):
        return value

    if isinstance(value, list | tuple):
        value = value[:max_items] if len(value) > max_items else value
    elif isinstance(value, dict) and len(value) > max_items:
        value = dict(list(value.items())[:max_items])

    try:
        json_bytes = pydantic_core.to_json(value, fallback=str)
        return json.loads(json_bytes)
    except Exception:
        return f"<{type(value).__name__}>"


@overload
def instrument(
    func: None = None,
    *,
    name: str | None = None,
    category: str = "function",
    record_args: bool = True,
    record_result: bool = True,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...


@overload
def instrument(
    func: Callable[P, R],
    *,
    name: str | None = None,
    category: str = "function",
    record_args: bool = True,
    record_result: bool = True,
) -> Callable[P, R]: ...


@overload
def instrument(
    func: Callable[P, Awaitable[R]],
    *,
    name: str | None = None,
    category: str = "function",
    record_args: bool = True,
    record_result: bool = True,
) -> Callable[P, Awaitable[R]]: ...


def instrument(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    category: str = "function",
    record_args: bool = True,
    record_result: bool = True,
) -> Callable[..., Any]:
    """Instrument a function to record spans within trace context.

    This decorator records function calls as spans, compatible with env.trace().

    Args:
        func: The function to instrument
        name: Custom span name (defaults to module.function)
        category: Span category (e.g., "agent", "tool", "function")
        record_args: Whether to record function arguments
        record_result: Whether to record function result

    Returns:
        The instrumented function

    Examples:
        @hud.instrument
        async def process_data(items: list[str]) -> dict:
            return {"count": len(items)}

        @hud.instrument(category="agent")
        async def call_model(messages: list) -> str:
            return await model.generate(messages)
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if hasattr(func, "_hud_instrumented"):
            return func

        func_module = getattr(func, "__module__", "unknown")
        func_name = getattr(func, "__name__", "unknown")
        func_qualname = getattr(func, "__qualname__", func_name)
        span_name = name or f"{func_module}.{func_qualname}"

        try:
            sig = inspect.signature(func)
        except (ValueError, TypeError):
            sig = None

        def _build_span(
            trace_id: str,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            start_time: str,
            end_time: str,
            duration_ms: float,
            result: Any = None,
            error: str | None = None,
        ) -> dict[str, Any]:
            """Build a span record."""
            attributes: dict[str, Any] = {
                "category": category,
                "function": func_qualname,
                "module": func_module,
                "duration_ms": duration_ms,
            }

            # Record arguments
            if record_args and sig:
                try:
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()
                    args_dict = {
                        k: _serialize_value(v)
                        for k, v in bound_args.arguments.items()
                        if k not in ("self", "cls")
                    }
                    if args_dict:
                        attributes["request"] = json.dumps(args_dict)
                except Exception as e:
                    logger.debug("Failed to serialize args: %s", e)

            # Record result
            if record_result and result is not None and error is None:
                try:
                    attributes["result"] = json.dumps(_serialize_value(result))
                except Exception as e:
                    logger.debug("Failed to serialize result: %s", e)

            # Record error
            if error:
                attributes["error"] = error

            return {
                "trace_id": trace_id,
                "span_id": uuid.uuid4().hex[:16],
                "name": span_name,
                "start_time": start_time,
                "end_time": end_time,
                "status_code": "ERROR" if error else "OK",
                "attributes": attributes,
            }

        def _get_trace_id() -> str | None:
            """Get trace_id from current trace context."""
            from hud.trace.context import get_current_trace_headers

            headers = get_current_trace_headers()
            if headers:
                return headers.get("Trace-Id")
            return None

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            trace_id = _get_trace_id()
            start_time = datetime.now(UTC).isoformat()
            start_perf = time.perf_counter()
            error: str | None = None
            result: Any = None

            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                raise
            finally:
                end_time = datetime.now(UTC).isoformat()
                duration_ms = (time.perf_counter() - start_perf) * 1000

                if trace_id:
                    _build_span(
                        trace_id, args, kwargs, start_time, end_time, duration_ms, result, error
                    )
                    logger.debug("Span: %s (%.2fms)", span_name, duration_ms)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            trace_id = _get_trace_id()
            start_time = datetime.now(UTC).isoformat()
            start_perf = time.perf_counter()
            error: str | None = None
            result: Any = None

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                raise
            finally:
                end_time = datetime.now(UTC).isoformat()
                duration_ms = (time.perf_counter() - start_perf) * 1000

                if trace_id:
                    _build_span(
                        trace_id, args, kwargs, start_time, end_time, duration_ms, result, error
                    )
                    logger.debug("Span: %s (%.2fms)", span_name, duration_ms)

        wrapper = async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        wrapper._hud_instrumented = True  # type: ignore[attr-defined]
        wrapper._hud_original = func  # type: ignore[attr-defined]

        return wrapper

    if func is None:
        return decorator
    return decorator(func)


__all__ = ["instrument"]
