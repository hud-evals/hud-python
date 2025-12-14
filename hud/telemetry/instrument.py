"""Instrumentation decorator for HUD telemetry.

This module provides a lightweight @instrument decorator that records
function calls and sends them to the HUD telemetry backend.

Usage:
    @hud.instrument
    async def my_function(arg1, arg2):
        ...

    # Within an eval context, calls are recorded and sent to HUD
    async with env.eval("task") as ctx:
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

from hud.telemetry.exporter import queue_span
from hud.types import TraceStep


def _get_trace_id() -> str | None:
    """Lazy import to avoid circular dependency with eval.context."""
    from hud.eval.context import get_current_trace_id

    return get_current_trace_id()


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


def _now_iso() -> str:
    """Get current time as ISO-8601 string."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _normalize_trace_id(trace_id: str) -> str:
    """Normalize trace_id to 32-character hex string."""
    clean = trace_id.replace("-", "")
    return clean[:32].ljust(32, "0")


@overload
def instrument(
    func: None = None,
    *,
    name: str | None = None,
    category: str = "function",
    span_type: str | None = None,
    record_args: bool = True,
    record_result: bool = True,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...


@overload
def instrument(
    func: Callable[P, R],
    *,
    name: str | None = None,
    category: str = "function",
    span_type: str | None = None,
    record_args: bool = True,
    record_result: bool = True,
) -> Callable[P, R]: ...


@overload
def instrument(
    func: Callable[P, Awaitable[R]],
    *,
    name: str | None = None,
    category: str = "function",
    span_type: str | None = None,
    record_args: bool = True,
    record_result: bool = True,
) -> Callable[P, Awaitable[R]]: ...


def instrument(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    category: str = "function",
    span_type: str | None = None,
    record_args: bool = True,
    record_result: bool = True,
) -> Callable[..., Any]:
    """Instrument a function to record spans within eval context.

    This decorator records function calls as spans and sends them to the HUD API.

    Args:
        func: The function to instrument
        name: Custom span name (defaults to module.function)
        category: Span category (e.g., "agent", "tool", "function", "mcp")
        span_type: Alias for category (deprecated, use category instead)
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
    effective_category = span_type if span_type is not None else category

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
            task_run_id: str,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            start_time: str,
            end_time: str,
            result: Any = None,
            error: str | None = None,
        ) -> dict[str, Any]:
            """Build a HudSpan-compatible span record."""
            # Build attributes using TraceStep
            attributes = TraceStep(
                task_run_id=task_run_id,
                category=effective_category,
                type="CLIENT",
                start_timestamp=start_time,
                end_timestamp=end_time,
            )

            # Record arguments as request
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
                        attributes.request = args_dict
                except Exception as e:
                    logger.debug("Failed to serialize args: %s", e)

            # Record result
            if record_result and result is not None and error is None:
                try:
                    attributes.result = _serialize_value(result)
                except Exception as e:
                    logger.debug("Failed to serialize result: %s", e)

            # Build span
            span_id = uuid.uuid4().hex[:16]
            span = {
                "name": span_name,
                "trace_id": _normalize_trace_id(task_run_id),
                "span_id": span_id,
                "parent_span_id": None,
                "start_time": start_time,
                "end_time": end_time,
                "status_code": "ERROR" if error else "OK",
                "status_message": error,
                "attributes": attributes.model_dump(mode="json", exclude_none=True),
                "exceptions": [{"message": error}] if error else None,
            }
            return span

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            task_run_id = _get_trace_id()
            start_time = _now_iso()
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
                end_time = _now_iso()
                duration_ms = (time.perf_counter() - start_perf) * 1000

                if task_run_id:
                    span = _build_span(
                        task_run_id, args, kwargs, start_time, end_time, result, error
                    )
                    queue_span(span)
                    logger.debug("Span: %s (%.2fms)", span_name, duration_ms)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            task_run_id = _get_trace_id()
            start_time = _now_iso()
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
                end_time = _now_iso()
                duration_ms = (time.perf_counter() - start_perf) * 1000

                if task_run_id:
                    span = _build_span(
                        task_run_id, args, kwargs, start_time, end_time, result, error
                    )
                    queue_span(span)
                    logger.debug("Span: %s (%.2fms)", span_name, duration_ms)

        wrapper = async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        wrapper._hud_instrumented = True  # type: ignore[attr-defined]
        wrapper._hud_original = func  # type: ignore[attr-defined]

        return wrapper

    if func is None:
        return decorator
    return decorator(func)


__all__ = [
    "instrument",
]
