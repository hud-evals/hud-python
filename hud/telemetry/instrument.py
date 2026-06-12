"""``@instrument``: OTel-shaped debug spans for any function.

Records one span per call — name, timing, status, args/result as span events —
and queues it for export. Spans from this decorator are diagnostics: they carry
no domain schema tag, so the platform surfaces them in debug tooling only. The
canonical run record is the ``Step`` stream (``hud.types``), not this.

Usage:
    @hud.instrument
    async def my_function(arg1, arg2):
        ...
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import time
from typing import TYPE_CHECKING, Any, TypeVar, cast, overload

from hud.telemetry.context import get_current_trace_id
from hud.telemetry.exporter import queue_span
from hud.telemetry.span import (
    PAYLOAD_ATTRIBUTE,
    TASK_RUN_ID_ATTRIBUTE,
    Span,
    SpanEvent,
    new_span_id,
    normalize_trace_id,
)
from hud.utils.serialization import JsonObject, JsonValue, json_safe_value
from hud.utils.time import now_iso

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from typing import ParamSpec

    P = ParamSpec("P")
    R = TypeVar("R")

logger = logging.getLogger(__name__)


def _serialize_value(value: Any, max_items: int = 10) -> JsonValue:
    """Serialize a value for recording (domain-agnostic; pydantic models dump)."""
    if isinstance(value, str | int | float | bool | type(None)):
        return value

    if isinstance(value, list):
        items = cast("list[Any]", value)
        value = items[:max_items] if len(items) > max_items else items
    elif isinstance(value, tuple):
        items = list(cast("tuple[Any, ...]", value))
        value = items[:max_items] if len(items) > max_items else items
    elif isinstance(value, dict):
        mapping = cast("dict[Any, Any]", value)
        value = dict(list(mapping.items())[:max_items]) if len(mapping) > max_items else mapping

    return cast("JsonValue", json_safe_value(value))


@overload
def instrument(
    func: None = None,
    *,
    name: str | None = None,
    record_args: bool = True,
    record_result: bool = True,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...


@overload
def instrument(
    func: Callable[P, Awaitable[R]],
    *,
    name: str | None = None,
    record_args: bool = True,
    record_result: bool = True,
) -> Callable[P, Awaitable[R]]: ...


@overload
def instrument(
    func: Callable[P, R],
    *,
    name: str | None = None,
    record_args: bool = True,
    record_result: bool = True,
) -> Callable[P, R]: ...


def instrument(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    record_args: bool = True,
    record_result: bool = True,
) -> Callable[..., Any]:
    """Record each call of ``func`` as an OTel-shaped debug span.

    Args:
        func: The function to instrument.
        name: Custom span name (defaults to ``module.qualname``).
        record_args: Record bound call arguments as a ``hud.request`` event.
        record_result: Record the return value as a ``hud.result`` event.
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
            task_run_id: str,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            start_time: str,
            end_time: str,
            result: Any = None,
            error: str | None = None,
        ) -> Span:
            events: list[SpanEvent] = []

            if record_args and sig:
                try:
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()
                    args_dict: JsonObject = {
                        k: _serialize_value(v)
                        for k, v in bound_args.arguments.items()
                        if k not in ("self", "cls")
                    }
                    if args_dict:
                        events.append(
                            SpanEvent(
                                name="hud.request",
                                timestamp=start_time,
                                attributes={PAYLOAD_ATTRIBUTE: args_dict},
                            )
                        )
                except Exception as exc:
                    logger.debug("Failed to serialize args: %s", exc)

            if record_result and result is not None and error is None:
                try:
                    events.append(
                        SpanEvent(
                            name="hud.result",
                            timestamp=end_time,
                            attributes={PAYLOAD_ATTRIBUTE: _serialize_value(result)},
                        )
                    )
                except Exception as exc:
                    logger.debug("Failed to serialize result: %s", exc)

            if error is not None:
                events.append(
                    SpanEvent(
                        name="exception",
                        timestamp=end_time,
                        attributes={"exception.message": error},
                    )
                )

            return Span(
                name=span_name,
                trace_id=normalize_trace_id(task_run_id),
                span_id=new_span_id(),
                start_time=start_time,
                end_time=end_time,
                status_code="ERROR" if error else "OK",
                status_message=error,
                attributes={
                    TASK_RUN_ID_ATTRIBUTE: task_run_id,
                    "code.function": func_qualname,
                    "code.namespace": func_module,
                },
                events=events,
            )

        def _emit_span(
            task_run_id: str | None,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            start_time: str,
            start_perf: float,
            result: Any,
            error: str | None,
        ) -> None:
            if task_run_id is None:
                return
            end_time = now_iso()
            span = _build_span(task_run_id, args, kwargs, start_time, end_time, result, error)
            queue_span(span.model_dump(mode="json"))
            logger.debug("Span: %s (%.2fms)", span_name, (time.perf_counter() - start_perf) * 1000)

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            task_run_id = get_current_trace_id()
            start_time = now_iso()
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
                _emit_span(task_run_id, args, kwargs, start_time, start_perf, result, error)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            task_run_id = get_current_trace_id()
            start_time = now_iso()
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
                _emit_span(task_run_id, args, kwargs, start_time, start_perf, result, error)

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
